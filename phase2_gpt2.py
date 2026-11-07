"""
Phase 2: GPT-2 from scratch.

  1. Load pretrained weights and compare outputs with HuggingFace's GPT-2
  2. Parameter count per component
  3. FLOPs per forward pass: formula vs torch.profiler vs FlopCounterMode
  4. Latency and peak memory over sequence lengths / batch sizes, FP32 and FP16
  5. Perplexity on the WikiText-2 test set

Usage: python phase2_gpt2.py
"""

import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode

from common import get_device, set_seed, make_results_dir, save_json, count_parameters
from gpt2_model import GPT2, GPT2Config, load_huggingface_weights
from evaluation import load_wikitext2_tokens, evaluate_perplexity


# ---------------------------------------------------------------------------
# Step 1: compare with HuggingFace
# ---------------------------------------------------------------------------

def greedy_generate(model, input_ids, num_new_tokens, is_huggingface):
    generated = input_ids
    with torch.no_grad():
        for step in range(num_new_tokens):
            if is_huggingface:
                logits = model(generated).logits
            else:
                logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


def step1_verify_against_huggingface(device):
    print("\n========== STEP 1: our GPT-2 vs HuggingFace GPT-2 ==========")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(device)
    hf_model.eval()

    our_model = GPT2(GPT2Config())
    load_huggingface_weights(our_model, hf_model.state_dict())
    our_model = our_model.to(device)
    our_model.eval()

    text = ("The transformer architecture was introduced in 2017 and quickly replaced "
            "recurrent neural networks for most language tasks, because")
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        our_logits = our_model(input_ids)

    max_difference = (hf_logits - our_logits).abs().max().item()
    mean_difference = (hf_logits - our_logits).abs().mean().item()
    top1_agreement = (hf_logits.argmax(dim=-1) == our_logits.argmax(dim=-1)).float().mean().item()
    print("Logits shape:", tuple(our_logits.shape))
    print("Max |difference| of logits : %.2e" % max_difference)
    print("Mean |difference| of logits: %.2e" % mean_difference)
    print("Top-1 next-token agreement : %.1f%%" % (top1_agreement * 100))

    hf_generated = greedy_generate(hf_model, input_ids, 20, is_huggingface=True)
    our_generated = greedy_generate(our_model, input_ids, 20, is_huggingface=False)
    hf_text = tokenizer.decode(hf_generated[0])
    our_text = tokenizer.decode(our_generated[0])
    print("HuggingFace generates:", repr(hf_text[len(text):]))
    print("Our model generates  :", repr(our_text[len(text):]))
    print("Identical generated text:", hf_text == our_text)

    del hf_model
    torch.cuda.empty_cache()
    results = {
        "max_abs_logit_difference": max_difference,
        "mean_abs_logit_difference": mean_difference,
        "top1_agreement": top1_agreement,
        "identical_generation": hf_text == our_text,
        "generated_text": our_text,
    }
    return our_model, results


# ---------------------------------------------------------------------------
# Step 2: parameters
# ---------------------------------------------------------------------------

def component_of(parameter_name):
    if parameter_name.startswith("token_embedding"):
        return "token embedding (also lm_head, tied)"
    if parameter_name.startswith("position_embedding"):
        return "position embedding"
    if ".attention." in parameter_name:
        return "attention (q, k, v, out)"
    if ".mlp." in parameter_name:
        return "MLP (fc_in, fc_out)"
    if "norm" in parameter_name:
        return "LayerNorms"
    return "other"


def step2_count_parameters(model, results_dir):
    print("\n========== STEP 2: parameters per component ==========")
    counts = {}
    for name, parameter in model.named_parameters():
        group = component_of(name)
        if group not in counts:
            counts[group] = 0
        counts[group] = counts[group] + parameter.numel()

    total = count_parameters(model)
    for group in counts:
        print("%-40s %12s  (%.1f%%)" % (group, "{:,}".format(counts[group]), 100.0 * counts[group] / total))
    print("%-40s %12s" % ("TOTAL", "{:,}".format(total)))

    block = model.blocks[0]
    block_attention = count_parameters(block.attention)
    block_mlp = count_parameters(block.mlp)
    block_norms = count_parameters(block.norm1) + count_parameters(block.norm2)
    print("One block: attention %s, MLP %s, LayerNorms %s"
          % ("{:,}".format(block_attention), "{:,}".format(block_mlp), "{:,}".format(block_norms)))

    names = list(counts.keys())
    values_in_millions = []
    for name in names:
        values_in_millions.append(counts[name] / 1e6)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.barh(names, values_in_millions)
    for index in range(len(names)):
        axis.text(values_in_millions[index] + 0.5, index, "%.1fM" % values_in_millions[index], va="center")
    axis.set_xlabel("parameters (millions)")
    axis.set_title("GPT-2 small parameters per component (total %.1fM)" % (total / 1e6))
    figure.tight_layout()
    path = results_dir + "/phase2_parameters.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)

    return {"total": total, "per_component": counts,
            "per_block": {"attention": block_attention, "mlp": block_mlp, "layernorms": block_norms}}


# ---------------------------------------------------------------------------
# Step 3: FLOPs
# ---------------------------------------------------------------------------

def flops_by_formula(config, batch_size, seq_len):
    """Matmul FLOPs only (a Linear costs 2 * in * out per token). Element-wise ops are ignored."""
    d = config.d_model
    T = seq_len

    qkv_projections = 3 * (2 * T * d * d)
    attention_scores = 2 * T * T * d
    attention_weighted_sum = 2 * T * T * d
    output_projection = 2 * T * d * d
    mlp = 2 * T * d * config.d_mlp + 2 * T * config.d_mlp * d
    one_block = qkv_projections + attention_scores + attention_weighted_sum + output_projection + mlp
    lm_head = 2 * T * d * config.vocab_size

    breakdown = {
        "linear layers in blocks": batch_size * config.num_layers * (qkv_projections + output_projection + mlp),
        "attention scores + weighted sum": batch_size * config.num_layers * (attention_scores + attention_weighted_sum),
        "lm_head": batch_size * lm_head,
    }
    total = batch_size * (config.num_layers * one_block + lm_head)
    return total, breakdown


def flops_by_torch_profiler(model, input_ids):
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as profiler:
            model(input_ids)
    total = 0
    for event in profiler.key_averages():
        if event.flops is not None and event.flops > 0:
            total = total + event.flops
    return total


def flops_by_flop_counter(model, input_ids):
    counter = FlopCounterMode(display=False)
    with torch.no_grad():
        with counter:
            model(input_ids)
    return counter.get_total_flops()


def step3_count_flops(model, device, results_dir):
    print("\n========== STEP 3: FLOPs per forward pass (batch 1) ==========")
    config = model.config
    sequence_lengths = [128, 256, 512, 1024]
    rows = []
    for seq_len in sequence_lengths:
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        formula_total, breakdown = flops_by_formula(config, 1, seq_len)
        profiler_total = flops_by_torch_profiler(model, input_ids)
        counter_total = flops_by_flop_counter(model, input_ids)
        print("seq_len %4d | formula %7.1f GFLOPs | torch.profiler %7.1f | FlopCounterMode %7.1f | profiler/formula %.4f"
              % (seq_len, formula_total / 1e9, profiler_total / 1e9, counter_total / 1e9,
                 profiler_total / formula_total))
        rows.append({
            "seq_len": seq_len,
            "formula_flops": formula_total,
            "torch_profiler_flops": profiler_total,
            "flop_counter_flops": counter_total,
            "breakdown": breakdown,
            "flops_per_token": formula_total / seq_len,
        })

    labels = []
    linear_share = []
    attention_share = []
    head_share = []
    for row in rows:
        labels.append(str(row["seq_len"]))
        total = row["formula_flops"]
        linear_share.append(100.0 * row["breakdown"]["linear layers in blocks"] / total)
        attention_share.append(100.0 * row["breakdown"]["attention scores + weighted sum"] / total)
        head_share.append(100.0 * row["breakdown"]["lm_head"] / total)
    bottom_for_head = []
    for index in range(len(labels)):
        bottom_for_head.append(linear_share[index] + attention_share[index])

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, linear_share, label="Linear layers in blocks (QKV, out, MLP)")
    axis.bar(labels, attention_share, bottom=linear_share, label="Attention scores + weighted sum (grows with T^2)")
    axis.bar(labels, head_share, bottom=bottom_for_head, label="lm_head (768 -> 50257)")
    axis.set_xlabel("sequence length")
    axis.set_ylabel("% of forward FLOPs")
    axis.set_title("Where GPT-2's forward-pass compute goes")
    axis.legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    path = results_dir + "/phase2_flops_breakdown.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)
    return rows


# ---------------------------------------------------------------------------
# Step 4: latency
# ---------------------------------------------------------------------------

def measure_latency(model, batch_size, seq_len, device, num_repeats):
    """Returns (latency in ms, peak memory in MB), or (None, None) on out-of-memory."""
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i in range(2):
                model(input_ids)
            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_repeats):
                model(input_ids)
            torch.cuda.synchronize()
            elapsed = time.time() - start
        latency_ms = elapsed / num_repeats * 1000.0
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        return latency_ms, peak_memory_mb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def step4_latency(model, device, results_dir):
    print("\n========== STEP 4: latency and peak memory on this GPU ==========")
    sequence_lengths = [128, 256, 512, 1024]
    batch_sizes = [1, 4, 8]
    rows = []
    for precision in ["fp32", "fp16"]:
        if precision == "fp16":
            model = model.half()
        weights_mb = 0
        for parameter in model.parameters():
            weights_mb = weights_mb + parameter.numel() * parameter.element_size()
        weights_mb = weights_mb / 1024 / 1024
        print("--- %s (weights take %.0f MB) ---" % (precision, weights_mb))
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                latency_ms, peak_memory_mb = measure_latency(model, batch_size, seq_len, device, num_repeats=5)
                formula_total, breakdown = flops_by_formula(model.config, batch_size, seq_len)
                if latency_ms is None:
                    print("%s | batch %d | seq_len %4d | OUT OF MEMORY" % (precision, batch_size, seq_len))
                    achieved_tflops = None
                else:
                    achieved_tflops = formula_total / (latency_ms / 1000.0) / 1e12
                    tokens_per_second = batch_size * seq_len / (latency_ms / 1000.0)
                    print("%s | batch %d | seq_len %4d | %8.1f ms | %6.0f MB peak | %6.0f tokens/s | %.2f TFLOPS"
                          % (precision, batch_size, seq_len, latency_ms, peak_memory_mb,
                             tokens_per_second, achieved_tflops))
                rows.append({
                    "precision": precision,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "latency_ms": latency_ms,
                    "peak_memory_mb": peak_memory_mb,
                    "achieved_tflops": achieved_tflops,
                })
    model = model.float()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    precisions = ["fp32", "fp16"]
    for plot_index in range(2):
        axis = axes[plot_index]
        for batch_size in batch_sizes:
            x_values = []
            y_values = []
            for row in rows:
                if row["precision"] == precisions[plot_index] and row["batch_size"] == batch_size:
                    if row["latency_ms"] is not None:
                        x_values.append(row["seq_len"])
                        y_values.append(row["latency_ms"])
            axis.plot(x_values, y_values, marker="o", label="batch %d" % batch_size)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("sequence length")
        axis.set_title("GTX 1650, " + precisions[plot_index].upper())
        axis.legend()
    axes[0].set_ylabel("forward-pass latency (ms, log scale)")
    figure.tight_layout()
    path = results_dir + "/phase2_latency.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)
    return model, rows


# ---------------------------------------------------------------------------
# Step 5: perplexity
# ---------------------------------------------------------------------------

def step5_perplexity(model, device):
    print("\n========== STEP 5: perplexity on WikiText-2 test ==========")
    token_ids = load_wikitext2_tokens("test")
    print("Test set: %d tokens" % token_ids.shape[0])

    start = time.time()
    ppl_non_overlapping = evaluate_perplexity(model, token_ids, device, seq_len=1024)
    time_non_overlapping = time.time() - start
    print("Perplexity, non-overlapping 1024-token windows: %.2f  (%.0f s)"
          % (ppl_non_overlapping, time_non_overlapping))

    start = time.time()
    ppl_sliding = evaluate_perplexity(model, token_ids, device, seq_len=1024, stride=512)
    time_sliding = time.time() - start
    print("Perplexity, sliding window (stride 512):        %.2f  (%.0f s)"
          % (ppl_sliding, time_sliding))

    start = time.time()
    ppl_subset = evaluate_perplexity(model, token_ids, device, seq_len=1024, max_windows=40)
    time_subset = time.time() - start
    print("Perplexity, first 40 windows only: %.2f  (%.0f s)" % (ppl_subset, time_subset))

    return {
        "num_test_tokens": int(token_ids.shape[0]),
        "ppl_non_overlapping_1024": ppl_non_overlapping,
        "ppl_sliding_stride_512": ppl_sliding,
        "ppl_first_40_windows": ppl_subset,
        "seconds_non_overlapping": time_non_overlapping,
        "seconds_sliding": time_sliding,
        "seconds_subset_40": time_subset,
    }


def main():
    set_seed(0)
    device = get_device()
    results_dir = make_results_dir()
    results = {}
    model, results["step1_match_huggingface"] = step1_verify_against_huggingface(device)
    results["step2_parameters"] = step2_count_parameters(model, results_dir)
    results["step3_flops"] = step3_count_flops(model, device, results_dir)
    model, results["step4_latency"] = step4_latency(model, device, results_dir)
    results["step5_perplexity"] = step5_perplexity(model, device)
    save_json(results, "phase2_results.json")


if __name__ == "__main__":
    main()
