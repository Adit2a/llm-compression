"""
Phase 5: 2:4 sparsity, RTN vs AWQ (INT8 / INT4), and 2:4 + INT8.
Reports perplexity and model size, and builds the final table together with the
latency results from kaggle_benchmark/ (any results/benchmark_results_*.json).

Usage: python phase5_hardware.py [--report-only]
"""

import os
import sys
import glob
import json
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import get_device, set_seed, make_results_dir, save_json, RESULTS_DIR
from gpt2_model import load_pretrained_gpt2, GPT2Config
from evaluation import load_wikitext2_tokens, evaluate_perplexity
from pruning import (save_original_weights, restore_original_weights, sample_calibration_windows,
                     compute_fisher_diagonal, prune_model_uniform, measure_sparsity)
from quantization import quantize_model_rtn, quantize_model_awq, model_size_mb
from phase2_gpt2 import flops_by_formula


REPORT_ONLY = "--report-only" in sys.argv
GROUP_SIZE = 128
NUM_AWQ_SAMPLES = 32
AWQ_SEQ_LEN = 512

# bits per original block weight; scale + zero point = 32 bits per group
GROUP_OVERHEAD_BITS = 32.0 / GROUP_SIZE
BITS_PER_WEIGHT = {
    "fp16": 16.0,
    "int8": 8.0 + GROUP_OVERHEAD_BITS,
    "int4": 4.0 + GROUP_OVERHEAD_BITS,
    # 2:4 stores half the values plus a 2-bit index per kept value
    "2:4 fp16": 0.5 * 16.0 + 0.5 * 2.0,
    "2:4 int8": 0.5 * 8.0 + 0.5 * 2.0 + GROUP_OVERHEAD_BITS,
}


def round_whole_model_to_fp16(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(parameter.half().float())


def evaluate_configuration(name, storage_format, model, test_tokens, device, rows, extra=None):
    start = time.time()
    perplexity = evaluate_perplexity(model, test_tokens, device)
    size = model_size_mb(model, BITS_PER_WEIGHT[storage_format])
    sparsity = measure_sparsity(model)
    print("%-20s | perplexity %8.2f | size %6.1f MB | block sparsity %4.1f%% | %.0f s"
          % (name, perplexity, size, sparsity * 100, time.time() - start))
    row = {"name": name, "storage_format": storage_format, "perplexity": perplexity,
           "size_mb": size, "bits_per_block_weight": BITS_PER_WEIGHT[storage_format],
           "block_sparsity": sparsity}
    if extra is not None:
        row.update(extra)
    rows.append(row)


def average(values):
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


def run_experiments(device):
    model = load_pretrained_gpt2(device)
    round_whole_model_to_fp16(model)
    original_weights = save_original_weights(model)
    test_tokens = load_wikitext2_tokens("test")
    train_tokens = load_wikitext2_tokens("train")
    awq_windows = sample_calibration_windows(train_tokens, NUM_AWQ_SAMPLES, AWQ_SEQ_LEN, seed=1)
    rows = []
    alpha_summary = {}

    print("\n--- FP16 baseline ---")
    evaluate_configuration("FP16 baseline", "fp16", model, test_tokens, device, rows)

    print("\n--- Quantization: RTN vs AWQ ---")
    for bits in [8, 4]:
        restore_original_weights(model, original_weights)
        quantize_model_rtn(model, bits, GROUP_SIZE)
        evaluate_configuration("RTN INT%d" % bits, "int%d" % bits, model, test_tokens, device, rows)

        restore_original_weights(model, original_weights)
        start = time.time()
        alphas = quantize_model_awq(model, bits, GROUP_SIZE, awq_windows, device)
        awq_seconds = time.time() - start
        alpha_values = list(alphas.values())
        alpha_summary["int%d" % bits] = alphas
        print("AWQ INT%d search took %.0f s; mean chosen alpha %.2f; layers with alpha 0 (=RTN): %d / %d"
              % (bits, awq_seconds, average(alpha_values), alpha_values.count(0.0), len(alpha_values)))
        evaluate_configuration("AWQ INT%d" % bits, "int%d" % bits, model, test_tokens, device, rows,
                               {"awq_search_seconds": awq_seconds, "mean_alpha": average(alpha_values)})

    print("\n--- 2:4 sparsity (OBD scores, as in Phase 3) ---")
    restore_original_weights(model, original_weights)
    fisher_windows = sample_calibration_windows(train_tokens, 128, 512, seed=0)
    fisher, gate_fisher = compute_fisher_diagonal(model, fisher_windows, device)
    del gate_fisher
    prune_model_uniform(model, "obd", "nm", 0.5, fisher, None, n_keep=2, m=4)
    del fisher
    torch.cuda.empty_cache()
    sparse_weights = save_original_weights(model)
    evaluate_configuration("2:4 sparse (FP16)", "2:4 fp16", model, test_tokens, device, rows)

    print("\n--- 2:4 + INT8 ---")
    restore_original_weights(model, sparse_weights)
    quantize_model_rtn(model, 8, GROUP_SIZE)
    evaluate_configuration("2:4 + RTN INT8", "2:4 int8", model, test_tokens, device, rows)

    restore_original_weights(model, sparse_weights)
    # calibrate AWQ on the pruned model's activations
    alphas = quantize_model_awq(model, 8, GROUP_SIZE, awq_windows, device)
    alpha_summary["2:4 + int8"] = alphas
    evaluate_configuration("2:4 + AWQ INT8", "2:4 int8", model, test_tokens, device, rows)

    restore_original_weights(model, original_weights)
    return {"rows": rows, "awq_alphas": alpha_summary, "group_size": GROUP_SIZE,
            "num_awq_calibration_tokens": NUM_AWQ_SAMPLES * AWQ_SEQ_LEN}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def load_benchmarks():
    benchmarks = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "benchmark_results_*.json")):
        with open(path) as benchmark_file:
            data = json.load(benchmark_file)
        benchmarks[data["gpu"]] = data
    return benchmarks


def matmul_row(benchmark, layer_prefix, tokens):
    for row in benchmark["matmuls"]:
        if row["layer"].startswith(layer_prefix) and row["tokens"] == tokens:
            return row
    return None


def theoretical_2to4_speedup(seq_len):
    """Amdahl's law with 2x on the block Linear layers and everything else unchanged."""
    total, breakdown = flops_by_formula(GPT2Config(), 1, seq_len)
    linear_fraction = breakdown["linear layers in blocks"] / total
    return linear_fraction, 1.0 / ((1.0 - linear_fraction) + linear_fraction / 2.0)


def write_report(experiment_results, results_dir):
    rows = experiment_results["rows"]
    benchmarks = load_benchmarks()
    lines = []
    lines.append("# Final results table (GPT-2 small, WikiText-2 test perplexity)\n")
    lines.append("| Configuration | Perplexity | Model size (MB) | Bits per block weight |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append("| %s | %.2f | %.1f | %.2f |" % (row["name"], row["perplexity"], row["size_mb"],
                                                       row["bits_per_block_weight"]))
    lines.append("")
    lines.append("Accuracy is measured with simulated (fake) quantization and masked weights, computed in FP32 on the GTX 1650.")
    lines.append("Size = block Linear weights in the given format + everything else (embeddings, LayerNorms, biases) in FP16.\n")

    kinds = [("fp32", "FP32"), ("fp16", "FP16"), ("int8", "INT8"),
             ("fp16_50pct_zeros_dense", "FP16, 50% zeros (dense kernel)"),
             ("fp16_2to4_sparse", "FP16 2:4 sparse kernel")]
    for gpu_name in benchmarks:
        benchmark = benchmarks[gpu_name]
        lines.append("## Latency on %s (compute capability %d.%d)\n"
                     % (gpu_name, benchmark["compute_capability"][0], benchmark["compute_capability"][1]))
        lines.append("Matrix multiply, MLP fc_in shape (2048 tokens x 768 -> 3072):\n")
        lines.append("| Kernel | Time (ms) | Throughput (TFLOPS / TOPS) | Speedup vs FP32 |")
        lines.append("|---|---|---|---|")
        row = matmul_row(benchmark, "mlp fc_in", 2048)
        for key, label in kinds:
            if row is None:
                continue
            milliseconds = row.get(key + "_ms")
            if milliseconds is None:
                lines.append("| %s | not supported: %s | - | - |" % (label, row.get(key + "_error", "?")))
            else:
                lines.append("| %s | %.3f | %.1f | %.2fx |" % (label, milliseconds, row[key + "_tops"],
                                                               row["fp32_ms"] / milliseconds))
        lines.append("")
        if len(benchmark["full_gpt2"]) > 0:
            lines.append("Full GPT-2 small forward pass:\n")
            lines.append("| Batch x seq | FP32 (ms) | FP16 (ms) | FP16 speedup |")
            lines.append("|---|---|---|---|")
            for fp32_row in benchmark["full_gpt2"]:
                if fp32_row["precision"] != "fp32":
                    continue
                for fp16_row in benchmark["full_gpt2"]:
                    same_shape = (fp16_row["batch_size"] == fp32_row["batch_size"]
                                  and fp16_row["seq_len"] == fp32_row["seq_len"])
                    if fp16_row["precision"] == "fp16" and same_shape:
                        if fp32_row["latency_ms"] is None or fp16_row["latency_ms"] is None:
                            continue
                        lines.append("| %d x %d | %.1f | %.1f | %.2fx |" % (
                            fp32_row["batch_size"], fp32_row["seq_len"], fp32_row["latency_ms"],
                            fp16_row["latency_ms"], fp32_row["latency_ms"] / fp16_row["latency_ms"]))
            lines.append("")

    lines.append("## Theoretical 2:4 speedup on Ampere+ (Amdahl's law)\n")
    lines.append("| Sequence length | Share of FLOPs in block Linears | Best-case end-to-end speedup |")
    lines.append("|---|---|---|")
    for seq_len in [128, 512, 1024]:
        fraction, speedup = theoretical_2to4_speedup(seq_len)
        lines.append("| %d | %.0f%% | %.2fx |" % (seq_len, fraction * 100, speedup))
    lines.append("")

    path = os.path.join(results_dir, "final_results_table.md")
    with open(path, "w", encoding="utf-8") as table_file:
        table_file.write("\n".join(lines))
    print("\n".join(lines))
    print("Saved", path)

    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        axis.scatter(row["size_mb"], row["perplexity"], s=60)
        axis.annotate(row["name"], (row["size_mb"], row["perplexity"]), textcoords="offset points",
                      xytext=(5, 5), fontsize=8)
    axis.set_yscale("log")
    axis.set_xlabel("model size (MB)")
    axis.set_ylabel("WikiText-2 test perplexity (log, lower = better)")
    axis.set_title("Size vs quality of compressed GPT-2 small")
    figure.tight_layout()
    plot_path = os.path.join(results_dir, "phase5_size_vs_perplexity.png")
    figure.savefig(plot_path, dpi=120)
    plt.close(figure)
    print("Saved", plot_path)


def main():
    set_seed(0)
    results_dir = make_results_dir()
    json_path = os.path.join(results_dir, "phase5_results.json")
    if REPORT_ONLY:
        with open(json_path) as json_file:
            experiment_results = json.load(json_file)
    else:
        device = get_device()
        experiment_results = run_experiments(device)
        save_json(experiment_results, "phase5_results.json")
    write_report(experiment_results, results_dir)


if __name__ == "__main__":
    main()
