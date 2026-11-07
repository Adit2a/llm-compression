"""
Phase 3: magnitude vs OBD pruning at unstructured, N:M and structured granularity,
10-70% uniform sparsity, one-shot (no retraining).

Usage: python phase3_pruning.py [--quick]
"""

import sys
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import get_device, set_seed, make_results_dir, save_json
from gpt2_model import load_pretrained_gpt2
from evaluation import load_wikitext2_tokens, evaluate_perplexity
from pruning import (save_original_weights, restore_original_weights, sample_calibration_windows,
                     compute_fisher_diagonal, prune_model_uniform, measure_sparsity)


QUICK_MODE = "--quick" in sys.argv

if QUICK_MODE:
    EVAL_WINDOWS = 10
    NUM_FISHER_SAMPLES = 8
    SPARSITY_LEVELS = [0.3, 0.6]
    NM_PATTERNS = [(4, 8), (2, 4)]
else:
    EVAL_WINDOWS = None  # full test set
    NUM_FISHER_SAMPLES = 128
    SPARSITY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    NM_PATTERNS = [(7, 8), (6, 8), (5, 8), (4, 8), (3, 8), (2, 4)]  # (n_keep, m)

FISHER_SEQ_LEN = 512
METHODS = ["magnitude", "obd"]
STRUCTURED_METHODS = ["magnitude", "obd", "obd_naive"]


def run_one_experiment(model, original_weights, method, granularity, sparsity, fisher, gate_fisher,
                       test_tokens, device, n_keep=None, m=None):
    restore_original_weights(model, original_weights)
    prune_model_uniform(model, method, granularity, sparsity, fisher, gate_fisher, n_keep, m)
    actual_sparsity = measure_sparsity(model)
    start = time.time()
    perplexity = evaluate_perplexity(model, test_tokens, device, max_windows=EVAL_WINDOWS)
    seconds = time.time() - start
    label = granularity
    if granularity == "nm":
        label = "%d:%d" % (n_keep, m)
    print("%-9s | %-12s | target %4.1f%% | actual %5.1f%% | perplexity %10.2f | %3.0f s"
          % (method, label, sparsity * 100, actual_sparsity * 100, perplexity, seconds))
    return {
        "method": method,
        "granularity": granularity,
        "pattern": label,
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "perplexity": perplexity,
    }


def plot_results(rows, baseline_ppl, results_dir):
    granularities = ["unstructured", "nm", "structured"]
    titles = {"unstructured": "Unstructured", "nm": "N:M (N:8 series, star = 2:4)",
              "structured": "Structured (heads + MLP neurons)"}
    colors = {"magnitude": "tab:blue", "obd": "tab:red", "obd_naive": "tab:orange"}

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for plot_index in range(3):
        granularity = granularities[plot_index]
        axis = axes[plot_index]
        methods_to_plot = METHODS
        if granularity == "structured":
            methods_to_plot = STRUCTURED_METHODS
        for method in methods_to_plot:
            x_values = []
            y_values = []
            for row in rows:
                if row["method"] == method and row["granularity"] == granularity and row["pattern"] != "2:4":
                    x_values.append(row["actual_sparsity"] * 100)
                    y_values.append(row["perplexity"])
            axis.plot(x_values, y_values, marker="o", color=colors[method], label=method)
            for row in rows:
                if row["method"] == method and row["pattern"] == "2:4" and granularity == "nm":
                    axis.plot([row["actual_sparsity"] * 100], [row["perplexity"]], marker="*",
                              markersize=16, color=colors[method], linestyle="none",
                              label=method + " 2:4")
        axis.axhline(baseline_ppl, color="gray", linestyle="--", label="dense baseline")
        axis.set_yscale("log")
        axis.set_xlabel("sparsity of block Linear weights (%)")
        axis.set_title(titles[granularity])
        axis.legend(fontsize=8)
    axes[0].set_ylabel("WikiText-2 perplexity (log scale, lower = better)")
    figure.tight_layout()
    path = results_dir + "/phase3_perplexity_vs_sparsity.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)

    line_styles = {"unstructured": "-", "nm": "--", "structured": ":"}
    figure, axis = plt.subplots(figsize=(8, 5))
    for granularity in granularities:
        for method in METHODS:
            x_values = []
            y_values = []
            for row in rows:
                if row["method"] == method and row["granularity"] == granularity and row["pattern"] != "2:4":
                    x_values.append(row["actual_sparsity"] * 100)
                    y_values.append(row["perplexity"])
            axis.plot(x_values, y_values, marker="o", color=colors[method],
                      linestyle=line_styles[granularity], label=method + ", " + granularity)
    axis.axhline(baseline_ppl, color="gray", linestyle="-", linewidth=0.8, label="dense baseline")
    axis.set_yscale("log")
    axis.set_xlabel("sparsity of block Linear weights (%)")
    axis.set_ylabel("WikiText-2 perplexity (log scale)")
    axis.set_title("GPT-2 small, uniform pruning, no retraining")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = results_dir + "/phase3_all_methods.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print("Saved", path)


def main():
    set_seed(0)
    device = get_device()
    results_dir = make_results_dir()
    if QUICK_MODE:
        print("QUICK MODE: tiny subset, results are only a smoke test")

    model = load_pretrained_gpt2(device)
    original_weights = save_original_weights(model)
    test_tokens = load_wikitext2_tokens("test")
    train_tokens = load_wikitext2_tokens("train")

    baseline_ppl = evaluate_perplexity(model, test_tokens, device, max_windows=EVAL_WINDOWS)
    print("Dense baseline perplexity: %.2f" % baseline_ppl)

    print("\nComputing diagonal Fisher on %d train samples of %d tokens..."
          % (NUM_FISHER_SAMPLES, FISHER_SEQ_LEN))
    start = time.time()
    calibration_windows = sample_calibration_windows(train_tokens, NUM_FISHER_SAMPLES, FISHER_SEQ_LEN, seed=0)
    fisher, gate_fisher = compute_fisher_diagonal(model, calibration_windows, device)
    fisher_seconds = time.time() - start
    print("Fisher done in %.0f s" % fisher_seconds)

    rows = []
    results = {"quick_mode": QUICK_MODE, "baseline_perplexity": baseline_ppl,
               "fisher_seconds": fisher_seconds, "num_fisher_samples": NUM_FISHER_SAMPLES,
               "rows": rows}
    output_name = "phase3_results_quick.json" if QUICK_MODE else "phase3_results.json"

    print("\n--- Unstructured ---")
    for sparsity in SPARSITY_LEVELS:
        for method in METHODS:
            rows.append(run_one_experiment(model, original_weights, method, "unstructured",
                                           sparsity, fisher, gate_fisher, test_tokens, device))
        save_json(results, output_name)

    print("\n--- N:M ---")
    for pattern in NM_PATTERNS:
        n_keep = pattern[0]
        m = pattern[1]
        sparsity = 1.0 - n_keep / m
        for method in METHODS:
            rows.append(run_one_experiment(model, original_weights, method, "nm", sparsity,
                                           fisher, gate_fisher, test_tokens, device,
                                           n_keep=n_keep, m=m))
        save_json(results, output_name)

    print("\n--- Structured (heads + MLP neurons) ---")
    for sparsity in SPARSITY_LEVELS:
        for method in STRUCTURED_METHODS:
            rows.append(run_one_experiment(model, original_weights, method, "structured",
                                           sparsity, fisher, gate_fisher, test_tokens, device))
        save_json(results, output_name)

    restore_original_weights(model, original_weights)
    if not QUICK_MODE:
        plot_results(rows, baseline_ppl, results_dir)


if __name__ == "__main__":
    main()
