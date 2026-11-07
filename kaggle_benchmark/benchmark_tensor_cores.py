"""
Standalone GPU latency benchmark (PyTorch only; transformers optional).

  1. Matmuls with GPT-2 small's Linear shapes: FP32, FP16, INT8 (torch._int_mm),
     FP16 with 50% zeros in a dense kernel, and FP16 2:4 semi-structured sparse
     (needs Sparse Tensor Cores, compute capability >= 8.0; failures are recorded).
  2. Full GPT-2 small forward pass, FP32 vs FP16 (random weights).

Writes benchmark_results_<gpu>.json to the current directory.
"""

import json
import time
import platform

import torch


WARMUP_ITERATIONS = 10
TIMED_ITERATIONS = 50

# (name, K = in_features, N = out_features)
LAYER_SHAPES = [
    ("attention q/k/v/out  768 -> 768", 768, 768),
    ("mlp fc_in            768 -> 3072", 768, 3072),
    ("mlp fc_out          3072 -> 768", 3072, 768),
]
# M = batch_size x seq_len
TOKEN_COUNTS = [512, 2048, 8192]


def time_function(function):
    """Average milliseconds per call, timed with CUDA events."""
    for i in range(WARMUP_ITERATIONS):
        function()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(TIMED_ITERATIONS):
        function()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / TIMED_ITERATIONS


class MatmulRunner:

    def __init__(self, kind, tokens, in_features, out_features, device):
        self.kind = kind
        if kind == "fp32":
            self.x = torch.randn(tokens, in_features, device=device)
            self.w = torch.randn(out_features, in_features, device=device)
        elif kind == "fp16" or kind == "fp16_50pct_zeros_dense":
            self.x = torch.randn(tokens, in_features, device=device, dtype=torch.float16)
            self.w = torch.randn(out_features, in_features, device=device, dtype=torch.float16)
            if kind == "fp16_50pct_zeros_dense":
                self.w[:, 0::2] = 0.0
        elif kind == "int8":
            self.x = torch.randint(-128, 127, (tokens, in_features), device=device, dtype=torch.int8)
            # torch._int_mm wants (M, K) @ (K, N)
            self.w_t = torch.randint(-128, 127, (in_features, out_features), device=device, dtype=torch.int8)
        elif kind == "fp16_2to4_sparse":
            from torch.sparse import to_sparse_semi_structured
            self.x = torch.randn(tokens, in_features, device=device, dtype=torch.float16)
            w = torch.randn(out_features, in_features, device=device, dtype=torch.float16)
            w[:, 0::4] = 0.0
            w[:, 1::4] = 0.0
            self.w = to_sparse_semi_structured(w)
        else:
            raise ValueError(kind)

    def run(self):
        if self.kind == "int8":
            return torch._int_mm(self.x, self.w_t)
        return torch.nn.functional.linear(self.x, self.w)


def benchmark_matmuls(device):
    kinds = ["fp32", "fp16", "int8", "fp16_50pct_zeros_dense", "fp16_2to4_sparse"]
    rows = []
    for layer_name, in_features, out_features in LAYER_SHAPES:
        for tokens in TOKEN_COUNTS:
            operations = 2 * tokens * in_features * out_features
            row = {"layer": layer_name, "tokens": tokens, "K": in_features, "N": out_features}
            text = "%s | M=%5d |" % (layer_name, tokens)
            for kind in kinds:
                try:
                    runner = MatmulRunner(kind, tokens, in_features, out_features, device)
                    milliseconds = time_function(runner.run)
                    tera_ops = operations / (milliseconds / 1000.0) / 1e12
                    row[kind + "_ms"] = milliseconds
                    row[kind + "_tops"] = tera_ops
                    text = text + " %s %.3f ms (%.1f T) |" % (kind, milliseconds, tera_ops)
                    del runner
                except Exception as error:
                    message = str(error).split("\n")[0][:150]
                    row[kind + "_ms"] = None
                    row[kind + "_error"] = message
                    text = text + " %s FAILED |" % kind
            print(text)
            rows.append(row)
    return rows


def benchmark_full_gpt2(device):
    """Full GPT-2 small forward pass, FP32 vs FP16, random weights."""
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        print("transformers not installed - skipping full-model benchmark")
        return []
    rows = []
    settings = [(1, 128), (1, 1024), (8, 512), (8, 1024)]
    for precision in ["fp32", "fp16"]:
        model = GPT2LMHeadModel(GPT2Config()).to(device).eval()
        if precision == "fp16":
            model = model.half()
        for batch_size, seq_len in settings:
            input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            def forward():
                with torch.no_grad():
                    return model(input_ids).logits

            try:
                for i in range(3):
                    forward()
                torch.cuda.synchronize()
                start = time.time()
                for i in range(10):
                    forward()
                torch.cuda.synchronize()
                milliseconds = (time.time() - start) / 10 * 1000
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                print("GPT-2 %s | batch %d | seq %4d | %8.1f ms | peak %6.0f MB"
                      % (precision, batch_size, seq_len, milliseconds, peak_mb))
                rows.append({"precision": precision, "batch_size": batch_size, "seq_len": seq_len,
                             "latency_ms": milliseconds, "peak_memory_mb": peak_mb})
            except torch.cuda.OutOfMemoryError:
                print("GPT-2 %s | batch %d | seq %4d | OUT OF MEMORY" % (precision, batch_size, seq_len))
                rows.append({"precision": precision, "batch_size": batch_size, "seq_len": seq_len,
                             "latency_ms": None, "peak_memory_mb": None})
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    return rows


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print("GPU:", gpu_name, "| compute capability", capability, "| torch", torch.__version__)
    print("Sparse Tensor Cores (2:4) need compute capability >= 8.0 (Ampere).")

    results = {
        "gpu": gpu_name,
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "matmuls": benchmark_matmuls(device),
        "full_gpt2": benchmark_full_gpt2(device),
    }
    short_name = gpu_name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "_")
    output_path = "benchmark_results_" + short_name + ".json"
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=2)
    print("Saved", output_path)


if __name__ == "__main__":
    main()
