# LLM Compression & Custom Transformer Architecture

A from-scratch PyTorch project: build GPT-2 (124M) by hand, then compress it with
pruning (magnitude vs second-order, unstructured / 2:4 / structured), non-uniform
per-layer sparsity (NetAdapt, simplified AMC), and quantization (AWQ INT8 / INT4),
measuring perplexity on WikiText-2, model size, memory and latency.

All work runs on a laptop GTX 1650 (4 GB). The only extra GPU is Kaggle's free T4,
used for one Tensor Core latency benchmark.

## Setup

```
pip install -r requirements.txt
```

GPT-2 weights and WikiText-2 download automatically on first run.

## Project layout

| File | What it does |
|---|---|
| `common.py` | Shared helpers: device, seeds, results folder, parameter count |
| `phase1_foundations.py` | MLP + RNN from scratch; why transformers replaced RNNs |
| `gpt2_model.py` | GPT-2 (124M) written from scratch + loader for HuggingFace's pretrained weights |
| `evaluation.py` | WikiText-2 loading (cached in `data/`) and perplexity evaluation |
| `phase2_gpt2.py` | Verifies our GPT-2 against HuggingFace, counts params/FLOPs, measures latency, baseline perplexity |
| `pruning.py` | Pruning library: diagonal + gate Fisher, magnitude/OBD scores, unstructured / N:M / structured masks |
| `phase3_pruning.py` | Magnitude vs second-order (OBD) pruning across granularities, 10-70% sparsity |
| `phase4_nonuniform.py` | Per-layer sensitivity, NetAdapt-style greedy search, simplified AMC (DDPG) |
| `quantization.py` | Round-to-nearest and AWQ (written from scratch), model-size calculator |
| `phase5_hardware.py` | 2:4 sparsity, RTN vs AWQ INT8/INT4, 2:4 + INT8, final results table |
| `kaggle_benchmark/benchmark_tensor_cores.py` | Standalone latency benchmark (FP32/FP16/INT8/2:4 matmuls + full GPT-2), run on both GPUs |
| `results/` | Plots and JSON results from every phase |

## Phase 1 - Foundations

Run: `python phase1_foundations.py` (about 3 minutes on the GTX 1650)

**A. MLP vs linear model on three intertwined spirals**

| Model | Parameters | Test accuracy |
|---|---|---|
| Linear (no hidden layer) | 9 | 46.1% |
| MLP (2 hidden layers of 64, ReLU) | 4,547 | 99.4% |

A linear model can only draw straight boundaries. ReLU between layers lets the MLP
bend its boundary to follow the spirals (`results/phase1_mlp_decision_boundary.png`).

**B. Hand-written RNN loop vs `nn.RNN`**: max difference 8.3e-06 (float rounding),
so the hand-written `h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)` is exactly what PyTorch computes.

**C. Why transformers replaced RNNs**

First-token recall (output the first digit after reading the whole sequence; chance = 10%),
2000 training steps each:

| Sequence length | RNN | 1-layer self-attention |
|---|---|---|
| 5 | 100.0% | 100.0% |
| 20 | 9.2% | 100.0% |
| 50 | 11.2% | 100.0% |
| 100 | 9.9% | 100.0% |

Gradient reaching each input position (untrained models, length 100, float64):

| Position | RNN | Attention |
|---|---|---|
| 0 (first) | 5.4e-33 | 1.3e-05 |
| 50 | 2.2e-18 | 2.3e-05 |
| 99 (last) | 7.5e-04 | 2.1e-03 |

The RNN gradient shrinks by ~30 orders of magnitude from the last position to the first
(vanishing gradients). In float32 it rounds to exactly 0. Attention gives every position a direct path.

Forward time, one layer, batch 16 (GTX 1650, laptop timings vary ±20%):

| Width | Length | RNN | Attention |
|---|---|---|---|
| 128 | 64 / 2048 | 1.0 / 2.4 ms | 1.4 / 32.0 ms |
| 768 | 64 / 256 | 6.5 / 21.6 ms | 2.5 / 11.2 ms |
| 768 | 1024 / 2048 | 84 / 254 ms | 72 / 212 ms |

At GPT-2's width (768), attention is about 2-2.5x faster at typical lengths because all positions
run in parallel. At tiny width, cuDNN's RNN kernel is very fast, and at long lengths attention's
seq_len x seq_len score matrix (quadratic cost) catches up. The bigger win for transformers is
training: every position of every sequence is trained in parallel, and gradients don't vanish.

## Phase 2 - GPT-2 from scratch

Run: `python phase2_gpt2.py` (about 6 minutes on the GTX 1650, plus a one-time ~550 MB download)

**Correctness vs HuggingFace** (same pretrained weights, same input, FP32):
max logit difference 7.6e-05, mean 1.5e-05, 100% top-1 agreement, identical 20-token greedy generation.

**Parameters** (124,439,808 total):

| Component | Parameters | Share |
|---|---|---|
| Token embedding (shared with lm_head) | 38,597,376 | 31.0% |
| Position embedding | 786,432 | 0.6% |
| Attention, 12 blocks (Q, K, V, out) | 28,348,416 | 22.8% |
| MLP, 12 blocks (fc_in, fc_out) | 56,669,184 | 45.5% |
| LayerNorms | 38,400 | 0.03% |

Per block: attention 2.36M, MLP 4.72M (MLP is 2x attention).
The 72 Linear layers inside the blocks (85M params) are what Phases 3-5 prune and quantize.

**FLOPs per forward pass, batch 1** (formula vs two PyTorch counters):

| Sequence length | Formula | torch.profiler | FlopCounterMode |
|---|---|---|---|
| 128 | 32.2 G | 32.2 G | 32.2 G |
| 256 | 65.7 G | 65.7 G | 65.7 G |
| 512 | 136.2 G | 136.2 G | 136.2 G |
| 1024 | 291.6 G | 291.7 G | 291.6 G |

Rule of thumb: about 2 x parameters FLOPs per token (~250 MFLOPs), plus the attention term that grows
with length squared (2% of compute at 128 tokens, 14% at 1024). See `results/phase2_flops_breakdown.png`.

**Latency on the GTX 1650** (forward pass, ms; peak GPU memory in brackets):

| Batch | Seq len | FP32 | FP16 |
|---|---|---|---|
| 1 | 128 | 26 (521 MB) | 95 (286 MB) |
| 1 | 1024 | 208 (695 MB) | 767 (401 MB) |
| 4 | 1024 | 1,046 (1,294 MB) | 3,063 (784 MB) |
| 8 | 1024 | 2,163 (2,090 MB) | 6,138 (1,294 MB) |

FP32 reaches 1.1-1.8 TFLOPS. FP16 halves memory but runs about 3-4x slower (~0.4 TFLOPS) because FP16
matrix multiplies are slow on this GPU (also seen in a plain 4096x4096 matmul test); the likely cause is
that the GTX 1650 has no Tensor Cores. So everything on the laptop computes in FP32.

**Baseline perplexity, WikiText-2 test (287,644 tokens), FP32:**

| Evaluation | Perplexity | Time |
|---|---|---|
| Non-overlapping 1024-token windows | **29.94** | 63 s |
| Sliding window, stride 512 | 25.17 | 131 s |
| First 40 windows only (quick subset) | 31.33 | 10 s |

These match the published GPT-2 small numbers (~29.9 / ~25.2). The non-overlapping 29.94 is the
baseline that later phases compare against.

## Phase 3 - Pruning comparison

Run: `python phase3_pruning.py` (about 55 minutes: 47 pruned models x full test set)

Setup: one-shot pruning, no retraining. Every block is pruned to the same sparsity (uniform).
Sparsity = fraction of the 72 block Linear weights set to zero. Fisher from 128 x 512 tokens of the
WikiText-2 **train** split (38 s). Perplexity on the full **test** set. Dense baseline: **29.94**.

- **Magnitude:** score = |w|.
- **OBD:** score = ½ · F_ii · w², where F_ii = E[(dL/dw_i)²] is the diagonal Fisher (a Hessian approximation).
- **Structured OBD:** OBD applied to a 0/1 gate on each head / neuron:
  score = ½ · E[(Σ g·w over the unit's output weights)²].
  `obd_naive` = sum of per-weight ½·F·w² over the unit (ignores interactions inside the unit).

**Unstructured** (perplexity):

| Sparsity | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|
| Magnitude | 30.3 | 32.1 | 40.5 | 87.9 | 343.8 | 1,249 | 4,296 |
| OBD | **30.1** | **30.5** | **32.0** | **37.8** | **109.4** | **596** | **3,028** |

**N:M** (keep N of every M weights along each row):

| Pattern (sparsity) | 7:8 (12.5%) | 6:8 (25%) | 5:8 (37.5%) | 4:8 (50%) | **2:4 (50%)** | 3:8 (62.5%) |
|---|---|---|---|---|---|---|
| Magnitude | 31.4 | 37.7 | 66.9 | 345.2 | 719.8 | 1,786 |
| OBD | **30.6** | **32.4** | **39.3** | **75.1** | **132.6** | **643** |

**Structured** (whole heads + MLP neurons; actual sparsity differs slightly because 12 heads round):

| Sparsity | 9.4% | 18.9% | 31.1% | 40.6% | 50% | 59.4% | 68.9% |
|---|---|---|---|---|---|---|---|
| Magnitude | 407 | **309** | **744** | **2,072** | **2,763** | **3,063** | **4,683** |
| OBD (gate Fisher) | **40.8** | 975 | 9,706 | 67,596 | 71,635 | 3.5M | 233k |
| OBD naive (diagonal sum) | 86.8 | 1,591 | 45,461 | 14,490 | 41,653 | 1.0M | 1.0M |

What this shows:
1. **OBD beats magnitude everywhere for unstructured and N:M**, by 3x at 50% unstructured (109 vs 344)
   and 5x at 2:4 (133 vs 720). Magnitude ignores how much the loss depends on each weight; the Fisher
   captures that, including the effect of large input activations.
2. **Granularity:** at 50%, OBD gives 4:8 (75) < unstructured (109) < 2:4 (133) << structured (72k).
   2:4 is worse than 4:8 because a group of 4 is a tighter constraint than a group of 8.
   4:8 even beats per-layer unstructured: forcing every row to keep half its weights stops pruning from
   wiping out whole rows/columns that a few important features depend on.
3. **Structured pruning without retraining is destructive** (300-4,700 perplexity for magnitude).
   Whole heads/neurons are removed, so the model can't compensate unless it is fine-tuned.
4. **Second-order scores only work for small changes.** Structured OBD is the best method at 9.4%
   (40.8 vs 407) but collapses beyond that. OBD is a Taylor expansion around the current weights:
   accurate for small perturbations, wrong for deleting many whole units at once. The gate Fisher is
   far better than the naive per-weight sum at small sparsity (40.8 vs 86.8; 975 vs 1,591), because it
   includes interactions between the weights of a unit. Adding the first-order gradient term
   (LLM-Pruner style) made it worse in a quick test.
5. **Magnitude's structured curve is not monotonic** (407 at 9.4% vs 309 at 18.9%): with no
   retraining, removing extra units can partly cancel the damage done by earlier ones.

Plots: `results/phase3_perplexity_vs_sparsity.png`, `results/phase3_all_methods.png`.

## Phase 4 - Non-uniform sparsity per layer

Run: `python phase4_nonuniform.py` (about 25 minutes)

Same overall budget for everyone: **50% of all block weights**, unstructured OBD scores inside each
layer. Sparsity is allocated to 24 groups (attention and MLP of each of the 12 blocks).
All searching uses 4 windows of the WikiText-2 **validation** split; only the final numbers use the **test** set.

1. **Sensitivity scan** (91 s): prune one group alone at 30/50/70/90%. Almost every group barely
   notices even 90% when pruned alone (validation perplexity 27.2 -> 28-31). The exceptions are
   **block 0**: its attention goes 27 -> 47 at just 50%, and its MLP explodes to 406 at 90%.
   Block 1's MLP is the next most sensitive. See `results/phase4_sensitivity.png`.
2. **NetAdapt-style greedy** (15 min): 40 iterations. Each iteration tries removing another 1.25% of
   all block weights from each of the 24 groups, keeps the one change that hurts validation perplexity
   least, and repeats. Differences from the paper: the resource is weight count (not measured
   latency), and there is no short fine-tuning after each step.
3. **Simplified AMC** (5 min): DDPG-style actor-critic. The agent walks through the 24 groups and
   outputs a sparsity for each. Actions are clipped so every episode lands exactly on 50%; the reward is
   -log(validation perplexity). 60 random warm-up episodes, then 240 policy episodes with decaying
   noise; the best allocation found is kept. Simplified vs the paper: 24 groups instead of every layer,
   the critic regresses the final episode reward directly (no target networks/bootstrapping),
   a perplexity reward instead of accuracy, and no fine-tuning.

| Allocation at 50% overall | Validation subset | **Test perplexity** |
|---|---|---|
| Uniform (every group 50%) | 96.4 | 109.4 |
| NetAdapt-style greedy | 49.6 | **55.6** |
| AMC (simplified, DDPG) | 52.6 | 58.0 |
| Dense model (reference) | 27.2 | 29.9 |

Same number of zeros, but choosing *where* they go halves the perplexity. Both searches discovered the
same pattern (`results/phase4_allocations.png`):
- **Protect block 0** (NetAdapt left b0.attn completely dense; AMC gave it 40%).
- **Prune the last blocks hardest** (b10/b11 pushed to 67-90%).
- The middle blocks sit around 33-58%.

NetAdapt wins slightly and is deterministic, but it needed 3x the search time (926 s vs 323 s) and
~960 evaluations vs 300. AMC only began beating its random warm-up after ~150 episodes
(`results/phase4_amc_search.png`), which is typical of RL search: it is sample-hungry, but its
cost does not grow with the number of candidate moves the way greedy search does.

## Phase 5 - Hardware-aware compression

Run: `python phase5_hardware.py` (about 15 minutes), plus the benchmark script on each GPU.

**Quantization format** (RTN and AWQ alike): group-wise asymmetric, 128 weights per group, each group
with an FP16 scale and zero point (+0.25 bits per weight). Accuracy is simulated ("fake quantization"):
weights are rounded to exactly the values INT8/INT4 can represent, then computed in FP32.

**AWQ, implemented from scratch** (`quantization.py`): for each Linear layer, collect its inputs on
16k WikiText-2 **train** tokens and measure the average |activation| of every input channel. Try scales
s = |activation|^α for α = 0, 0.05, ..., 1, quantize W·s, divide the input by s, and keep the α with the
smallest output error. α = 0 is plain RTN, so AWQ can't do worse than RTN on calibration data.
Simplifications vs the paper: a separate scale search per layer (the paper shares one scale across layers
that read the same input, so 1/s can be folded into the previous LayerNorm), and no weight-clipping search.

**2:4** uses OBD scores (Phase 3's best 2:4 method). **2:4 + INT8** = prune first, then quantize the
kept weights. AWQ calibration runs on the pruned model, and zeros stay exactly zero because zero points
are integers.

### Final results table

| Configuration | WikiText-2 test perplexity | Model size (MB) | Bits per block weight |
|---|---|---|---|
| FP16 baseline | 29.94 | 237.4 | 16 |
| RTN INT8 | 29.95 | 158.9 | 8.25 |
| **AWQ INT8** | **29.93** | 158.9 | 8.25 |
| RTN INT4 | 33.25 | 118.4 | 4.25 |
| **AWQ INT4** | **32.18** | 118.4 | 4.25 |
| **2:4 sparse (FP16)** | 132.51 | 166.5 | 9 (8 value bits + 1 index bit, averaged) |
| 2:4 + RTN INT8 | 133.45 | 128.5 | 5.25 |
| **2:4 + AWQ INT8** | 132.76 | 128.5 | 5.25 |

Size = the 72 block Linear weights in the listed format + everything else (embeddings, LayerNorms, biases)
in FP16. "Memory" for the weights is this same number when the format is stored natively. In our
simulation the GPU holds FP32 copies, so runtime memory savings need real low-bit kernels.

What this shows:
1. **INT8 is free:** 29.94 -> 29.95 (RTN) / 29.93 (AWQ). With 256 levels per group of 128 weights, the
   rounding error is tiny, so AWQ has almost nothing to fix.
2. **INT4 is where AWQ matters:** RTN loses 3.31 perplexity, AWQ loses 2.24, so AWQ removes a third of the
   damage with no retraining. With only 16 levels, rounding error is large, and AWQ protects the
   channels with large activations, which amplify that error. Mean chosen α was 0.35; only 1 of 72
   layers preferred plain RTN.
3. **INT4 also creates zeros:** 15% of block weights round to exactly 0 at INT4 (0.9% at INT8). Small
   weights fall into the zero bucket.
4. **Embeddings limit compression:** the token embedding table (38.6M params = 74 MB in FP16) is untouched,
   so INT4 shrinks the model only 2x (237 -> 118 MB) even though block weights shrink 3.8x.
5. **2:4 without retraining is expensive in accuracy:** 132.5 perplexity, versus 32.2 for AWQ INT4, which
   is *smaller* (118 vs 128-167 MB). After one-shot compression, quantization is the much better deal.
   NVIDIA's 2:4 recipe includes fine-tuning after pruning, which this project does not do.
6. **2:4 + INT8 stacks well:** adding INT8 to 2:4 barely changes accuracy (132.5 -> 132.8 with AWQ) and
   cuts size from 166.5 to 128.5 MB.

### Latency

Same standalone script on both GPUs (`kaggle_benchmark/benchmark_tensor_cores.py`).
Matrix multiply with GPT-2's MLP fc_in shape (2048 tokens x 768 -> 3072):

| Kernel | GTX 1650 (laptop) | Kaggle T4 |
|---|---|---|
| FP32 | 5.98 ms (1.6 TFLOPS) | 2.22 ms (4.4 TFLOPS) |
| FP16 | 23.28 ms (0.4 TFLOPS), 0.26x | **0.36 ms (26.7 TFLOPS), 6.1x faster than FP32** |
| INT8 (`torch._int_mm`) | **0.88 ms (10.9 TOPS), 6.8x** | 0.71 ms (13.6 TOPS), 3.1x |
| FP16 with 50% zeros, dense kernel | 23.24 ms (same as dense) | 0.38 ms (same as dense) |
| FP16 2:4 sparse kernel | not supported: "cuSPARSELt not supported on your machine" | not supported: "architecture mismatch when calling cusparseLtInit" |

Full GPT-2 small forward pass (random weights, HF implementation; peak memory in brackets):

| Batch x seq | GTX 1650 FP32 | GTX 1650 FP16 | T4 FP32 | T4 FP16 | T4 FP16 speedup |
|---|---|---|---|---|---|
| 1 x 128 | 37 ms (518 MB) | 96 ms (273 MB) | 16.8 ms | 9.5 ms | 1.8x |
| 1 x 1024 | 166 ms (755 MB) | 665 ms (392 MB) | 78.9 ms | 23.0 ms | 3.4x |
| 8 x 512 | 709 ms | 2,578 ms | 298 ms | 86.5 ms | 3.4x |
| 8 x 1024 | 1,504 ms (2,654 MB) | 5,266 ms (1,342 MB) | 618 ms | 178 ms | 3.5x |

All shapes and token counts: `results/final_results_table.md` and the two `results/benchmark_results_*.json` files.

T4 findings:
- **FP16 Tensor Cores are the big win:** 6x FP32 on the matmul and 3.4-3.5x on the whole model. Short
  sequences gain less (1.8x at 128 tokens), because the GPU isn't kept busy and non-matmul work matters more.
- **INT8 via `torch._int_mm` was *slower* than FP16 on the T4** (6-15 TOPS vs 11-27 TFLOPS at every shape),
  far below the T4's INT8 Tensor Core rating. The likely reason: Turing's INT8 Tensor Core kernels need
  special interleaved memory layouts that this generic PyTorch op doesn't use, so it falls back to DP4A
  on the CUDA cores. Evidence: T4 INT8 (13.6 TOPS) is close to the GTX 1650's DP4A INT8 (10.9 TOPS), and
  the GTX 1650 has no Tensor Cores at all. Production INT8 on Turing (e.g. TensorRT) uses those layouts.
  From Ampere on, INT8 Tensor Cores accept the normal layout.
- **2:4 fails on the T4 too** ("architecture mismatch" from cuSPARSELt), and 50% zeros in a dense kernel
  run at dense speed. Same conclusion as on the laptop.

Same software on different hardware: FP16 is **4x slower** than FP32 on the GTX 1650 but **6x faster** on
the T4. Which format is "fast" depends entirely on which units the GPU has.

Laptop findings:
- **INT8 is ~7x faster than FP32 on the GTX 1650, even without Tensor Cores.** Turing's CUDA cores have a
  DP4A instruction (4 INT8 multiply-adds per instruction), so INT8 throughput is about 4x FP32 peak,
  and the measured 10-12 TOPS fits a laptop GTX 1650's FP32 peak of ~3 TFLOPS.
- **FP16 is 4x slower than FP32** on this GPU, at every shape and in the full model. FP16 halves memory,
  but not time.
- **Zeros don't make a dense kernel faster:** FP16 with 50% zeros runs at exactly the dense speed. Only a
  sparse kernel can skip work, and the 2:4 kernel refuses to run on this GPU.

## Hardware notes

| GPU | Used for |
|---|---|
| GTX 1650 laptop (Turing, sm_75, 4 GB, no Tensor Cores; torch 2.14.1 + CUDA 12.6) | All training/evaluation, every perplexity number, laptop latency |
| Kaggle T4 (Turing, sm_75, FP16/INT8 Tensor Cores, no Sparse Tensor Cores; torch 2.11 + CUDA 13.0) | Tensor Core latency benchmark only |

**Running the T4 benchmark yourself** (needs a phone-verified Kaggle account and `kaggle auth login`):

```
kaggle kernels push -p kaggle_benchmark --accelerator NvidiaTeslaT4
kaggle kernels status <username>/llm-compression-tc-benchmark
kaggle kernels output <username>/llm-compression-tc-benchmark -p results/kaggle_t4
```

Then copy `benchmark_results_Tesla_T4.json` into `results/` and run `python phase5_hardware.py --report-only`.
Change the `id` in `kaggle_benchmark/kernel-metadata.json` to your own username first.

Measured on the GTX 1650: FP32 matmul ~1.8 TFLOPS, FP16 matmul only ~0.4 TFLOPS (with both cuBLAS
and cuBLASLt), while FP16 element-wise ops run at full memory bandwidth. INT8 matmul reaches 10-12 TOPS
via DP4A. Laptop accuracy experiments compute in FP32; FP16 is used for size/memory comparisons.

### Why no 2:4 speedup is shown

- 2:4 acceleration needs **Sparse Tensor Cores**, which first appeared in **Ampere** (A100, RTX 30xx,
  compute capability 8.0+). They read the compressed 2:4 format (the kept values plus 2-bit position
  indices) and skip the zero multiplies, doing half the math.
- The **GTX 1650** (compute capability 7.5) has no Tensor Cores at all. The **T4** (also 7.5, Turing) has
  FP16/INT8 Tensor Cores but **no sparse support**. On both, PyTorch's 2:4 kernel (cuSPARSELt) refuses
  to run ("not supported on your machine" / "architecture mismatch"), and a dense kernel with 50% zeros
  takes exactly the dense time (measured above).
- So on this hardware, 2:4 gives **accuracy and storage** results, but no latency gain.

**Theoretical speedup on Ampere+**: Sparse Tensor Cores double the throughput of the block Linear layers
only. Attention scores, the 768 -> 50,257 lm_head, LayerNorm and so on are unchanged. Amdahl's law with
Phase 2's FLOP breakdown:

| Sequence length | Share of FLOPs in block Linears | Best-case end-to-end speedup |
|---|---|---|
| 128 | 67% | 1.51x |
| 512 | 64% | 1.47x |
| 1024 | 60% | 1.42x |

This is an upper bound: real sparse kernels reach less than 2x on the matrix multiplies themselves,
especially for small ones like GPT-2 small's. Memory-bound single-token decoding gains mostly from
having fewer weight bytes to read.
