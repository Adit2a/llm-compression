# Final results table (GPT-2 small, WikiText-2 test perplexity)

| Configuration | Perplexity | Model size (MB) | Bits per block weight |
|---|---|---|---|
| FP16 baseline | 29.94 | 237.4 | 16.00 |
| RTN INT8 | 29.95 | 158.9 | 8.25 |
| AWQ INT8 | 29.93 | 158.9 | 8.25 |
| RTN INT4 | 33.25 | 118.4 | 4.25 |
| AWQ INT4 | 32.18 | 118.4 | 4.25 |
| 2:4 sparse (FP16) | 132.51 | 166.5 | 9.00 |
| 2:4 + RTN INT8 | 133.45 | 128.5 | 5.25 |
| 2:4 + AWQ INT8 | 132.76 | 128.5 | 5.25 |

Accuracy is measured with simulated (fake) quantization and masked weights, computed in FP32 on the GTX 1650.
Size = block Linear weights in the given format + everything else (embeddings, LayerNorms, biases) in FP16.

## Latency on NVIDIA GeForce GTX 1650 (compute capability 7.5)

Matrix multiply, MLP fc_in shape (2048 tokens x 768 -> 3072):

| Kernel | Time (ms) | Throughput (TFLOPS / TOPS) | Speedup vs FP32 |
|---|---|---|---|
| FP32 | 5.982 | 1.6 | 1.00x |
| FP16 | 23.277 | 0.4 | 0.26x |
| INT8 | 0.884 | 10.9 | 6.76x |
| FP16, 50% zeros (dense kernel) | 23.242 | 0.4 | 0.26x |
| FP16 2:4 sparse kernel | not supported: cuSPARSELt not supported on your machine. | - | - |

Full GPT-2 small forward pass:

| Batch x seq | FP32 (ms) | FP16 (ms) | FP16 speedup |
|---|---|---|---|
| 1 x 128 | 37.1 | 96.2 | 0.39x |
| 1 x 1024 | 166.2 | 664.6 | 0.25x |
| 8 x 512 | 709.3 | 2577.8 | 0.28x |
| 8 x 1024 | 1504.1 | 5266.3 | 0.29x |

## Latency on Tesla T4 (compute capability 7.5)

Matrix multiply, MLP fc_in shape (2048 tokens x 768 -> 3072):

| Kernel | Time (ms) | Throughput (TFLOPS / TOPS) | Speedup vs FP32 |
|---|---|---|---|
| FP32 | 2.217 | 4.4 | 1.00x |
| FP16 | 0.362 | 26.7 | 6.13x |
| INT8 | 0.713 | 13.6 | 3.11x |
| FP16, 50% zeros (dense kernel) | 0.381 | 25.4 | 5.82x |
| FP16 2:4 sparse kernel | not supported: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` | - | - |

Full GPT-2 small forward pass:

| Batch x seq | FP32 (ms) | FP16 (ms) | FP16 speedup |
|---|---|---|---|
| 1 x 128 | 16.8 | 9.5 | 1.77x |
| 1 x 1024 | 78.9 | 23.0 | 3.43x |
| 8 x 512 | 298.3 | 86.5 | 3.45x |
| 8 x 1024 | 618.2 | 177.5 | 3.48x |

## Theoretical 2:4 speedup on Ampere+ (Amdahl's law)

| Sequence length | Share of FLOPs in block Linears | Best-case end-to-end speedup |
|---|---|---|
| 128 | 67% | 1.51x |
| 512 | 64% | 1.47x |
| 1024 | 60% | 1.42x |
