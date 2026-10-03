# MI210 single-card long-context experiment (2026-10-03)

Status: operator numerics and latency gated; end-to-end 64K A/B pending.
One gfx90a MI210, 64 GB, 300 W. Preserve PTQR target/draft, CK W8A8,
inline grouped INT8 KV with FP16 per-group scales, BF16 attention operands,
and FP32 Mamba state. No dual-card measurements are possible on this host.

## Changes

Native CDNA2 AMD MFMA v2 16x16x16 layouts in grouped INT8 Gluon prefill,
with optional 32/64/128 KV tiles; host dispatch restricted to gfx90a and
explicit opt-in. The new image opts into BF16/64-column prefill.
The CDNA2 decode path was investigated separately: 
the first decode measurement was a generic fallback because gfx90a uses tile16.
After explicitly setting tile32 and tracking candidate launches, native
tile64 improved TP1 decode from 4.028 ms to 1.643 ms. The final image
enables the validated native decode path.
The generic attention path remains available. Unsupported layouts use
existing dispatch; the opt-in tile must divide the physical block size.

## Measured operator results

Reverse-ordered physical pages, page size 1728, head dimension 256,
GQA 6:1 (initial synthetic results used 6 Q heads / 1 KV head), causal masking, inline G128 scale bytes. CUDA/HIP event timing,
2 warmups and median of 5. Numerical comparison to FP32 attention.

| Context / query | Path | Median ms | Max absolute error |
|---|---|---:|---:|
| 49152 / 512 | generic | 30.415 | 4.00e-5 |
| 49152 / 512 | legacy Gluon BF16 | 12.843 | 4.00e-5 |
| 49152 / 512 | CDNA2 BF16 tile32 | 11.047 | 4.00e-5 |
| 49152 / 512 | CDNA2 BF16 tile64 | 8.965 | 4.00e-5 |
| 65537 / 2048 | CDNA2 BF16 tile64 | 24.019 | 3.32e-5 |
| 65536 / 14 split | generic | 1.729 | 2.26e-5 |
| 65536 / 14 split | requested CDNA2 (generic fallback) | 1.707 | 2.26e-5 |

The 3.39x result is **attention-kernel latency**, not whole-model tok/s.
Changing generic prefill to BLOCK_M=64/TILE=64 caused a GPU memory fault
(exit 139) and is rejected. Those exploratory arms require explicit --arm;
the default sweep excludes them. Each experimental arm should run in its
own container so GPU faults cannot corrupt a subsequent measurement.
Raw measurements are alongside this document. Model fidelity and real
>32K serving throughput still require separate end-to-end gates.

## Source provenance

- davetha/aiter-cdna2: 1552f41046b6000e766497f8638084de3208e154
- davetha/mi210-vllm: 8a8488801bd0122069e0653cb63ede979e22d29b
- davetha/mi210-llm-stack: eb0484d00a2e21ea1f1314de2b533718258b9499
- Local base: 5226dbdbd055b645e0a2e972ddb54e97b839858b

ASM BF16 KV attention is not a compatible substitute for inline INT8 G128.
The new kernels derive from this fork's gfx908 G128 Gluon implementations,
using the CDNA2 MFMA layout described by the MI210 source repositories.

## Local deployment

`docker/Dockerfile.mi210-long-context` extends the previously built and
qualified local image `int8-vllm-mi210:32k-vision-optimized`; it is an
incremental experiment image, not a clean-checkout standalone build.
Image: sha256:ea0be21b9b8fdb9390f521f7d451a35392c695e5843989c2e049622c4c6a83b3.
The launcher now accepts MAX_MODEL_LEN through 131072, default 32768.
A larger advertised limit does not prove sustained throughput or quality.

The snapshot patch records all tracked local changes against the local
base, including previous gfx90a integration. Untracked build support is
not completely represented by that patch. Do not apply it blindly over a
different revision; the kernel files and updated dispatch are also stored
as normal source files for review.

## Single-card dispatch correction

The actual TP1 model config has 24 Q heads and 4 KV heads. The original
Gluon host guard only admitted one KV head (the TP4 layout). gfx90a opt-in
dispatch is extended to four KV heads; default sweeps now use 24:4.
Real-layout numerical gates and end-to-end results are recorded separately.

## TP1 numerical gates

Actual 24 Q heads / 4 KV heads, 65537 KV tokens, q=512 prefill:
generic 123.554 ms vs native BF16 tile64 24.051 ms (5.14x).
Decode q=14: generic tile16 4.028 ms, generic tile32 4.074 ms,
generic tile64 10.717 ms (rejected), native tile32 1.929 ms,
native tile64 1.643 ms (2.45x vs default). Native launches were counted
(8 per timed case) to prevent fallback measurements being mislabeled.
The early decode rows are annotated as fallback measurements.

12 boundary cases pass the stricter <1% relative L2 gate (measured
0.23%-0.30%), in addition to elementwise closeness: target q=1/7/14,
KV lengths 33/4097/65537, partial pages, prefill q=257 and 2048, and
draft sliding-window q=14. Native decode masks the 4 unused MFMA rows
to prevent overlapping writes between neighboring 10-query CTAs.
Changing the native decode tile also changes the reduction tile, so
short sequences do not read uninitialized split outputs.
