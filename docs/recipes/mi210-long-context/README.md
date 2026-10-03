# MI210 single-card long-context experiment (2026-10-03)

Status: operator numerics and latency gated; end-to-end 64K A/B pending.
One gfx90a MI210, 64 GB, 300 W. Preserve PTQR target/draft, CK W8A8,
inline grouped INT8 KV with FP16 per-group scales, BF16 attention operands,
and FP32 Mamba state. No dual-card measurements are possible on this host.

## Changes

Native CDNA2 AMD MFMA v2 16x16x16 layouts in grouped INT8 Gluon prefill,
with optional 32/64/128 KV tiles; host dispatch restricted to gfx90a and
explicit opt-in. The new image opts into BF16/64-column prefill.
A CDNA2 decode candidate is also recorded, but not enabled by the image:
it showed no clear gain in the 64K synthetic split-attention test.
The generic attention path remains available. Unsupported layouts use
existing dispatch; the opt-in tile must divide the physical block size.

## Measured operator results

Reverse-ordered physical pages, page size 1728, head dimension 256,
GQA 6:1, causal masking, inline G128 scale bytes. CUDA/HIP event timing,
2 warmups and median of 5. Numerical comparison to FP32 attention.

| Context / query | Path | Median ms | Max absolute error |
|---|---|---:|---:|
| 49152 / 512 | generic | 30.415 | 4.00e-5 |
| 49152 / 512 | legacy Gluon BF16 | 12.843 | 4.00e-5 |
| 49152 / 512 | CDNA2 BF16 tile32 | 11.047 | 4.00e-5 |
| 49152 / 512 | CDNA2 BF16 tile64 | 8.965 | 4.00e-5 |
| 65537 / 2048 | CDNA2 BF16 tile64 | 24.019 | 3.32e-5 |
| 65536 / 14 split | generic | 1.729 | 2.26e-5 |
| 65536 / 14 split | CDNA2 decode | 1.707 | 2.26e-5 |

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
Image: sha256:1e362354664e1c8080b9481cef1dfaf57bf24c35a91432ffda2f887f1228c4b0.
The launcher now accepts MAX_MODEL_LEN through 131072, default 32768.
A larger advertised limit does not prove sustained throughput or quality.

The snapshot patch records all tracked local changes against the local
base, including previous gfx90a integration. Untracked build support is
not completely represented by that patch. Do not apply it blindly over a
different revision; the kernel files and updated dispatch are also stored
as normal source files for review.
