# MI210 TP1 long-context attention and measured deployment

One MI210 gfx90a, 64 GB, 300 W. Native CDNA2 BF16 G128 Gluon attention
roughly doubles generation speed at 48K input and reduces uncached prefill
latency by about 4x in this measured PTQR target/draft configuration.
The running local service now advertises 131072 tokens and completed
three requests with 120081 input tokens plus 512 generated tokens each.

## Whole-model measurements

TP1, one benchmark client, server max_num_seqs=8 for decode graph capture,
DFlash NS13, BF16 compute, CK W8A8, G128 INT8 KV on target and draft,
FP32 Mamba state, 2048 batched tokens, one image allowed. Both arms use
FULL_DECODE_ONLY graphs with fuse_allreduce_rms disabled. Same weights,
prompt and API generation parameters; token counts come from API usage.

| Actual input / output | Path | TTFT, first / second / cached third (s) | Generation estimate, median (tok/s) | Wall output rate, second / third (tok/s) |
|---|---|---|---:|---|
| 48081 / 512 | generic MI210 reference | 152.42 / 151.04 / 18.04 | 42.58 | 3.14 / 17.04 |
| 48081 / 512 | native CDNA2 prefill + decode | 37.64 / 36.59 / 3.51 | 84.89 | 12.01 / 53.74 |
| 120081 / 512 | native CDNA2, 128K service | 145.16 / 143.13 / 5.42 | 47.41 | 3.32 / 31.61 |

Generation estimate = (completion_tokens - 1) / (last SSE time - first
content SSE time). Speculative chunks may contain several tokens. This
excludes TTFT and must not be presented as end-to-end wall throughput.
The third repetition reused aligned prefix state; it is not a cold prefill
measurement. No generic 120K baseline was run, so no speedup is claimed
for that row. Synthetic repeated-background prompts are not a real-world
throughput distribution. First rounds include inference-shape JIT.

The 48K A/B image was ea0be21b9b8fdb9390f521f7d451a35392c695e5843989c2e049622c4c6a83b3.
Final deployed image is 8df86c75bfb56d3654bca7dee59838c323b6250eb7649a0f28de9c13c930d79a.
The final helper/cache packaging changed; the three attention source blobs
are identical in both images and the GitHub snapshot:

- unified dispatch: fe9751b14e65f3587298f3a64c3bb1d1063f50f4
- native prefill: 226c84adefbd1139b9181aba22bee1f1b23e7066
- native decode: 16128c16062410534dfdbcac2339fdc4fba10de5

## Operator selection and validation

Actual TP1 geometry is 24 Q heads / 4 KV heads, head dimension 256,
GQA 6:1. Old gfx908 dispatch only admitted the TP4 geometry (6:1 heads).
The new opt-in gfx90a dispatch also admits 24:4. Kernels use AMD MFMA v2
16x16x16, retain BF16 operands, and load the existing inline INT8 KV plus
FP16 G128 scales. Physical block offsets are widened to int64.

Reversed physical pages, page size 1728, causal masking, HIP event timings,
2 warmups, median of 5, FP32 attention reference:

| 65537 KV tokens, TP1 | Generic | Native BF16 CDNA2 tile64 |
|---|---:|---:|
| Prefill q=512 | 123.554 ms | 24.051 ms (5.14x) |
| Decode / verify q=14 | 4.028 ms | 1.643 ms (2.45x) |

The sweep counts native kernel launches to reject accidental fallback.
MI210 default decode tile16 originally prevented native dispatch; early
"CDNA2 decode" rows are explicitly annotated as generic fallback and are
not used as evidence for the candidate. Native decode needs VLLM_UA_TILE=32;
its own KV tile64 is passed separately, including to split reduction.
The four unused MFMA rows are masked to avoid writes overlapping the next
10-query CTA. Native tile64 divides the real 1728-token hybrid-model page.

12 target/draft boundary cases pass elementwise closeness plus the
stricter relative L2 error <1% (observed 0.23%-0.30%): target q=1/7/14 at
KV lengths 33/4097/65537, partial final pages, prefill q=257 and q=2048,
and draft sliding-window q=14. The original elementwise absolute tolerance
alone was too loose for small long-context outputs; L2 prevents a zero
output from passing. All raw values are included.

Rejected candidates: generic prefill BLOCK_M=64/TILE=64 caused a GPU memory
fault (exit139); generic decode tile64 passed numerics but took 10.717 ms.
Unsafe exploratory prefill arms are excluded from the default sweep and
must be isolated in their own containers. Native tile32 decode measured
1.929 ms, slower than native tile64. Legacy BF16 G128 Gluon and packed
prefill are also recorded. ASM BF16-KV attention cannot replace this inline
G128 INT8 representation and was not substituted.

## Functional gates

- Arithmetic and capital answers pass.
- Red-image recognition passes on final 128K service.
- 16 distinct additions at client concurrency8 all return exact answers.
- Actual 61287-token retrieval on 64K service returns MI210-7391.
- Actual 120037-token retrieval on final 128K service returns MI210-7391.
- All three 120081-token generation requests complete and contain the code.
- 12 existing helper/port unit tests pass in final image; source compilation
  and launcher shell syntax checks pass.

These are operator numerics and functional smoke gates, not a full model
fidelity or downstream task evaluation. Benchmark output is capped at
512 tokens and may include model reasoning. No dual-card measurements are
possible on this host, which currently exposes one MI210.

## Local use and reproduction

Running container: int8-vllm-mi210-ptqr. Host/port: 10.168.1.4:8080.
Image tag: int8-vllm-mi210:long-context-cdna2. Old 32K container retained as
int8-vllm-mi210-32k-backup-20261003; 64K reference/candidate containers are
stopped and retained. No models or old images were deleted.

```bash
docker logs -f --tail 100 int8-vllm-mi210-ptqr
# To create a new instance when the name/port is free:
bash scripts/serve_mi210_long_context.sh graph-c8
# Defaults to MAX_MODEL_LEN=131072; can select 65536 for 64K.
```

The launcher retains the user's model mounts, API key configuration,
vision/tool-call options, target/draft quantization and FP32 Mamba state.
Compile and Triton caches persist in separate native-CDNA2 directories.
Advertised max_num_seqs=8 is not eight simultaneous full 128K contexts;
startup reports capacity equivalent to 2.76 full-length requests for this
specific deployment. Scheduler memory availability still controls admission.

The Dockerfile is an incremental local build extending the previously
qualified image int8-vllm-mi210:32k-vision-optimized. That local base image
is required; this branch does not publish an image registry artifact or
claim a standalone clean-checkout build. Model files remain local.

```bash
docker build -f docker/Dockerfile.mi210-long-context \
  -t int8-vllm-mi210:long-context-cdna2 .
# Run inside the serving environment; set the benchmark API key via env.
python docker/mi210/make_long_prompt.py --tokens 48000 --output /tmp/prompt.txt
python docker/mi210/benchmark.py --url http://10.168.1.4:8080 \
  --model qwen-ptqr --prompt-file /tmp/prompt.txt --concurrency 1 \
  --requests 1 --max-tokens 512 --repetitions 3 --output /tmp/bench.jsonl
```

The compressed local-tracked-integration.patch.gz records tracked changes
against local base 5226dbdbd055b645e0a2e972ddb54e97b839858b, including the
previous gfx90a integration. It is an archival patch; untracked historical
build support is not fully represented. Check revisions before applying.
The native kernels, current dispatch, Dockerfile and helpers are also
stored as ordinary files for review and modification.

For later two-card work, compare two isolated TP1 replicas with TP2 on the
actual interconnect. The current kernel is validated at TP1; TP2's 12:2
geometry uses existing fallback. Both numerical and end-to-end gates must
be rerun. No linear scaling or globally optimal dual-card claim is made.

## Source provenance

- [davetha/aiter-cdna2](https://github.com/davetha/aiter-cdna2):
  1552f41046b6000e766497f8638084de3208e154
- [davetha/mi210-vllm](https://github.com/davetha/mi210-vllm):
  8a8488801bd0122069e0653cb63ede979e22d29b
- [davetha/mi210-llm-stack](https://github.com/davetha/mi210-llm-stack):
  eb0484d00a2e21ea1f1314de2b533718258b9499

The new kernels derive from this int8-vllm fork's existing gfx908 G128
Gluon implementations; native CDNA2 layout selection is informed by the
MI210 operator/recipe repositories above. All precision contracts remain.
