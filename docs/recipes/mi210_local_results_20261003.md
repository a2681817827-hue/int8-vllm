> 最终部署已升级为 CDNA2 算子、128K 上限；实际 48K 输入约 84.9 token/s、120K 输入约 47.4 token/s。详见 [长上下文实测](mi210-long-context/README.md)。下文原 32K 部署状态为历史记录。

# MI210 local Docker results, 2026-10-03

The host exposes one gfx90a GPU with 64 GB HBM and a 300 W power cap.
Dual-card results cannot be measured on this host yet.

The integrated source build is available as
`int8-vllm-mi210:qualified-20261003`, image ID
`sha256:f99481e53da7d19d6dadc697be94dd1a8eebc235af330e4a06b70a710057d9d6`.
It retains the local ROCm 7.2 / PyTorch 2.13 dependencies and compiled native
extensions rather than replacing them with the source projects' dependencies.

## Gates

- Tier 2 qualification passes CK INT8 numerical reference, INT8 KV micro and
  nine grouped INT8 attention cases.
- The extended grouped INT8 attention reference passes 14 cases including
  32k decode/verification with the hybrid model's 1728-token page size.
- All 12 integration, streaming proxy and benchmark regression tests pass
  inside the integrated image.
- Integrated C1 serving smoke returns `42`, `Paris`, `Red`, and the remembered
  code in a 30,037-token request. This is smoke coverage, not a fidelity eval.
- Target and DFlash2 graph capture actually completed. The launch log confirms
  G128 INT8 KV on both models and gfx90a CK INT8 dispatch.

## C1 measurements

The same short text prompt (62 prompt tokens), greedy decoding, 512 output
tokens per request, four sequential requests per repetition and three
repetitions were used. Figures are wall output tokens per second including
prefill and client overhead; they are not Reddit's reciprocal-TPOT TG metric.
Earlier baseline and graph-only runs are retained local measurements, not
newly interleaved control arms for the integrated run.

| Image/config | Raw repetitions, tokens/s | Median |
|---|---|---|
| original vision baseline, eager/C1 | 33.39, 33.29, 33.76 | 33.39 |
| original vision baseline, graph-only/C1 | 88.77, 88.97, 89.00 | 88.97 |
| qualified integrated, compiled decode graphs/C1 | 108.65, 114.62, 114.60 | 114.60 |

The integrated run's median is 3.43 times the recorded eager baseline.
This comparison includes both source/image and execution-mode changes;
it does not isolate an individual patch's benefit or establish a global
optimum. No full target-oracle fidelity gate has been run for these outputs.

The 30k retrieval smoke took 64.92 seconds on its first run, including
first-use shape execution. It emitted only ten output tokens; this is not
a long-context decode throughput benchmark. Eight full 32k contexts also
cannot be assumed to fit simply because C8 is configured.

Raw artifacts are under `logs/mi210/`: `baseline-c1.jsonl`,
`graph-only-c1.jsonl`, `integrated-graph-c1.jsonl`,
`integrated-graph-smoke.json`, `integrated-graph-server.log`,
`qualification-20261003.json`, and `grouped-int8-32k-reference.log`.

## Reproduce the C1 configuration

```bash
MI210_IMAGE=int8-vllm-mi210:qualified-20261003 \
MI210_NAME=int8-vllm-mi210-integrated-graph \
bash scripts/serve_mi210_32k_ab.sh graph
```

Stop the previous test container before allocating the GPU to another arm.
The script preserves 32k context, one image, the local PTQR target/draft,
FP32 Mamba state, NS13 and C1. The corresponding C8 arm is `graph-c8`.
The two-card `replicas2` and `tp2` launchers remain candidates requiring
actual two-card topology, numeric and matched performance gates.

## C8 measurements and selected local service

The qualified image's compiled graph/C8 service completed three repetitions
of eight concurrent requests, each with the same 62-token text input and
512-token output budget. Wall throughput was 320.98, 321.46 and 348.11
tokens/s (median 321.46). Mean TPOT estimates were 20.47, 21.76 and 20.58 ms;
the `8 / mean_TPOT_seconds` aggregate proxy was 390.74, 367.71 and 388.66.
These proxies are not wall throughput and do not establish an exact match
to the unknown Reddit comment's benchmark procedure.

On this same C8 server with only one sequential active client, three
repetitions of four requests measured 115.33, 115.43 and 115.19 wall tokens/s
(median 115.33). Thus this short workload does not show a single-client
speed penalty from allowing C8. Arithmetic, capital and image-color smoke
also pass on C8.

The initial C8 test service was `int8-vllm-mi210-integrated-graph-c8`,
listening at `http://10.168.1.4:8080`, with the existing API key and served
model `qwen-ptqr`. It retains the user-provided vision/32k/PTQR parameters,
changes eager execution to compiled decode graphs and permits eight
requests. It establishes a control arm for the threshold experiment below,
not an exhaustive optimum or a completed fidelity evaluation.

The image reports KV capacity of 179,869 tokens and only 5.49 times a full
32k context. C8 permits eight short requests; it does not promise eight
simultaneous full-32k contexts. Draft vision embeddings are not supported
by this DFlash2 model; the target handles images while the draft uses text.

Build the packaged tooling layer without recompiling the qualified native
extensions:

```bash
docker build -f docker/Dockerfile.mi210-32k-optimized \
  -t int8-vllm-mi210:32k-vision-optimized .
```

The live service uses the qualified base image; the optimized image adds
the authenticated benchmark and launch tools and the result documentation.
The final packaged image also sets the numerically checked
`VLLM_UA_3D_MAXQ=16`, as described below. It does not change model weights.
Extra raw artifacts:
`integrated-graph-c8.jsonl`, `integrated-graph-c8-client-c1.jsonl`,
`integrated-graph-c8-smoke.json` and `integrated-graph-c8-server.log`.

## NS13 verification split-K threshold experiment

The grouped INT8 backend's default `VLLM_UA_3D_MAXQ=8` excludes the
14-query target verification pass used by NS13. The numeric reference suite
already covers threshold 16, including actual 32k KV pages. A matched local
arm changes this one serving setting to 16 and reuses the compiled cache.

| Measurement | Default threshold | Threshold 16 |
|---|---|---|
| short text C8 wall throughput, three repetitions | 320.98, 321.46, 348.11 | 344.83, 345.81, 345.18 |
| 30,065-input / 256-output, C1 decode proxy tokens/s | 20.40, 20.39, 20.41 | 55.34, 55.35, 55.45 |
| corresponding TTFT seconds | 64.31, 64.62, 9.29 | 64.30, 64.23, 9.18 |

The decode proxy is the reciprocal of the streaming TPOT estimate and
excludes the first-token wait. It is not the whole-request throughput:
threshold-16 long-request wall output rates are 3.71, 3.72 and 18.58 tokens/s.
The ~2.71x gain concerns long-context decode, not prefill. Cache behavior
changes TTFT between repetitions; report individual samples rather than
averaging cold and warm first-token latency into a prefill claim.

Threshold 16 also passes arithmetic, capital, image-color, and 30k retrieval
smoke. This does not replace a full target-oracle quality gate. The final
selected service `int8-vllm-mi210-ptqr` uses threshold 16, C8, NS13,
grouped INT8 KV on both models,
compiled decode graphs, and the user's original 32k/vision settings.

Start/reproduce it after stopping a previously running test arm:

```bash
bash scripts/serve_mi210_optimized.sh graph-c8
docker logs -f --tail 100 int8-vllm-mi210-ptqr
```

The wrapper uses `int8-vllm-mi210:32k-vision-optimized`, serves `qwen-ptqr`
on `10.168.1.4:8080`, and persists compilation under
`/data/cache/mi210-vllm/qualified-c8`. The API key remains configurable using
`MI210_API_KEY`. The current live container was launched from the qualified
base with the same threshold-16 setting; the packaged image preserves the
same serving code. `graph` selects C1 when a concurrency cap of one is wanted.

Raw threshold artifacts: `integrated-graph-c8-q16.jsonl`,
`integrated-graph-30k-q16.jsonl`, `integrated-graph-c8-q16-smoke.json`,
and `integrated-q16-server.log`.

On the final threshold-16/C8 server, short-text sequential C1 wall throughput
was 107.12, 106.95 and 105.42 tokens/s (median 106.95). This is ~7.3% below
the threshold-8 C1 median of 115.33, while the long-context decode proxy
improves ~2.71x. Threshold 16 is selected for the requested 32k-capable
deployment; short-only C1 users can compare threshold 8 via
`MI210_UA_3D_MAXQ=8`. This measured tradeoff precludes claiming one setting
is universally fastest.
