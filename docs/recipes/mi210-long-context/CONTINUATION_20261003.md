# 单卡长上下文优化续记：80 token/s 的证据边界

本轮重新计算已有原始 JSONL，并准备下一轮深度测试工具。没有新增 GPU
性能实测：当前会话无 `/dev/kfd`、`/dev/dri`，Docker socket 拒绝访问，
原服务 `10.168.1.4:8080/health` 连接失败。此前部署状态不能当作当前在线证明。

## 核验结果

| 实际输入 token | 算子 | 三次解码估计 token/s | 中位数 | 达到 80 |
|---:|---|---|---:|---|
| 48081 | generic | 42.943 / 42.573 / 42.579 | 42.579 | 否 |
| 48081 | native CDNA2 BF16 G128 | 87.679 / 84.790 / 84.891 | 84.891 | 三次均达到 |
| 120081 | native CDNA2 BF16 G128 | 48.534 / 46.747 / 47.408 | 47.408 | 否 |

48K 的中位数比值约 1.994 倍，已经符合“超过 32K 的长上下文从约 40 到
80 token/s”的特定工作负载目标。尚不能声称 32K 以上所有深度都超过 80。
120K 距离 80 仍需约 1.687 倍提升。131072 是配置上限；120081 输入加
512 输出是现有生成测试的最大已验证长度，不能据此宣称已找出单卡最大极限。

上述速度使用已有 `tpot_estimate_seconds` 的倒数，排除 TTFT，受投机 SSE
分块影响。48K 候选整请求速度为 11.777 / 12.014 / 53.745 token/s，
120K 为 3.289 / 3.323 / 31.607 token/s。第三次复用前缀，不能算冷启动
prefill。现有 A/B 并非本轮新增交错测试；合成重复背景也不代表所有业务文本。

## 当前最有证据支持的算子

原生 CDNA2 prefill + decode，BF16 MFMA v2 16x16x16，沿用 G128 INT8 KV
和 FP16 scales；TP1 的 24 Q / 4 KV heads、head_dim=256、GQA=6:1。
现有 65537-token 算子测试中，q=512 prefill 123.554 → 24.051 ms，
q=14 验证 4.028 → 1.643 ms。算子测试计数原生 launch，排除 fallback。
数值和功能门槛详见原 README；本轮没有重新执行 GPU 数值测试。

部署设置必须一起保留：

```text
VLLM_UA_3D_MAXQ=16
VLLM_UA_TILE=32
VLLM_G128_GLUON=64
VLLM_G128_GLUON_MMA=bf16
VLLM_G128_DECODE_CDNA2=1
VLLM_G128_DECODE_CDNA2_TILE=64
VLLM_G128_PREFILL_GLUON=1
VLLM_G128_PREFILL_GLUON_MMA=bf16
VLLM_G128_PREFILL_CDNA2=1
VLLM_G128_PREFILL_CDNA2_TILE=64
```

`VLLM_UA_TILE=32` 是进入原生路径的条件，原生 KV tile=64 是另一个参数。
NS13 有 14 个验证 query，阈值必须覆盖它。当前证据不支持改用不兼容
G128 布局的 ASM BF16-KV，也不支持重试曾 exit139 的 generic prefill arm。

## 下一轮工具和验证

新增 `docker/mi210/depth_campaign.py`，默认打印计划，加 `--execute` 才发送
请求。测试 32768 / 49152 / 65536 / 98304 / 120000 背景 token，每个深度
C1、512 输出、三次重复。支持多个 URL，通过已有 benchmark 轮换 A/B
次序。每个深度结束检查 API 实际输入不低于指定深度、完整输出预算、记录数；
已有结果文件会拒绝覆盖或追加。不同深度都复用相同背景，因此缓存状态必须
结合原始 TTFT 和服务指标单独判读；该工具不保证冷缓存。

在已有模型、依赖和服务可用的环境，从仓库根目录执行：

```bash
python docker/mi210/depth_campaign.py \
  --url http://10.168.1.4:8080 --label cdna2 \
  --output-dir /tmp/mi210-depth-campaign
# 审阅计划后，同一命令加 --execute；API key 使用 MI210_API_KEY 环境变量。
python docker/mi210/summarize_depth.py \
  /tmp/mi210-depth-campaign/depth-*.jsonl \
  --output /tmp/mi210-depth-campaign/summary.json
```

`summarize_depth.py` 仅重算记录，按实际输入、输出、模型、镜像、标签分组；
拒绝 C8、低于 32K 输入、无效时间和重复 repetition。输出各次解码估计、
中位数、最小值、TTFT、整请求速度和 80 token/s 判定，不自动背书模型质量。
本轮检查通过：两个脚本语法；五档计划；已有九条样本重算；80 阈值边界；
C8 / 低于 32K / NaN / 重复记录拒绝检查。

120K 下一步需在 GPU profiler 中确定验证 attention、权重 GEMM、draft 和
调度各自占比，再测 split-K 分区与 KV tile。任何新的分区策略必须验证
graph replay 长度覆盖、尾块掩码和 ragged batch，且 whole-model A/B 后
才可部署。目前没有足够新证据选择另一算子或承诺 120K 达到 80 token/s。

机器可读核验结果：`depth-audit-20261003.json`。原始数据沿用同目录
`generic-48k.jsonl`、`cdna2-48k.jsonl`、`cdna2-120k.jsonl`。
