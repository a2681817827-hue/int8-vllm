# 48K W8A8：冲刺 90–100 token/s 的分区实验

状态：**实验代码和 CPU 检查已完成，GPU 数值、图重放与吞吐尚未验证。**
本轮仍缺少 GPU 设备和 Docker 权限，原服务 health 连接失败。没有新增
90 或 100 token/s 结果，也没有更改正在运行的服务或默认 split 策略。

基线保持单 MI210、TP1、PTQR target/draft、CK W8A8 INT8 GEMM、
target/draft G128 INT8 KV、BF16 计算、FP32 Mamba、NS13、C8 服务/C1 客户端。
已有实际 48081 输入/512 输出的三次解码估计是 87.679 / 84.790 / 84.891，
中位数 84.891 token/s。90 需要约 +6.02%，100 需要约 +17.80%。这些目标
是排除 TTFT 的生成估计，整请求吞吐另报。

## 为何先试 split-K

`_compute_flash_decoding_splits` 的 base_grid 使用 `num_seqs * num_kv_heads`，
默认 target_cus=120，没有显式计入 verification query blocks。TP1 C1、
4 KV heads 通常选 32 splits；NS13 的原生 core 使用 query_block=10，
多个 query blocks 会进一步扩大实际 grid。较少 splits 可能减少部分输出
和 reduction 开销，更多 splits 可能改善长扫描延迟。哪个更快需要测量，
仅凭源码不能证明默认过细，也不能保证收益。

新增 `VLLM_MI210_FLASH_SPLITS=8|16|32|64`。只作用于 gfx90a、TP1
24:4 头、head_dim256、causal、无 sliding window、G128 INT8 的 3D 路径，
且 CDNA2 decode 已开启。不会改变 draft SWA 或默认配置。分区数必须
不超过 output/max/expsum 三个 scratch buffer 的最小容量；越界直接报错。
选择为启动环境常量，必须在新进程完成图捕获，不能在已捕获图之间切换。

分区变化会改变浮点归约顺序，需要质量门槛；W8A8 和 KV 精度不变不等于
生成逐 token 必然一致。本轮没有发布新的最快配置。

## GPU 验证步骤

从本分支构建独立实验镜像，保留原镜像和服务：

```bash
docker build -f docker/Dockerfile.mi210-long-context \
  -t int8-vllm-mi210:48k-split-experimental .
```

必须在可用 GPU 上、同一个实验镜像中执行原生算子 reference。默认臂要
指定 `--segment-capacity 64`，与服务 scratch 容量一致；过去默认 micro
只有 16，不能拿来与生产 split 数直接比较。以下为容器内命令：

```bash
.venv/bin/python docker/mi210/operator_sweep.py \
  --context 48081 --query 14 --kv-heads 4 --split --segment-capacity 64 \
  --arm gluon_cdna2_decode_tile64 --output /tmp/default-split.jsonl
for splits in 8 16 32 64; do
  .venv/bin/python docker/mi210/operator_sweep.py \
    --context 48081 --query 14 --kv-heads 4 --split --target-splits "$splits" \
    --arm gluon_cdna2_decode_tile64 --output "/tmp/split-$splits.jsonl" || exit
done
```

还需 q=1/7/14、KV=33/4097/48081/65537、ragged batch、尾块、实际
page_size1728，以及短长度 graph capture 后长长度 replay 的比较。现有
micro 工具不覆盖 ragged batch 和 graph replay，不能只凭它部署候选。
原生 launch 计数必须非零，L2 数值门槛通过；本轮也修正了 operator_sweep
记录 FAIL 后仍 exit0 的问题，现在数值失败会终止后续实验。

候选每次单独启动，确保原容器已由 GPU 操作者安排资源，别在同一张卡并发
跑 A/B。环境示例（需先完成前述 GPU 门槛）：

```bash
MI210_IMAGE=int8-vllm-mi210:48k-split-experimental \
MI210_NAME=mi210-48k-split16 PORT=8106 MI210_FLASH_SPLITS=16 \
MI210_COMPILE_CACHE=/data/cache/mi210-vllm/split16 \
bash scripts/serve_mi210_long_context.sh graph-c8
```

保持原 max_model_len=131072 和所有模型选项，避免同时改变上下文配置。
取消 `MI210_FLASH_SPLITS` 是同镜像默认策略的控制臂。原生 tile64、
UA tile32、MAXQ16、NS13 均固定。先 warmup，再每臂至少五次 C1、512
输出；跑 baseline/candidate/baseline，记录真实输入、镜像和服务环境。
单 GPU 通过串行换臂，双 URL 工具不意味着两臂可同时驻留同卡。

```bash
python docker/mi210/depth_campaign.py \
  --url http://10.168.1.4:8106 --label cdna2-split16 \
  --depths 48000 --repetitions 5 --image-id IMAGE_SHA256 \
  --output-dir /tmp/48k-split16 --execute
python docker/mi210/summarize_depth.py /tmp/48k-split16/depth-48000.jsonl \
  --target 90 --output /tmp/48k-split16/target90.json
python docker/mi210/summarize_depth.py /tmp/48k-split16/depth-48000.jsonl \
  --target 100 --output /tmp/48k-split16/target100.json
```

精确 API 输入预计含 chat template，需确认与现有 48081 基线一致。记录
投机 acceptance、实际 accepted tokens、TTFT、图捕获和缓存命中指标。
输入文本与输出长度一致，并增加非重复真实文本与 retrieval 检查，防止只
挑一个高 acceptance 提示词。中位数和最慢样本分别报告；单次峰值不算达标。

本轮 CPU 检查：4 个 unittest 通过（合法 split/容量检查，默认保持，draft
和非 gfx90a 排除，最小 buffer 容量，launcher 参数与错误拒绝），shell
语法和 diff whitespace 通过。GPU 验证待执行；生产默认不变。

## 达标后保存 Docker 与全部算子

新增 `archive_qualified.py`，只接受单候选至少五个 C1 样本、实际输入
>=48000、输出>=512、每个样本均达到指定 90 或 100 阈值，且 benchmark
镜像 ID 与待保存容器一致。GPU 数值、graph replay、质量三项必须在独立
JSON 报告中标为 PASS。报告由 GPU 测试者根据实际结果填写，不能用 CPU
测试代替。这是归档条件，不是本轮已达标的证明。

```bash
# IMAGE_SHA256 来自 docker inspect --format '{{.Image}}' mi210-48k-split16
python docker/mi210/archive_qualified.py /tmp/48k-split16/depth-48000.jsonl \
  --target 90 --container mi210-48k-split16 \
  --output-dir /data/artifacts/mi210-48k-qualified
# 先检查计划；实际归档添加 --gpu-gate-report /tmp/gpu-gates.json --execute。
```

归档包含镜像 `image.tar`、完整 vLLM/AITER package（含已存在二进制）、
运行时可见的 AITER JIT/Triton/vLLM 缓存、测试数据、GPU 门槛报告、SHA256
校验、镜像 ID 和脱敏启动参数。只复制实际存在的缓存目录；镜像本身也保存
安装在镜像内的其它算子。外部模型权重与 API key 需独立提供。
若任一步失败，manifest 保持 exporting；只有全部成功才标 complete。
本轮没有执行 Docker 导出或声称归档已完成。

恢复镜像用 `docker load -i image.tar`。最终启用参数从获胜归档的
`manifest.json.operator_env` 获取，尤其是实测获胜的
`VLLM_MI210_FLASH_SPLITS`。此前 CDNA2 参数见 CONTINUATION_20261003.md；
若默认策略获胜则不设置分区 override。不能预先将 split16 称为最快。
