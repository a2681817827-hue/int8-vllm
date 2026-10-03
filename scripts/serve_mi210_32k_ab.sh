#!/usr/bin/env bash
# Reproduce the user's vision baseline; vary only eager vs decode graphs.
set -euo pipefail
arm=${1:-baseline}
spec_tokens=${SPEC_TOKENS:-13}
[[ $spec_tokens =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || {
  echo 'SPEC_TOKENS must be in [1, 32]' >&2; exit 2;
}
batched_tokens=${BATCHED_TOKENS:-2048}
[[ $batched_tokens =~ ^[1-9][0-9]*$ ]] || {
  echo 'BATCHED_TOKENS must be a positive integer' >&2; exit 2;
}
max_model_len=${MAX_MODEL_LEN:-32768}
[[ $max_model_len =~ ^[1-9][0-9]*$ ]] && ((max_model_len <= 131072)) || {
  echo 'MAX_MODEL_LEN must be in [1, 131072]' >&2; exit 2;
}
concurrency=1
case "$arm" in
  baseline) execution=(--enforce-eager) ;;
  graph) execution=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","pass_config":{"fuse_allreduce_rms":false}}') ;;
  c8) concurrency=8; execution=(--enforce-eager) ;;
  graph-c8) concurrency=8; execution=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","pass_config":{"fuse_allreduce_rms":false}}') ;;
  graph-only) execution=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  graph-only-c8) concurrency=8; execution=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  *) echo 'Usage: bash scripts/serve_mi210_32k_ab.sh {baseline|graph|c8|graph-c8|graph-only|graph-only-c8} [--dry-run]' >&2; exit 2 ;;
esac
cache_args=()
experiment_args=()
if [[ -n ${MI210_UA_3D_MAXQ:-} ]]; then
  [[ $MI210_UA_3D_MAXQ =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || {
    echo 'MI210_UA_3D_MAXQ must be in [1, 32]' >&2; exit 2;
  }
  experiment_args=(-e "VLLM_UA_3D_MAXQ=$MI210_UA_3D_MAXQ")
fi
if [[ -n ${MI210_FLASH_SPLITS:-} ]]; then
  [[ $MI210_FLASH_SPLITS =~ ^(8|16|32|64)$ ]] || {
    echo 'MI210_FLASH_SPLITS must be 8, 16, 32 or 64' >&2; exit 2;
  }
  experiment_args+=(-e "VLLM_MI210_FLASH_SPLITS=$MI210_FLASH_SPLITS")
fi
if [[ -n ${MI210_JIT_CACHE:-} ]]; then
  cache_args=(-v "$MI210_JIT_CACHE:/mi210-jit" -e AITER_JIT_DIR=/mi210-jit)
fi
if [[ -n ${MI210_TRITON_CACHE:-} ]]; then
  cache_args+=(-v "$MI210_TRITON_CACHE:/root/.triton")
fi
if [[ -n ${MI210_COMPILE_CACHE:-} ]]; then
  cache_args+=(-v "$MI210_COMPILE_CACHE:/root/.cache/vllm")
fi
cmd=(docker run -d --name "${MI210_NAME:-int8-vllm-mi210-32k-$arm}"
  --restart unless-stopped --network host --device=/dev/kfd --device=/dev/dri
  --ipc=host -v "${MODEL_ROOT:-/data/models}:/models:ro"
  -v "${HF_CACHE_ROOT:-/data/cache/huggingface}:/root/.cache/huggingface"
  -e "HIP_VISIBLE_DEVICES=${MI210_GPU:-0}" -e GPU_ARCHS=gfx90a
  -e AITER_USE_SYSTEM_TRITON=1 -e VLLM_ROCM_USE_AITER=1
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 -e VLLM_GFX908_INT8_LM_HEAD=1
  -e VLLM_GFX908_ACT_QUANT=round -e VLLM_DISABLED_KERNELS=TritonW8A16LinearKernel
  "${cache_args[@]}"
  "${experiment_args[@]}"
  --entrypoint vllm "${MI210_IMAGE:-int8-vllm-mi210:32k-vision-baseline}"
  serve /models/Qwen3.8-27B-PTQR-R10S60 --served-model-name qwen-ptqr
  --host "${MI210_HOST:-10.168.1.4}" --port "${PORT:-8080}"
  --api-key "${MI210_API_KEY:-mi210-local-api}"
  --tensor-parallel-size 1 --dtype bfloat16 --kv-cache-dtype int8_block_g128
  --mamba-ssm-cache-dtype float32 --max-model-len "$max_model_len" --max-num-seqs "$concurrency"
  --max-num-batched-tokens "$batched_tokens" --gpu-memory-utilization 0.90
  --limit-mm-per-prompt '{"image":1,"video":0}' "${execution[@]}"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/dflash2-ptqr-r1\",\"num_speculative_tokens\":$spec_tokens,\"kv_cache_dtype\":\"int8_block_g128\"}")
if [[ ${2:-} == --dry-run ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  "${cmd[@]}"
fi
