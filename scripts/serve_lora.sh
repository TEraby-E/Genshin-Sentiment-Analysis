#!/usr/bin/env bash
# =============================================================================
# 场景 B｜云端把微调后的 Qwen 起成 OpenAI 兼容服务（vLLM）
#
# 在 AutoDL / RunPod 等 GPU 实例上运行：把基座 + LoRA 适配器加载成一个
# OpenAI 兼容端点，本地只需把 .env 的 LORA_SERVER_BASE_URL 指过来即可调用。
#
# 前置：
#   pip install "vllm>=0.5"
#   # 已在云端 git clone 本项目并跑完 train_lora.sh，产物在 outputs/finetune/...
#
# 用法：
#   bash scripts/serve_lora.sh
#   PORT=8000 LORA_SERVER_API_KEY=mysecret bash scripts/serve_lora.sh
#
# 暴露公网（任选其一，便于本地直连）：
#   cloudflared tunnel --url http://localhost:8000
#   ngrok http 8000
# 然后把得到的 https 地址 + /v1 填进本地 .env 的 LORA_SERVER_BASE_URL。
# =============================================================================
set -euo pipefail

# 国内服务器拉基座权重同样需要走 HF 镜像，避免 Network is unreachable。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL="${LORA_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER="${LORA_ADAPTER_DIR:-outputs/finetune/qwen2.5-7b-lora}"
SERVED_NAME="${LORA_SERVER_MODEL:-qwen2.5-7b-lora}"
PORT="${PORT:-8000}"
API_KEY="${LORA_SERVER_API_KEY:-EMPTY}"

# 7B 以 bf16 起服务约需 ~16GB 显存，单张 4090(24G) 富余；
# 若显存吃紧，追加：--quantization bitsandbytes --load-format bitsandbytes
vllm serve "${MODEL}" \
    --enable-lora \
    --lora-modules "${SERVED_NAME}=${ADAPTER}" \
    --max-lora-rank 8 \
    --max-model-len 4096 \
    --port "${PORT}" \
    --api-key "${API_KEY}"

echo "✅ 服务已启动：base_url=http://<本机或隧道地址>:${PORT}/v1，model='${SERVED_NAME}'"
echo "   本地 .env 配置：LORA_SERVER_BASE_URL / LORA_SERVER_MODEL / LORA_SERVER_API_KEY"
