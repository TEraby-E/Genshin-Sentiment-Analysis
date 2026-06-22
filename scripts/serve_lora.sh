#!/usr/bin/env bash
# =============================================================================
# 场景 B｜云端把微调后的 Qwen 起成 OpenAI 兼容服务（自包含端点，不依赖 vLLM）
#
# 复用已验证可跑的 LocalLLMClassifier（transformers + peft）做推理，对外暴露
# /v1/chat/completions。规避 vLLM 与较新 torch/CUDA（如 CUDA 13）的导入期冲突。
#
# 前置：
#   uv sync --extra finetune --extra serve     # finetune=torch/transformers/peft；serve=fastapi/uvicorn
#   # 已在云端跑完 train_lora.sh，适配器在 outputs/finetune/qwen2.5-7b-lora
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

# 国内服务器拉基座权重走 HF 镜像，避免 Network is unreachable。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PORT="${PORT:-8000}"
SERVED_NAME="${LORA_SERVER_MODEL:-qwen2.5-7b-lora}"

uv run python -m src.finetune.serve \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT}" \
    --served-model "${SERVED_NAME}"
