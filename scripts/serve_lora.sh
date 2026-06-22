#!/usr/bin/env bash
# =============================================================================
# 场景 B｜云端把微调后的 Qwen 起成 OpenAI 兼容服务（纯标准库端点，零额外依赖）
#
# 复用已验证可跑的 LocalLLMClassifier（transformers + peft）做推理，对外暴露
# /v1/chat/completions。端点用 Python 标准库 http.server 实现，**不需要安装
# fastapi/uvicorn/vLLM**，直接跑在已装好 torch 的那个环境里，避开依赖同步问题。
#
# 前置：已在云端跑完 train_lora.sh，适配器在 outputs/finetune/qwen2.5-7b-lora
#
# 用法：
#   bash scripts/serve_lora.sh
#   PORT=8000 LORA_SERVER_API_KEY=mysecret bash scripts/serve_lora.sh
#
# 只给本地用：在笔记本开 SSH 隧道把 localhost:8000 接到实例（见 docs/CLOUD_LORA.md）：
#   ssh -p <端口> root@<host> -L 8000:localhost:8000 -N
# 本地 .env 填 LORA_SERVER_BASE_URL=http://localhost:8000/v1 即可。
# =============================================================================
set -euo pipefail

# 国内服务器拉基座权重走 HF 镜像，避免 Network is unreachable。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PORT="${PORT:-8000}"
SERVED_NAME="${LORA_SERVER_MODEL:-qwen2.5-7b-lora}"

# --no-sync：直接在当前 .venv 里运行，不让 uv 重新同步环境（避免动到已装好的 CUDA torch）。
# 若你不用 uv、就在已装 torch 的 conda 环境里，可把下面这行换成：python -m src.finetune.serve ...
uv run --no-sync python -m src.finetune.serve \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT}" \
    --served-model "${SERVED_NAME}"
