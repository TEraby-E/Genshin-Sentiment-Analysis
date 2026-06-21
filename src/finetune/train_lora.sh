#!/usr/bin/env bash
# =============================================================================
# QLoRA 微调便捷封装：调用自包含的 src/finetune/train_lora.py
# （只用 transformers + peft + bitsandbytes，不依赖 LLaMA-Factory，云端更稳）
#
# 前置：
#   uv sync --extra finetune        # 注意 torch 需与云端 CUDA 匹配（用镜像自带的或官方 cuXXX 源）
#   uv run python scripts/build_finetune_dataset.py --sample 1800   # 生成训练集（或上传已标注的）
#
# 用法：
#   bash src/finetune/train_lora.sh
#   BATCH_SIZE=4 EPOCHS=3 bash src/finetune/train_lora.sh
#   FLASH_ATTN=fa2 bash src/finetune/train_lora.sh   # 装了 flash-attn 才用，否则默认 sdpa
# =============================================================================
set -euo pipefail

# 国内服务器拉权重走镜像，避免 huggingface.co 不通（脚本内也有兜底）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run python -m src.finetune.train_lora \
    --epochs "${EPOCHS:-3}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --grad-accum "${GRAD_ACCUM:-8}" \
    --cutoff-len "${CUTOFF_LEN:-1024}" \
    --lora-rank "${LORA_RANK:-8}"
