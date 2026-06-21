#!/usr/bin/env bash
# =============================================================================
# QLoRA 微调模板（LLaMA-Factory）
#
# 适配场景：消费级 GPU（如 RTX 4090, 24GB）。7B + 4-bit QLoRA 显存充裕，
# 配置以稳定可跑、训练高效为先。
#
# 前置：
#   1) pip install "llamafactory[torch,metrics]"  flash-attn  bitsandbytes
#   2) uv run python -m src.finetune.dataset_formatter   # 生成 JSONL + dataset_info.json
#   3) 把 outputs/finetune/dataset_info.json 的条目并入 LLaMA-Factory 的 data/dataset_info.json，
#      并将 genshin_sentiment.jsonl 拷到其 data/ 目录（或用 --dataset_dir 指向本仓库）。
#
# 用法：
#   bash src/finetune/train_lora.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct OUTPUT_DIR=outputs/finetune/qwen2.5-7b-lora bash src/finetune/train_lora.sh
# =============================================================================
set -euo pipefail

# HuggingFace 下载源：国内服务器直连 huggingface.co 常报 Network is unreachable，
# 默认改走镜像 hf-mirror.com；如在能直连的环境，export HF_ENDPOINT=https://huggingface.co 覆盖。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 基座模型与适配器输出目录默认沿用 .env 的 LORA_BASE_MODEL / LORA_ADAPTER_DIR，
# 与 LocalLLMClassifier 推理端保持同一套配置；也可直接用 MODEL / OUTPUT_DIR 覆盖。
MODEL="${MODEL:-${LORA_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
DATASET="${DATASET:-genshin_sentiment}"
DATASET_DIR="${DATASET_DIR:-outputs/finetune}"
OUTPUT_DIR="${OUTPUT_DIR:-${LORA_ADAPTER_DIR:-outputs/finetune/qwen2.5-7b-lora}}"
# 默认单卡训练：限定可见卡，避免 LLaMA-Factory 误起多卡 NCCL。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# FlashAttention：默认 fa2（需 pip install flash-attn）；装不上就 FLASH_ATTN=auto 自动退回。
FLASH_ATTN="${FLASH_ATTN:-fa2}"

llamafactory-cli train \
    --stage sft \
    --do_train true \
    --model_name_or_path "${MODEL}" \
    --dataset "${DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    `# ---- 效率设置（4-bit 量化 / FA2 / 24GB 适配的批大小）---- ` \
    --quantization_bit 4 \
    --quantization_method bnb \
    --flash_attn "${FLASH_ATTN}" \
    --gradient_checkpointing true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --cutoff_len 1024 \
    --bf16 true \
    --optim paged_adamw_8bit \
    `# ---- 训练计划 ---- ` \
    --learning_rate 1.0e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 200 \
    --plot_loss true \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir true \
    --ddp_timeout 180000000

echo "✅ LoRA 适配器已保存到 ${OUTPUT_DIR}"
echo "在 dashboard / LocalLLMClassifier 中设置 adapter_path=${OUTPUT_DIR} 即可加载推理。"
