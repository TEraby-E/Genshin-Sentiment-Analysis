"""自包含的 QLoRA 微调脚本：只用 transformers + peft + bitsandbytes，不依赖 LLaMA-Factory。

动机：LLaMA-Factory 功能全但依赖树庞大且带原生库（torchaudio/torchvision/gradio），
在云 GPU 上极易因版本不匹配而崩（libtorchaudio_sox.so 之类）。本项目只做文本情感
分类的监督微调，用不到那套框架，故自带一个最小实现，依赖面小、版本稳定、完全可控。

读取 `build_finetune_dataset.py` / `dataset_formatter.py` 产出的 alpaca JSONL
（instruction/input/output），套用 Qwen chat 模板，对回答部分做监督（prompt 部分
label 置 -100），4-bit QLoRA + LoRA 训练后保存适配器。

用法（云 GPU 上）：
    uv run python -m src.finetune.train_lora
    uv run python -m src.finetune.train_lora --epochs 3 --batch-size 4
重依赖（torch/transformers/peft/bitsandbytes）延迟到运行时导入，不影响 CI。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .. import config

logger = logging.getLogger(__name__)

DEFAULT_DATA = config.OUTPUT_DIR / "finetune" / "genshin_sentiment.jsonl"
# Qwen2.5 的 LoRA 注入层（显式列出，比 "all-linear" 跨 peft 版本更稳）
_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def to_messages(record: dict[str, str]) -> list[dict[str, str]]:
    """把一条 alpaca 记录转成 chat 消息（不含 assistant 回答，供拼 prompt）。"""
    return [
        {"role": "system", "content": record["instruction"]},
        {"role": "user", "content": record["input"]},
    ]


def load_records(path: str | Path) -> list[dict[str, str]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _encode(record: dict[str, str], tokenizer: Any, cutoff_len: int) -> dict[str, list[int]]:
    """编码单条样本：仅对 assistant 回答计损失，prompt 部分 label 置 -100。"""
    prompt = tokenizer.apply_chat_template(
        to_messages(record), tokenize=False, add_generation_prompt=True
    )
    full = prompt + record["output"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:cutoff_len]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="自包含 QLoRA 微调（无 LLaMA-Factory）")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--base-model", default=config.LORA_BASE_MODEL)
    parser.add_argument("--output-dir", default=str(config.LORA_ADAPTER_DIR))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-4bit", action="store_true", help="关闭 4-bit 量化（显存足够时）")
    args = parser.parse_args()

    if not Path(args.data).exists():
        logger.error("训练数据不存在：%s（先跑 scripts/build_finetune_dataset.py）", args.data)
        return 1

    # 国内拉权重走镜像，避免 huggingface.co 不通
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # flash-attn 装了才用，否则用 torch 自带的 sdpa（无需额外依赖）
    attn = "flash_attention_2" if os.getenv("FLASH_ATTN") == "fa2" else "sdpa"

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=_LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    records = load_records(args.data)
    encoded = [_encode(r, tokenizer, args.cutoff_len) for r in records]
    logger.info("加载训练样本 %d 条", len(encoded))

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(encoded)

        def __getitem__(self, i: int) -> dict[str, list[int]]:
            return encoded[i]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=not args.no_4bit,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_Dataset(),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("✅ LoRA 适配器已保存到 %s", args.output_dir)
    print("训练完成。用 `uv run python scripts/eval_lora.py --predictor lora` 在留出集上评估。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
