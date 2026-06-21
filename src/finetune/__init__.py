"""本地 LLM LoRA 微调子系统（eGPU 优化的进阶打标轨道）。

与 sentiment_train 的 TF-IDF 蒸馏并行：蒸馏轨道极轻、毫秒级、CPU 可跑；
本微调轨道把 LLM 标注的高置信数据进一步「灌」进一个轻量本地大模型
（如 Qwen2.5-7B），用 QLoRA 在 eGPU 上低显存微调，得到一个比 TF-IDF 更懂
语义与黑话、又无需联网/不烧 API 的本地分类器。

- dataset_formatter：outputs/ai_analysis.csv -> LLaMA-Factory 所需的严格 JSONL；
- train_lora.sh：LLaMA-Factory 的 QLoRA 训练模板（4bit / 梯度检查点 / FA2 / 微批=1~2）。
"""

from __future__ import annotations

from .dataset_formatter import (
    INSTRUCTION,
    format_dataset,
    load_labeled,
    to_alpaca_records,
    write_jsonl,
)

__all__ = [
    "INSTRUCTION",
    "format_dataset",
    "load_labeled",
    "to_alpaca_records",
    "write_jsonl",
]
