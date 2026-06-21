"""本地 LLM LoRA 微调子系统（进阶打标轨道）。

与 sentiment_train 的 TF-IDF 蒸馏并行：蒸馏轨道极轻、毫秒级、CPU 可跑；
本微调轨道把 LLM 标注的高置信数据进一步「灌」进一个轻量本地大模型
（如 Qwen2.5-7B），用 QLoRA 在消费级 GPU（如 RTX 4090, 24GB）上微调，得到一个
比 TF-IDF 更懂语义与黑话、又无需联网/不烧 API 的本地分类器。

- dataset_formatter：outputs/ai_analysis.csv -> LLaMA-Factory 所需的严格 JSONL；
- train_lora.sh：LLaMA-Factory 的 QLoRA 训练模板（4bit / 梯度检查点 / FA2 / 微批=1~2）。
"""

from __future__ import annotations

from .dataset_formatter import (
    INSTRUCTION,
    format_dataset,
    load_labeled,
    split_records,
    to_alpaca_records,
    write_jsonl,
)
from .evaluate import EvalReport, evaluate, find_error_cases, load_eval_set

__all__ = [
    "INSTRUCTION",
    "format_dataset",
    "load_labeled",
    "split_records",
    "to_alpaca_records",
    "write_jsonl",
    "EvalReport",
    "evaluate",
    "find_error_cases",
    "load_eval_set",
]
