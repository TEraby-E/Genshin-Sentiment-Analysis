"""本地 LLM LoRA 微调子系统（进阶打标轨道）。

与 sentiment_train 的 TF-IDF 蒸馏并行：蒸馏轨道极轻、毫秒级、CPU 可跑；
本微调轨道把 LLM 标注的高置信数据进一步「灌」进一个轻量本地大模型
（如 Qwen2.5-7B），用 QLoRA 在消费级 GPU（如 RTX 4090, 24GB）上微调，得到一个
比 TF-IDF 更懂语义与黑话、又无需联网/不烧 API 的本地分类器。

- dataset_formatter：标注结果 -> alpaca JSONL（高置信筛选 + 训练/评估切分）；
- train_lora：自包含 QLoRA 微调（transformers + peft + bitsandbytes，不依赖 LLaMA-Factory）；
- evaluate：留出集评估 + 反讽错例分析，驱动增量迭代。
"""

from __future__ import annotations

from .analysis import (
    AspectSummaryRow,
    DEFAULT_DATASET_PATH,
    SentimentDatasetReport,
    build_sentiment_dataset_report,
    load_genshin_sentiment_jsonl,
)
from .dataset_formatter import (
    INSTRUCTION,
    format_dataset,
    load_labeled,
    split_records,
    to_alpaca_records,
    write_jsonl,
)
from .evaluate import (
    EvalReport,
    build_markdown_report,
    evaluate,
    find_error_cases,
    load_eval_set,
    write_predictions_csv,
)

__all__ = [
    "INSTRUCTION",
    "format_dataset",
    "load_labeled",
    "split_records",
    "to_alpaca_records",
    "write_jsonl",
    "AspectSummaryRow",
    "DEFAULT_DATASET_PATH",
    "SentimentDatasetReport",
    "build_sentiment_dataset_report",
    "load_genshin_sentiment_jsonl",
    "EvalReport",
    "evaluate",
    "build_markdown_report",
    "find_error_cases",
    "load_eval_set",
    "write_predictions_csv",
]
