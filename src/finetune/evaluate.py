"""微调模型的留出评估与错例分析（第 3 步：评估-迭代闭环）。

给定任意「文本 -> 情感」的预测器（LoRA 本地模型、蒸馏模型、甚至关键词基线均可），
在留出评估集上算出准确率 / 宏 F1 / 各类指标 / 混淆矩阵，并把错例单独捞出来、
标注其中疑似反讽的样本，用于驱动「针对性补样 → 增量微调」的下一轮迭代。

预测器是注入式的纯函数接口，因此本模块无需 GPU/大模型即可被单测覆盖。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config

# 预测器：一批文本 -> 一批情感标签（与 config.LLM_SENTIMENT_LABELS 取值一致）
Predictor = Callable[[list[str]], list[str]]

# 反讽 / 阴阳标记：错例里命中这些的，多半是字面与真实情感相反的难样本，优先补样。
_IRONY_MARKERS = (
    "呵呵", "好家伙", "笑死", "行吧", "懂的都懂", "绝了", "阴阳", "典",
    "乐", "牛", "666", "格局打开", "谢谢你米哈游",
)


@dataclass
class EvalReport:
    """一次留出评估的完整结果。"""

    n: int
    accuracy: float
    macro_f1: float
    per_label: dict[str, Any]
    confusion: list[list[int]]
    labels: list[str]
    errors: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        irony_n = sum(1 for e in self.errors if e["irony"])
        return (
            f"n={self.n}  accuracy={self.accuracy:.3f}  macro_f1={self.macro_f1:.3f}  "
            f"errors={len(self.errors)}（其中疑似反讽 {irony_n}）"
        )


def load_eval_set(path: str | Path) -> tuple[list[str], list[str]]:
    """读取 alpaca JSONL 评估集，返回 (文本, 金标情感)。"""
    texts: list[str] = []
    gold: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        texts.append(str(rec.get("input", "")))
        gold.append(str(json.loads(rec["output"]).get("sentiment", "中性")))
    return texts, gold


def find_error_cases(
    texts: Sequence[str], gold: Sequence[str], pred: Sequence[str]
) -> list[dict[str, Any]]:
    """挑出预测错误的样本，并标注其中疑似反讽的（驱动针对性补样）。"""
    errors: list[dict[str, Any]] = []
    for text, g, p in zip(texts, gold, pred):
        if g != p:
            errors.append(
                {
                    "text": text,
                    "gold": g,
                    "pred": p,
                    "irony": any(m in text for m in _IRONY_MARKERS),
                }
            )
    return errors


def evaluate(predictor: Predictor, texts: list[str], gold: list[str]) -> EvalReport:
    """用预测器在留出集上评估，返回含指标与错例的报告。"""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    pred = [str(p) for p in predictor(texts)]
    labels = list(config.LLM_SENTIMENT_LABELS)
    return EvalReport(
        n=len(texts),
        accuracy=round(float(accuracy_score(gold, pred)), 4),
        macro_f1=round(
            float(f1_score(gold, pred, average="macro", labels=labels, zero_division=0)), 4
        ),
        per_label=classification_report(
            gold, pred, labels=labels, output_dict=True, zero_division=0
        ),
        confusion=confusion_matrix(gold, pred, labels=labels).tolist(),
        labels=labels,
        errors=find_error_cases(texts, gold, pred),
    )
