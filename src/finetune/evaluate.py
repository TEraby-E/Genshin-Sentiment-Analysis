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
    predictions: list[str] = field(default_factory=list)

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
        predictions=pred,
    )


# ---- 详细报告：把一次评估渲染成可直接展示的 Markdown + 逐条 CSV ----


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def per_class_table(report: EvalReport) -> list[dict[str, Any]]:
    """从 sklearn 的 classification_report 取每类精确率/召回率/F1/支持数。"""
    rows: list[dict[str, Any]] = []
    for label in report.labels:
        d = report.per_label.get(label, {})
        rows.append(
            {
                "label": label,
                "precision": float(d.get("precision", 0.0)),
                "recall": float(d.get("recall", 0.0)),
                "f1": float(d.get("f1-score", 0.0)),
                "support": int(d.get("support", 0)),
            }
        )
    return rows


def confusion_breakdown(report: EvalReport) -> list[dict[str, Any]]:
    """把混淆矩阵里所有非对角的错误格子按数量从多到少列出（最大误差模式一目了然）。"""
    cells: list[dict[str, Any]] = []
    for i, gold_label in enumerate(report.labels):
        for j, pred_label in enumerate(report.labels):
            if i != j and report.confusion[i][j] > 0:
                cells.append(
                    {"gold": gold_label, "pred": pred_label, "count": report.confusion[i][j]}
                )
    return sorted(cells, key=lambda c: c["count"], reverse=True)


def build_markdown_report(
    report: EvalReport,
    *,
    predictor_name: str,
    eval_set: str,
    baselines: dict[str, EvalReport] | None = None,
    max_error_samples: int = 30,
) -> str:
    """把评估结果渲染成一份详细的 Markdown 报告，完整体现训练成果。"""
    import datetime as _dt

    labels = report.labels
    lines: list[str] = []
    a = lines.append

    a(f"# LoRA 微调评估报告 · `{predictor_name}`")
    a("")
    a(f"- 生成时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- 评估集：`{eval_set}`（{report.n} 条留出样本）")
    a(f"- 预测器：`{predictor_name}`")
    a("")

    a("## 1. 总体指标")
    a("")
    a("| 指标 | 数值 |")
    a("| --- | --- |")
    a(f"| 准确率 Accuracy | **{_fmt_pct(report.accuracy)}** |")
    a(f"| 宏平均 F1 Macro-F1 | **{_fmt_pct(report.macro_f1)}** |")
    a(f"| 样本数 | {report.n} |")
    irony_err = sum(1 for e in report.errors if e["irony"])
    a(f"| 错误数 | {len(report.errors)}（其中疑似反讽 {irony_err}）|")
    a("")
    gap = abs(report.accuracy - report.macro_f1)
    if gap <= 0.03:
        a(
            f"> Accuracy 与 Macro-F1 仅相差 {_fmt_pct(gap)}，说明模型在三类上表现**均衡**，"
            "没有被多数类撑高整体准确率。"
        )
    else:
        a(
            f"> Accuracy 与 Macro-F1 相差 {_fmt_pct(gap)}，提示存在**类别不均衡**，"
            "需关注下表中召回率偏低的类别。"
        )
    a("")

    a("## 2. 分类别指标")
    a("")
    a("| 类别 | 精确率 Precision | 召回率 Recall | F1 | 支持数 Support |")
    a("| --- | --- | --- | --- | --- |")
    rows = per_class_table(report)
    for r in rows:
        a(
            f"| {r['label']} | {_fmt_pct(r['precision'])} | {_fmt_pct(r['recall'])} | "
            f"{_fmt_pct(r['f1'])} | {r['support']} |"
        )
    a("")
    worst = min(rows, key=lambda r: r["recall"]) if rows else None
    best = max(rows, key=lambda r: r["f1"]) if rows else None
    if worst and best:
        a(
            f"> 表现最好的类别是 **{best['label']}**（F1 {_fmt_pct(best['f1'])}）；"
            f"召回率最低的是 **{worst['label']}**（Recall {_fmt_pct(worst['recall'])}），"
            "是后续针对性补样的首选方向。"
        )
    a("")

    a("## 3. 混淆矩阵")
    a("")
    header = "| 金标＼预测 | " + " | ".join(labels) + " | 合计 |"
    a(header)
    a("|" + " --- |" * (len(labels) + 2))
    for i, gl in enumerate(labels):
        row = report.confusion[i]
        cells = []
        for j, v in enumerate(row):
            cells.append(f"**{v}**" if i == j else str(v))
        a(f"| {gl} | " + " | ".join(cells) + f" | {sum(row)} |")
    a("")
    a("> 对角线（加粗）是判对的数量，其余为误判。")
    a("")

    a("## 4. 主要误差模式")
    a("")
    breakdown = confusion_breakdown(report)
    if breakdown:
        a("| 金标 → 预测 | 数量 | 占总错误 |")
        a("| --- | --- | --- |")
        total_err = sum(c["count"] for c in breakdown)
        for c in breakdown:
            share = c["count"] / total_err if total_err else 0
            a(f"| {c['gold']} → {c['pred']} | {c['count']} | {_fmt_pct(share)} |")
        a("")
        top = breakdown[0]
        a(
            f"> 最大的单一误差模式是 **{top['gold']} 被判成 {top['pred']}**（{top['count']} 条），"
            "可优先围绕这条边界补充训练样本或在路由中升档复核。"
        )
    else:
        a("无误判。")
    a("")

    if baselines:
        a("## 5. 与基线对比")
        a("")
        a("| 模型 | 准确率 | 宏 F1 | 负面召回 |")
        a("| --- | --- | --- | --- |")

        def _neg_recall(rep: EvalReport) -> str:
            d = rep.per_label.get("负面", {})
            return _fmt_pct(float(d.get("recall", 0.0)))

        a(
            f"| **{predictor_name}（本次）** | {_fmt_pct(report.accuracy)} | "
            f"{_fmt_pct(report.macro_f1)} | {_neg_recall(report)} |"
        )
        for name, rep in baselines.items():
            a(
                f"| {name} | {_fmt_pct(rep.accuracy)} | {_fmt_pct(rep.macro_f1)} | "
                f"{_neg_recall(rep)} |"
            )
        a("")
        a(
            "> 负面召回单列出来，是因为它最能体现微调价值：轻量蒸馏模型常几乎不预测负面，"
            "微调后的大模型应在该列有明显提升。"
        )
        a("")

    sec = "6" if baselines else "5"
    a(f"## {sec}. 错例样本（前 {max_error_samples} 条）")
    a("")
    a("| 金标 | 预测 | 反讽? | 文本 |")
    a("| --- | --- | --- | --- |")
    for e in report.errors[:max_error_samples]:
        text = str(e["text"]).replace("|", "／").replace("\n", " ")[:60]
        a(f"| {e['gold']} | {e['pred']} | {'是' if e['irony'] else ''} | {text} |")
    a("")
    if len(report.errors) > max_error_samples:
        a(f"> 还有 {len(report.errors) - max_error_samples} 条错例见同目录 CSV。")
    a("")

    return "\n".join(lines)


def write_predictions_csv(
    path: str | Path, texts: list[str], gold: list[str], report: EvalReport
) -> None:
    """把逐条预测（含对错标记与反讽标记）写成 CSV，便于细看与复盘。"""
    import csv

    pred = report.predictions or [""] * len(texts)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["文本", "金标", "预测", "是否正确", "疑似反讽"])
        for text, g, p in zip(texts, gold, pred):
            irony = any(m in str(text) for m in _IRONY_MARKERS)
            w.writerow([text, g, p, "✓" if g == p else "✗", "是" if irony else ""])
