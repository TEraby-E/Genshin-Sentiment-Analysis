"""`genshin_sentiment.jsonl` 的专业舆情分析口径。

这一层专门服务数据看板：把 alpaca JSONL 样本解析成表，再输出
情感结构、方面风险、负面贡献与复合主题占比等可视化指标。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config

DEFAULT_DATASET_PATH = config.OUTPUT_DIR / "finetune" / "genshin_sentiment.jsonl"


@dataclass
class AspectSummaryRow:
    aspect: str
    mentions: int
    coverage_rate: float
    negative_mentions: int
    negative_rate: float
    negative_share_of_dataset: float
    negative_contribution_share: float
    delta_vs_overall: float


@dataclass
class SentimentDatasetReport:
    n_comments: int
    sentiment_counts: dict[str, int]
    sentiment_rates: dict[str, float]
    negative_count: int
    negative_rate: float
    multi_aspect_rate: float
    aspect_rows: list[AspectSummaryRow]
    sentiment_mix: pd.DataFrame
    overall: str
    insights: list[str]

    def aspect_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self.aspect_rows])


def _normalize_aspects(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif pd.isna(value):
        raw = []
    elif isinstance(value, str):
        raw = [value]
    else:
        raw = [str(value)]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        label = str(item).strip()
        if not label or label not in config.LLM_ASPECT_LABELS or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out or ["其他"]


def _parse_output(raw: Any, *, line_no: int) -> dict[str, Any]:
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError as exc:  # pragma: no cover - 数据异常时给出明确定位
            raise ValueError(f"第 {line_no} 行 output 不是合法 JSON") from exc

    sentiment = str(payload.get("sentiment", "中性")).strip()
    if sentiment not in config.LLM_SENTIMENT_LABELS:
        sentiment = "中性"
    aspects = _normalize_aspects(payload.get("aspects"))
    return {"sentiment": sentiment, "aspects": aspects}


def load_genshin_sentiment_jsonl(source: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """读取 `genshin_sentiment.jsonl` 并解析出适合分析的结构化表。"""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件：{path}")

    frame = pd.read_json(path, lines=True)
    if frame.empty:
        raise ValueError(f"数据文件为空：{path}")
    if "output" not in frame.columns or "input" not in frame.columns:
        raise KeyError("JSONL 缺少 input/output 列")

    parsed = [_parse_output(raw, line_no=i + 1) for i, raw in enumerate(frame["output"])]
    parsed_df = pd.DataFrame(parsed)

    out = frame.copy()
    out = out.rename(columns={"input": "text"})
    out = out.drop(columns=["output"], errors="ignore")
    out = pd.concat([out, parsed_df], axis=1)
    out["aspect_count"] = out["aspects"].map(len)
    out["is_negative"] = out["sentiment"].eq("负面")
    return out


def build_sentiment_dataset_report(frame: pd.DataFrame) -> SentimentDatasetReport:
    """把结构化样本表汇总成看板可直接渲染的报告。"""
    if frame.empty:
        raise ValueError("样本表为空，无法生成分析报告")
    if "sentiment" not in frame.columns or "aspects" not in frame.columns:
        raise KeyError("样本表缺少 sentiment / aspects 列")

    n_comments = len(frame)
    sentiment_counts = Counter(frame["sentiment"])
    sentiment_rates = {
        label: (sentiment_counts.get(label, 0) / n_comments if n_comments else 0.0)
        for label in config.LLM_SENTIMENT_LABELS
    }
    negative_count = int(sentiment_counts.get("负面", 0))
    multi_aspect_rate = (
        float((frame["aspect_count"] > 1).mean()) if "aspect_count" in frame else 0.0
    )

    exploded = frame.explode("aspects").dropna(subset=["aspects"]).copy()
    if exploded.empty:
        empty_mix = pd.DataFrame(columns=["aspect", "sentiment", "count"])
        return SentimentDatasetReport(
            n_comments=n_comments,
            sentiment_counts=dict(sentiment_counts),
            sentiment_rates=sentiment_rates,
            negative_count=negative_count,
            negative_rate=sentiment_rates.get("负面", 0.0),
            multi_aspect_rate=multi_aspect_rate,
            aspect_rows=[],
            sentiment_mix=empty_mix,
            overall="样本表没有可分析的方面标签。",
            insights=[],
        )

    aspect_counts = (
        exploded.groupby("aspects").size().reindex(config.LLM_ASPECT_LABELS, fill_value=0)
    )
    neg_counts = (
        exploded.loc[exploded["sentiment"] == "负面"]
        .groupby("aspects")
        .size()
        .reindex(config.LLM_ASPECT_LABELS, fill_value=0)
    )
    sentiment_mix = (
        exploded.groupby(["aspects", "sentiment"])
        .size()
        .reset_index(name="count")
        .rename(columns={"aspects": "aspect"})
    )

    total_negative_aspect_mentions = int(neg_counts.sum()) or 1
    overall_negative_rate = sentiment_rates.get("负面", 0.0)

    rows: list[AspectSummaryRow] = []
    for aspect in config.LLM_ASPECT_LABELS:
        mentions = int(aspect_counts.get(aspect, 0))
        neg_mentions = int(neg_counts.get(aspect, 0))
        coverage_rate = mentions / n_comments if n_comments else 0.0
        negative_rate = neg_mentions / mentions if mentions else 0.0
        negative_share_of_dataset = neg_mentions / n_comments if n_comments else 0.0
        negative_contribution_share = neg_mentions / total_negative_aspect_mentions
        rows.append(
            AspectSummaryRow(
                aspect=aspect,
                mentions=mentions,
                coverage_rate=coverage_rate,
                negative_mentions=neg_mentions,
                negative_rate=negative_rate,
                negative_share_of_dataset=negative_share_of_dataset,
                negative_contribution_share=negative_contribution_share,
                delta_vs_overall=negative_rate - overall_negative_rate,
            )
        )

    rows = [row for row in rows if row.mentions > 0]
    rows.sort(key=lambda r: (r.negative_rate, r.negative_mentions, r.mentions), reverse=True)

    top_risk = rows[0]
    top_volume = max(rows, key=lambda r: (r.negative_share_of_dataset, r.mentions))
    top_coverage = max(rows, key=lambda r: (r.coverage_rate, r.mentions))

    overall = (
        f"整体负面率 {overall_negative_rate:.1%}；"
        f"复合主题评论占比 {multi_aspect_rate:.1%}；"
        f"风险最高的是「{top_risk.aspect}」。"
    )
    insights = [
        (
            f"整体情绪结构：正面 {sentiment_rates['正面']:.1%}，"
            f"中性 {sentiment_rates['中性']:.1%}，负面 {sentiment_rates['负面']:.1%}。"
        ),
        (
            f"风险强度最高的是「{top_risk.aspect}」，方面内负面率 "
            f"{top_risk.negative_rate:.1%}，较整体高 {top_risk.delta_vs_overall:+.1%}。"
        ),
        (
            f"负面体量最多的是「{top_volume.aspect}」，负面样本占全样本 "
            f"{top_volume.negative_share_of_dataset:.1%}。"
        ),
        f"覆盖面最广的是「{top_coverage.aspect}」，评论提及率 {top_coverage.coverage_rate:.1%}。",
    ]

    return SentimentDatasetReport(
        n_comments=n_comments,
        sentiment_counts=dict(sentiment_counts),
        sentiment_rates=sentiment_rates,
        negative_count=negative_count,
        negative_rate=overall_negative_rate,
        multi_aspect_rate=multi_aspect_rate,
        aspect_rows=rows,
        sentiment_mix=sentiment_mix,
        overall=overall,
        insights=insights,
    )
