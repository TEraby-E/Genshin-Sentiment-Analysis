"""最新 B 站评论 -> 现有模型路由 -> 舆情聚合报告。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .agents import RouterAgent


@dataclass
class LiveSentimentReport:
    """一批最新评论的舆情汇总。"""

    n_comments: int
    sentiment_counts: dict[str, int]
    sentiment_rates: dict[str, float]
    aspect_counts: dict[str, int]
    route_counts: dict[str, int]
    n_escalated: int
    n_verified: int
    negative_examples: list[str]
    overall: str

    def to_markdown(self) -> str:
        lines = [
            "# 最新 B 站舆情分析",
            "",
            f"- 评论数：{self.n_comments}",
            f"- 总体判断：{self.overall}",
            f"- 负面占比：{self.sentiment_rates.get('负面', 0):.1%}",
            f"- 校验通过：{self.n_verified}/{self.n_comments}",
            f"- 升档次数：{self.n_escalated}",
            "",
            "## 情感分布",
            "",
        ]
        for label in ("正面", "中性", "负面"):
            lines.append(f"- {label}: {self.sentiment_counts.get(label, 0)}")
        lines.extend(["", "## 高频方面", ""])
        for aspect, count in self.aspect_counts.items():
            lines.append(f"- {aspect}: {count}")
        lines.extend(["", "## 代表性负面评论", ""])
        for text in self.negative_examples:
            lines.append(f"- {text}")
        return "\n".join(lines)


def analyze_recent_comments(
    comments: pd.DataFrame,
    *,
    text_column: str = "Comment_Content",
    router: RouterAgent | None = None,
) -> tuple[pd.DataFrame, LiveSentimentReport]:
    """用现有 RouterAgent 打标最新评论，并输出逐条结果与聚合报告。"""
    if text_column not in comments.columns:
        raise KeyError(f"评论数据缺少文本列 {text_column!r}")

    df = comments.copy()
    texts = df[text_column].fillna("").astype(str).tolist()
    router = router or RouterAgent.from_environment()
    results = router.tag(texts)

    df["sentiment"] = [r.sentiment for r in results]
    df["aspects"] = ["、".join(r.aspects) or "其他" for r in results]
    df["track"] = [r.track for r in results]
    df["confidence"] = [r.confidence for r in results]
    df["verified"] = [r.verified for r in results]
    df["escalations"] = [r.escalations for r in results]
    df["reason"] = [r.reason for r in results]

    sentiment_counts = Counter(df["sentiment"])
    n = len(df)
    sentiment_rates = {
        label: (sentiment_counts.get(label, 0) / n if n else 0.0)
        for label in ("正面", "中性", "负面")
    }

    aspect_counter: Counter[str] = Counter()
    for result in results:
        for aspect in result.aspects or ["其他"]:
            aspect_counter[aspect] += 1

    neg = df[df["sentiment"] == "负面"].sort_values(
        ["confidence"], ascending=False, kind="stable"
    )
    negative_examples = neg[text_column].head(8).astype(str).tolist()
    overall = _overall_judgement(sentiment_rates, aspect_counter)

    report = LiveSentimentReport(
        n_comments=n,
        sentiment_counts=dict(sentiment_counts),
        sentiment_rates=sentiment_rates,
        aspect_counts=dict(aspect_counter.most_common(8)),
        route_counts=dict(router.last_stats.get("route_counts", {})),
        n_escalated=int(router.last_stats.get("n_escalated", 0)),
        n_verified=int(router.last_stats.get("n_verified", 0)),
        negative_examples=negative_examples,
        overall=overall,
    )
    return df, report


def _overall_judgement(rates: dict[str, float], aspects: Counter[str]) -> str:
    neg_rate = rates.get("负面", 0.0)
    top_aspect = aspects.most_common(1)[0][0] if aspects else "其他"
    if neg_rate >= 0.45:
        return f"近期负面情绪偏高，主要集中在「{top_aspect}」相关讨论。"
    if neg_rate >= 0.25:
        return f"近期舆情有一定负面压力，建议重点关注「{top_aspect}」。"
    return f"近期整体舆情相对平稳，讨论焦点主要是「{top_aspect}」。"


def report_to_dict(report: LiveSentimentReport) -> dict[str, Any]:
    """便于脚本写 JSON 或测试断言。"""
    return {
        "n_comments": report.n_comments,
        "sentiment_counts": report.sentiment_counts,
        "sentiment_rates": report.sentiment_rates,
        "aspect_counts": report.aspect_counts,
        "route_counts": report.route_counts,
        "n_escalated": report.n_escalated,
        "n_verified": report.n_verified,
        "negative_examples": report.negative_examples,
        "overall": report.overall,
    }
