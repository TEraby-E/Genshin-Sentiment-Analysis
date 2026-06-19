"""方面级情感分析：原数据集的评论聚类是粗粒度主题簇（如"社区争议"），
无法区分玩家具体在吐槽剧情、抽卡还是活动设计。这里加一层方面标签，
回答 README 中"可深化方向"提出的问题：细粒度方面分析能否提供聚类之外的增量信息。

不依赖外部 LLM API（项目默认离线可跑），用可解释的关键词规则做基线方面分类器，
预留 classify_with_llm() 接口，装了 `llm` extra 且配置了 API key 时可替换为更准确的分类器。
"""

from __future__ import annotations

import pandas as pd

ASPECTS: dict[str, list[str]] = {
    "剧情": ["剧情", "故事", "主线", "支线", "角色塑造", "世界观", "结局"],
    "抽卡": ["抽卡", "保底", "歪", "出货", "卡池", "概率", "氪金", "十连"],
    "活动": ["活动", "副本", "活动关卡", "限时", "周年", "纪行"],
    "数值与机制": ["数值", "强度", "削弱", "加强", "平衡性", "圣遗物", "词条", "机制"],
    "运营": ["运营", "客服", "公告", "补偿", "更新", "服务器", "卡顿", "bug", "闪退"],
    "社交与同人": ["二创", "同人", "cos", "社区", "梗", "表情包"],
}


def tag_aspects(text: str) -> list[str]:
    """返回命中的方面标签列表；一条评论可同时命中多个方面。"""
    if not isinstance(text, str):
        return []
    hits = [aspect for aspect, keywords in ASPECTS.items() if any(kw in text for kw in keywords)]
    return hits


def aspect_breakdown(comments: pd.DataFrame, text_column: str = "Comment_Content") -> pd.DataFrame:
    """统计各方面被提及的评论数及对应的负面占比，用于和粗粒度聚类对比。"""
    if text_column not in comments.columns:
        raise KeyError(
            f"评论数据缺少文本列 {text_column!r}，方面级分析需要原始评论文本"
        )

    df = comments.copy()
    df["aspects"] = df[text_column].map(tag_aspects)
    exploded = df.explode("aspects").dropna(subset=["aspects"])

    if "is_neg" not in exploded.columns:
        raise KeyError("需要先在评论数据上计算 is_neg 列（参见 analysis.sentiment_trend）")

    summary = exploded.groupby("aspects").agg(
        mentions=("is_neg", "size"), neg_rate=("is_neg", "mean")
    )
    summary["neg_rate"] = (summary["neg_rate"] * 100).round(1)
    return summary.sort_values("mentions", ascending=False)


def agreement_with_cluster(comments: pd.DataFrame, text_column: str = "Comment_Content") -> dict:
    """方面标签覆盖率：衡量关键词方面分类器能在粗粒度聚类之外提供多少增量信息。

    覆盖率低说明评论中缺乏方面相关关键词（口语化、表情符号为主），
    这恰恰是 README 中建议引入 LLM 做语义级（而非关键词级）方面识别的依据。
    """
    if text_column not in comments.columns:
        raise KeyError(f"评论数据缺少文本列 {text_column!r}")

    tagged = comments[text_column].map(tag_aspects)
    coverage = (tagged.map(len) > 0).mean()
    return {
        "coverage_rate": round(float(coverage) * 100, 1),
        "total_comments": len(comments),
        "tagged_comments": int((tagged.map(len) > 0).sum()),
    }


def classify_with_llm(texts: list[str]) -> list[list[str]]:
    """用主流 AI 模型 API 做语义级方面分类，替代 tag_aspects() 的关键词规则。

    需要 `uv sync --extra llm` 并在 .env 配置 DEEPSEEK_API_KEY（或任意 OpenAI 兼容服务）。
    返回与输入等长的方面标签列表；具体的清洗/批处理/重试逻辑见 text_pipeline。
    与 tag_aspects() 接口一致，可在 aspect_breakdown 等下游直接替换。
    """
    from . import text_pipeline

    cleaned = [text_pipeline.clean_text(t) for t in texts]
    preds = text_pipeline.classify_with_llm(cleaned)
    return [p["aspects"] for p in preds]
