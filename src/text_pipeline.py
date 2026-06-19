"""非结构化文本的自动化「清洗 → 归类 → 分析」工作流（实习职责3落地）。

输入是采集到的 / 原始表单导出的原始评论文本（口语化、含表情/链接/回复标记），
输出是结构化的方面标签 + 情感极性 + 舆情总结，可直接接入舆情与内容分析看板。

分两层，可独立使用：
- 清洗（clean_*）：纯规则、零成本、可离线跑，去掉链接/@/回复标记/重复字符与空白；
- 归类 + 分析（classify_* / summarize_*）：调用主流 AI 模型 API 做语义级理解，
  解决关键词规则在口语化、反讽、表情符号场景下的覆盖盲区（见 aspect_sentiment 的覆盖率论证）。
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from . import config, llm_client

logger = logging.getLogger(__name__)

# ---- 1. 清洗：规则层 ----

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@[\w一-鿿\-]+")
# B 站评论常见的「回复 @某人 :」前缀
_REPLY_RE = re.compile(r"^回复\s*@?\S+?\s*[:：]")
# 连续 3 次以上重复的同一字符压缩为 2 次（如「哈哈哈哈哈」→「哈哈」）
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """单条文本规则清洗：去链接/@/回复前缀、压缩重复字符与空白。"""
    if not isinstance(text, str):
        return ""
    t = _REPLY_RE.sub("", text)
    t = _URL_RE.sub("", t)
    t = _MENTION_RE.sub("", t)
    t = _REPEAT_RE.sub(r"\1\1", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def clean_corpus(texts: list[str], *, min_len: int = 2, dedupe: bool = True) -> list[str]:
    """批量清洗：清洗 + 过滤过短/空文本 +（可选）去重，返回适合送入 LLM 的语料。"""
    cleaned = [clean_text(t) for t in texts]
    out: list[str] = []
    seen: set[str] = set()
    for t in cleaned:
        if len(t) < min_len:
            continue
        if dedupe and t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# ---- 2. 归类：调用 AI 模型 API 做方面 + 情感分类 ----

_CLASSIFY_SYSTEM = (
    "你是游戏社区舆情分析助手。针对每条玩家评论，判断它的情感极性，"
    "并归类到一个或多个内容方面。只输出 JSON，不要解释。\n"
    f"情感取值必须是其中之一：{config.LLM_SENTIMENT_LABELS}。\n"
    f"方面取值必须从下列集合中选取（可多选，无法归类则用「其他」）：{config.LLM_ASPECT_LABELS}。"
)


def _build_classify_prompt(batch: list[str]) -> str:
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(batch))
    return (
        "请逐条分析下列评论，返回 JSON 对象，键为 results，值为数组，"
        "每个元素形如 {\"id\": 序号, \"sentiment\": \"负面\", "
        "\"aspects\": [\"抽卡\"], \"reason\": \"一句话依据\"}。"
        "id 必须与输入序号一一对应，且覆盖全部评论。\n\n"
        f"评论：\n{numbered}"
    )


def _normalize_result(raw: dict, batch_len: int) -> list[dict]:
    """把 LLM 返回的 results 对齐成定长、字段合法的列表，缺失项兜底为中性/其他。"""
    by_id = {}
    for item in raw.get("results", []) or []:
        try:
            by_id[int(item.get("id"))] = item
        except (TypeError, ValueError):
            continue

    out: list[dict] = []
    for i in range(batch_len):
        item = by_id.get(i, {})
        sentiment = item.get("sentiment")
        if sentiment not in config.LLM_SENTIMENT_LABELS:
            sentiment = "中性"
        aspects = item.get("aspects") or []
        if not isinstance(aspects, list):
            aspects = [aspects]
        aspects = [a for a in aspects if a in config.LLM_ASPECT_LABELS] or ["其他"]
        out.append(
            {"sentiment": sentiment, "aspects": aspects, "reason": str(item.get("reason", ""))}
        )
    return out


def classify_batch(batch: list[str], *, client=None) -> list[dict]:
    """对一批评论做一次 API 调用，返回与输入等长的结构化结果。"""
    raw = llm_client.chat_json(_CLASSIFY_SYSTEM, _build_classify_prompt(batch), client=client)
    return _normalize_result(raw, len(batch))


def classify_with_llm(
    texts: list[str], *, client=None, batch_size: int | None = None
) -> list[dict]:
    """对任意长度的评论列表分批调用 API，返回每条的 {sentiment, aspects, reason}。

    单批失败时不让整个流程崩溃：记录告警并对该批兜底为中性/其他，保证产出可用。
    """
    batch_size = batch_size or config.LLM_BATCH_SIZE
    client = client or llm_client.get_client()
    results: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            results.extend(classify_batch(batch, client=client))
        except Exception as e:  # noqa: BLE001 - 单批失败兜底，不中断整体
            logger.error("第 %d 批分类失败，兜底为中性/其他：%s", start // batch_size, e)
            results.extend({"sentiment": "中性", "aspects": ["其他"], "reason": ""} for _ in batch)
    return results


def analyze_comments(
    comments: pd.DataFrame,
    *,
    text_column: str = "Comment_Content",
    sample: int | None = None,
    client=None,
) -> pd.DataFrame:
    """端到端工作流：清洗 → AI 归类 → 返回带 sentiment/aspects 的结构化 DataFrame。

    sample 用于控制成本：真实数据 40 万条，演示/迭代时只跑前 N 条即可。
    """
    if text_column not in comments.columns:
        raise KeyError(f"评论数据缺少文本列 {text_column!r}")

    df = comments.copy()
    if sample is not None:
        df = df.head(sample)

    df["clean_text"] = df[text_column].map(clean_text)
    valid = df[df["clean_text"].str.len() >= 2].copy()
    logger.info("清洗后有效评论 %d/%d 条，开始调用 API 分类", len(valid), len(df))

    preds = classify_with_llm(valid["clean_text"].tolist(), client=client)
    valid["llm_sentiment"] = [p["sentiment"] for p in preds]
    valid["llm_aspects"] = [p["aspects"] for p in preds]
    valid["llm_reason"] = [p["reason"] for p in preds]
    return valid


def aspect_sentiment_summary(analyzed: pd.DataFrame) -> pd.DataFrame:
    """把 analyze_comments 的结果聚合成「方面 × 负面占比」透视表，对接看板。"""
    exploded = analyzed.explode("llm_aspects")
    exploded["is_neg"] = exploded["llm_sentiment"] == "负面"
    summary = exploded.groupby("llm_aspects").agg(
        mentions=("is_neg", "size"), neg_rate=("is_neg", "mean")
    )
    summary["neg_rate"] = (summary["neg_rate"] * 100).round(1)
    return summary.sort_values("mentions", ascending=False)


# ---- 3. 分析：LLM 生成舆情总结（替代纯词频词云） ----

_SUMMARY_SYSTEM = (
    "你是游戏舆情分析师。基于给定的玩家评论样本，提炼负面舆情的核心议题，"
    "并给出可执行的运营建议。只输出 JSON，不要寒暄。"
)


_HIT_SUMMARY_SYSTEM = (
    "你是游戏内容运营分析师。给定一批已成为爆款的视频标题，"
    "提炼它们的共性选题与可复制的内容套路，输出对内容团队的选题指导。只输出 JSON。"
)


def summarize_hits(titles: list[str], *, client=None) -> dict:
    """内容爆点总结：对爆款标题做语义归纳，产出可复制的选题套路与建议。

    对接看板「内容爆点总结与预测」模块：预测回答"会不会爆"，总结回答"为什么爆、怎么复制"。
    """
    sample = clean_corpus(titles, dedupe=True)[:80]
    joined = "\n".join(f"- {t}" for t in sample)
    user = (
        "请基于下列爆款视频标题输出 JSON，字段为：\n"
        "patterns: 数组，每项 {theme: 选题名, summary: 说明, examples: 最多3个代表标题}；\n"
        "keywords: 数组，10个高频爆款关键词；\n"
        "advice: 数组，2-4条给内容团队的可复制选题建议。\n\n"
        f"爆款标题：\n{joined}"
    )
    return llm_client.chat_json(_HIT_SUMMARY_SYSTEM, user, client=client)


def summarize_opinions(texts: list[str], *, max_quotes: int = 3, client=None) -> dict:
    """对一批（通常是负面）评论做语义级舆情总结：议题、代表性原声、关键词、运营建议。

    相比纯词频词云，这一步能合并同义表达、过滤噪声、直接产出业务可读的结论。
    """
    sample = clean_corpus(texts)[:120]  # 控制 token：取清洗去重后的前 120 条
    joined = "\n".join(f"- {t}" for t in sample)
    user = (
        "请基于下列评论样本输出 JSON，字段为：\n"
        "themes: 数组，每项 {topic: 议题名, share: 该议题大致占比0-1, "
        "summary: 一句话概括, quotes: 最多"
        f"{max_quotes}条代表性原声}}；\n"
        "keywords: 数组，10个高频/高代表性关键词（供词云使用）；\n"
        "overall: 一句话总体判断；\n"
        "actions: 数组，2-4条给运营/内容团队的可执行建议。\n\n"
        f"评论样本：\n{joined}"
    )
    return llm_client.chat_json(_SUMMARY_SYSTEM, user, client=client)
