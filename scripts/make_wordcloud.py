"""生成舆情词云图（透明背景 PNG）并保存到 Business-report-1/image/。

两种数据源（--source）：
- keyword（默认）：纯 jieba 词频，按数据集自带的粗粒度聚类分正/负面。离线、零成本，保留作对照。
- ai：调用主流 AI 模型 API（DeepSeek）逐条打标，按 LLM 情感分组，并用 AI 提炼的语义关键词
       加权放大词频，让词云突出"AI 认为重要"的议题词，而非单纯高频的口水词。

核心分词/加权/渲染逻辑复用 src/wordcloud_gen.py（与数据看板共用）。

用法：
    uv sync --extra report
    uv run python scripts/make_wordcloud.py                              # 纯词频（默认）
    uv run python scripts/make_wordcloud.py --source ai --sample 800     # AI 打标词云
    uv run python scripts/make_wordcloud.py --source ai --by-aspect      # 额外按方面出词云
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config, data_loader, wordcloud_gen

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

IMG_DIR = Path(__file__).resolve().parents[1] / "Business-report-1" / "image"


def save_wordcloud(weights, save_name: str, colormap: str, font_path: str) -> bool:
    """渲染词云并保存为 PNG；失败返回 False，不中断主流程。"""
    if not weights:
        logger.warning("词频为空，跳过 %s", save_name)
        return False
    try:
        img = wordcloud_gen.render_wordcloud(weights, colormap=colormap, font_path=font_path)
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        img.save(str(IMG_DIR / save_name))
    except (OSError, ValueError) as exc:
        logger.error("生成/保存词云 %s 失败: %s", save_name, exc)
        return False
    print(f"saved {save_name} ({len(weights)} 个候选词)")
    return True


def run_keyword_mode(comments, font_path: str) -> None:
    """纯 jieba 词频，按数据集自带聚类分正/负面（离线对照基线）。"""
    comments = comments.copy()
    comments["is_neg"] = comments["cluster_name"].isin(config.NEGATIVE_CLUSTERS)
    neg = comments.loc[comments["is_neg"], "Comment_Content"].tolist()
    pos = comments.loc[~comments["is_neg"], "Comment_Content"].tolist()
    print(f"[keyword] 负面 {len(neg):,} 条 | 非负面 {len(pos):,} 条")
    save_wordcloud(wordcloud_gen.tokenize(neg), "wordcloud_negative.png", "Reds", font_path)
    save_wordcloud(wordcloud_gen.tokenize(pos), "wordcloud_positive.png", "GnBu", font_path)


def run_ai_mode(comments, font_path: str, sample: int, by_aspect: bool) -> None:
    """调用 AI 打标分组 + 关键词加权出词云。需要 llm extra 与 DEEPSEEK_API_KEY。"""
    from src import llm_client, text_pipeline

    try:
        llm_client.get_api_key()
    except llm_client.LLMNotConfiguredError as e:
        logger.error("AI 模式需要配置 API key：%s", e)
        return

    print(f"[ai] 抽样 {sample} 条评论调用 DeepSeek 打标…")
    analyzed = text_pipeline.analyze_comments(comments, sample=sample)

    groups = {
        "negative": (analyzed[analyzed["llm_sentiment"] == "负面"], "Reds"),
        "positive": (analyzed[analyzed["llm_sentiment"] != "负面"], "GnBu"),
    }
    for name, (df, cmap) in groups.items():
        texts = df["clean_text"].tolist()
        if not texts:
            continue
        summary = text_pipeline.summarize_opinions(texts)
        weighted = wordcloud_gen.boost_with_ai_keywords(
            wordcloud_gen.tokenize(texts), summary.get("keywords", [])
        )
        print(f"[ai] {name}: {len(df)} 条，AI 关键词 {len(summary.get('keywords', []))} 个")
        save_wordcloud(weighted, f"wordcloud_ai_{name}.png", cmap, font_path)

    if by_aspect:
        exploded = analyzed.explode("llm_aspects")
        for aspect, df in exploded.groupby("llm_aspects"):
            if aspect == "其他" or len(df) < 3:
                continue
            safe = re.sub(r"[^\w]", "_", str(aspect))
            freq = wordcloud_gen.tokenize(df["clean_text"].tolist())
            save_wordcloud(freq, f"wordcloud_ai_aspect_{safe}.png", "viridis", font_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成舆情词云（纯词频 / AI 打标）")
    parser.add_argument("--source", choices=["keyword", "ai"], default="keyword")
    parser.add_argument("--sample", type=int, default=800, help="AI 模式抽样条数（控成本）")
    parser.add_argument("--by-aspect", action="store_true", help="AI 模式额外按方面分别出词云")
    args = parser.parse_args()

    try:
        font_path = wordcloud_gen.resolve_font_path()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return

    try:
        _, comments, _ = data_loader.load_all()
    except (FileNotFoundError, OSError) as exc:
        logger.error("加载评论数据失败，请确认 data/ 下的 CSV 是否存在: %s", exc)
        return

    if args.source == "ai":
        run_ai_mode(comments, font_path, args.sample, args.by_aspect)
    else:
        run_keyword_mode(comments, font_path)


if __name__ == "__main__":
    main()
