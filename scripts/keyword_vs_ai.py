"""对比演示：关键词基线漏标的评论，AI 语义打标能不能标出来。

挑出 tag_aspects() 命中 0 个方面的评论，喂给 DeepSeek 做语义级打标，
输出「关键词漏标 → AI 标出」前后对比表，量化 AI 相对关键词规则的增量覆盖。

用法：
    uv run python scripts/keyword_vs_ai.py --sample 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import aspect_sentiment, config, llm_client, text_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="关键词漏标 vs AI 语义打标 对比")
    parser.add_argument("--sample", type=int, default=30, help="抽取多少条被漏标的评论送 AI")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        llm_client.get_api_key()
    except llm_client.LLMNotConfiguredError as e:
        raise SystemExit(f"[配置错误] {e}")

    c = pd.read_csv(config.DATA_DIR / config.FILES["comments"])
    text = c["Comment_Content"].dropna().astype(str)

    # 关键词漏标 = tag_aspects 命中 0 个方面；取长度适中、可读的样本
    missed = text[text.map(lambda t: len(aspect_sentiment.tag_aspects(t)) == 0)]
    missed = missed[missed.str.len().between(6, 40)]
    sample = missed.sample(n=min(args.sample, len(missed)), random_state=args.seed).tolist()
    print(f"全站漏标评论中抽样 {len(sample)} 条送 DeepSeek 打标…\n")

    cleaned = [text_pipeline.clean_text(t) for t in sample]
    preds = text_pipeline.classify_with_llm(cleaned)

    rows = []
    for orig, p in zip(sample, preds):
        rows.append(
            {
                "评论": orig[:30],
                "关键词": "—",  # 定义上这批就是关键词漏标的
                "AI情感": p["sentiment"],
                "AI方面": "、".join(p["aspects"]),
            }
        )
    df = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))

    # 增量覆盖：AI 把多少条从"无标签"救回成有效方面（非「其他」）
    rescued = sum(1 for p in preds if p["aspects"] != ["其他"])
    print(
        f"\n关键词对这批的覆盖率 = 0%（定义如此）；"
        f"AI 标出有效方面 {rescued}/{len(preds)} 条 = {rescued / len(preds) * 100:.0f}%"
    )
    neg = sum(1 for p in preds if p["sentiment"] == "负面")
    print(f"其中 AI 识别为负面 {neg} 条 —— 这些是关键词预警完全漏掉的潜在舆情。")


if __name__ == "__main__":
    main()
