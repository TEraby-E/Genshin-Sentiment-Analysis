"""命令行入口：对评论数据跑「AI 清洗 → 归类 → 舆情总结」完整工作流。

这是实习职责3（调用主流 AI 模型 API 处理非结构化文本）的可运行演示与可复用工具。
因为会产生 API 调用费用，故独立于离线主流程（src/main.py），并用 --sample 控制成本。

用法：
    uv sync --extra llm
    # 在 .env 配好 DEEPSEEK_API_KEY 后：
    uv run python scripts/ai_analyze.py --sample 60
    uv run python scripts/ai_analyze.py --input data/raw_form.csv --text-col 反馈内容 --sample 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，评论/总结里含 emoji 会触发 UnicodeEncodeError，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config, llm_client, text_pipeline  # noqa: E402


def _load_comments(input_path: str | None, text_col: str) -> pd.DataFrame:
    if input_path:
        df = pd.read_csv(input_path)
    else:
        df = pd.read_csv(config.DATA_DIR / config.FILES["comments"])
    if text_col not in df.columns:
        raise SystemExit(f"输入数据缺少文本列 {text_col!r}，可用 --text-col 指定")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 文本工作流：清洗 → 归类 → 舆情总结")
    parser.add_argument("--input", help="输入 CSV（默认用 data/ 下的评论数据）")
    parser.add_argument("--text-col", default="Comment_Content", help="文本列名")
    parser.add_argument("--sample", type=int, default=40, help="只处理前 N 条以控制 API 成本")
    parser.add_argument(
        "--out", default=str(config.OUTPUT_DIR / "ai_analysis.csv"), help="结构化结果输出路径"
    )
    args = parser.parse_args()

    try:
        llm_client.get_api_key()
    except llm_client.LLMNotConfiguredError as e:
        raise SystemExit(f"[配置错误] {e}")

    comments = _load_comments(args.input, args.text_col)
    print(f"读取 {len(comments):,} 条，按 --sample={args.sample} 处理\n")

    # 1+2. 清洗 + AI 归类
    analyzed = text_pipeline.analyze_comments(
        comments, text_column=args.text_col, sample=args.sample
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["clean_text", "llm_sentiment", "llm_aspects", "llm_reason"]
    analyzed[cols].to_csv(args.out, index=False, encoding="utf-8-sig")

    print("【情感分布】")
    print(analyzed["llm_sentiment"].value_counts().to_string(), "\n")
    print("【方面 × 负面占比】")
    print(text_pipeline.aspect_sentiment_summary(analyzed).to_string(), "\n")

    # 3. 对负面评论做舆情总结
    neg = analyzed[analyzed["llm_sentiment"] == "负面"]["clean_text"].tolist()
    if neg:
        summary = text_pipeline.summarize_opinions(neg)
        print("【AI 舆情总结】")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        summary_path = Path(args.out).with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n总结已保存：{summary_path}")
    else:
        print("样本中无负面评论，跳过舆情总结。")

    print(f"\n结构化结果已保存：{args.out}")


if __name__ == "__main__":
    main()
