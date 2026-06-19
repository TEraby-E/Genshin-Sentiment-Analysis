"""命令行入口：舆情打标训练（LLM 标注小样本 → 蒸馏轻量本地分类器）。

流程：抽样评论 → DeepSeek 打情感标签（老师）→ 训练字符级 TF-IDF + 逻辑回归（学生）
     → 留出集评估 → 保存模型。之后全量推理走本地模型，免 API。

用法：
    uv sync --extra llm
    uv run python scripts/train_sentiment.py --sample 600
    uv run python scripts/train_sentiment.py --sample 800 --out outputs/sentiment_clf.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from src import config, llm_client, sentiment_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="舆情打标训练（LLM 蒸馏轻量分类器）")
    parser.add_argument("--sample", type=int, default=600, help="LLM 标注的样本量（老师标注成本）")
    parser.add_argument("--out", default=str(sentiment_train.DEFAULT_MODEL_PATH))
    parser.add_argument("--input", help="输入 CSV（默认用 data/ 下评论数据）")
    parser.add_argument("--text-col", default="Comment_Content")
    args = parser.parse_args()

    try:
        llm_client.get_api_key()
    except llm_client.LLMNotConfiguredError as e:
        raise SystemExit(f"[配置错误] {e}")

    if args.input:
        comments = pd.read_csv(args.input)
    else:
        comments = pd.read_csv(config.DATA_DIR / config.FILES["comments"])

    print(f"用 DeepSeek 给 {args.sample} 条评论打情感标签（老师标注）…")
    train_df = sentiment_train.build_training_data(
        comments, text_column=args.text_col, sample=args.sample
    )
    train_df = train_df[train_df["text"].str.len() >= 2]
    print(f"有效标注 {len(train_df)} 条，标签分布：{train_df['label'].value_counts().to_dict()}\n")

    print("训练学生模型（字符级 TF-IDF + 逻辑回归）…")
    result = sentiment_train.train_classifier(
        train_df["text"].tolist(), train_df["label"].tolist()
    )
    m = result["metrics"]
    print("【留出集评估】")
    print(f"  训练/测试 = {m['n_train']}/{m['n_test']}")
    print(f"  准确率 = {m['accuracy']}　宏 F1 = {m['macro_f1']}")
    print(f"  与老师(LLM)标签一致率 = {m['agreement_with_teacher']}")

    path = sentiment_train.save_model(result["model"], args.out)
    print(f"\n学生模型已保存：{path}")
    print("之后可用 sentiment_train.load_model() + predict() 离线、免 API 地全量打标。")


if __name__ == "__main__":
    main()
