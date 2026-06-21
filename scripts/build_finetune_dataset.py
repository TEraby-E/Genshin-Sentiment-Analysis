"""构建微调「黄金样本集」（第 1 步）：分层抽样 → DeepSeek 标注 → 高置信筛选 → 训练/评估切分。

做法对应需求：
- 从 40 万评论里按聚类（运营/剧情/抽卡/角色 等主题簇）分层抽样，保证类别覆盖；
- 调 DeepSeek 逐条打标（清洗 + 情感 + 方面 + 依据），复用既有 text_pipeline；
- 用置信度规则（合法标签 + 非空判断依据 + 最小长度）筛出高置信样本；
- 切出 15% 留出评估集（供 scripts/eval_lora.py 做第 3 步验证）。

用法：
    uv run python scripts/build_finetune_dataset.py --sample 300
    uv run python scripts/build_finetune_dataset.py --sample 8000   # 正式规模（耗 API 与时间）
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import text_pipeline
from src.finetune import dataset_formatter as dfmt


def stratified_sample(
    comments: pd.DataFrame,
    n: int,
    *,
    cluster_col: str = "Cluster",
    text_col: str = "Comment_Content",
    seed: int = 42,
) -> pd.DataFrame:
    """按聚类分层抽样 n 条评论，保证各主题簇都有覆盖，避免单一类别主导。"""
    df = comments.dropna(subset=[text_col]).copy()
    df = df[df[text_col].astype(str).str.len() >= 6]
    if cluster_col in df.columns and df[cluster_col].nunique() > 1:
        groups = list(df.groupby(cluster_col))
        per = max(1, n // len(groups))
        parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in groups]
        out = pd.concat(parts)
        if len(out) > n:
            out = out.sample(n, random_state=seed)
    else:
        out = df.sample(min(n, len(df)), random_state=seed)
    return out.reset_index(drop=True)


def main() -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构建微调黄金样本集（分层抽样+标注+筛选+切分）")
    parser.add_argument("--sample", type=int, default=300, help="分层抽样的评论条数")
    parser.add_argument("--eval-ratio", type=float, default=0.15, help="留出评估集比例")
    parser.add_argument("--min-reason-len", type=int, default=4)
    parser.add_argument("--out-dir", default=str(dfmt.DEFAULT_OUT_DIR))
    args = parser.parse_args()

    try:
        from src import data_loader

        _, comments, _ = data_loader.load_all()
    except Exception as e:  # noqa: BLE001 - 无真实数据时明确报错
        print(f"无法加载评论数据（请确认 data/ 下已放置 CSV）：{e}", file=sys.stderr)
        return 1

    sampled = stratified_sample(comments, args.sample)
    print(f"分层抽样 {len(sampled)} 条，开始调用 DeepSeek 标注…")
    analyzed = text_pipeline.analyze_comments(sampled)  # 清洗 + LLM 打标

    records = dfmt.to_alpaca_records(analyzed, min_reason_len=args.min_reason_len)
    if not records:
        print("筛选后无高置信样本，增大 --sample 或放宽过滤。", file=sys.stderr)
        return 1

    train, holdout = dfmt.split_records(records, eval_ratio=args.eval_ratio)
    from pathlib import Path

    out_dir = Path(args.out_dir)
    train_path = dfmt.write_jsonl(train, out_dir / "train.jsonl")
    eval_path = dfmt.write_jsonl(holdout, out_dir / "eval.jsonl")
    info_path = dfmt.write_dataset_info(train_path.name, out_dir / "dataset_info.json")
    # 落一份完整标注（与 ai_analysis.csv 同构），便于复盘与复跑
    analyzed[["clean_text", "llm_sentiment", "llm_aspects", "llm_reason"]].to_csv(
        out_dir / "labeled.csv", index=False
    )

    print(f"高置信样本 {len(records)} 条 → 训练 {len(train)} / 评估 {len(holdout)}")
    print(f"  训练集     : {train_path}")
    print(f"  评估集     : {eval_path}（供 scripts/eval_lora.py 做第 3 步验证）")
    print(f"  数据集注册 : {info_path}")
    print("方面分布：", analyzed.explode("llm_aspects")["llm_aspects"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
