"""把 LLM 标注结果（outputs/ai_analysis.csv）转成 LLaMA-Factory 的 alpaca JSONL。

只保留「高置信」样本作为监督信号：
- 情感/方面取值合法（落在 config 的取值域内）；
- reason 非空且达到最小长度（老师给出了明确依据，置信度更高）；
- 清洗后文本不过短。

输出两份产物到 outputs/finetune/：
- genshin_sentiment.jsonl：每行一条 {instruction, input, output} 训练样本；
- dataset_info.json：可直接拷进 LLaMA-Factory 的 data/dataset_info.json 完成注册。
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = config.OUTPUT_DIR / "ai_analysis.csv"
DEFAULT_OUT_DIR = config.OUTPUT_DIR / "finetune"
DATASET_NAME = "genshin_sentiment"

INSTRUCTION = (
    "你是原神游戏社区舆情分析助手。判断这条玩家评论的情感极性，并归类到一个或多个内容方面。"
    f"情感取值必须是其中之一：{config.LLM_SENTIMENT_LABELS}。"
    f"方面取值必须从下列集合中选取（可多选）：{config.LLM_ASPECT_LABELS}。"
    '只输出 JSON，形如 {"sentiment": "负面", "aspects": ["抽卡"]}。'
)


def _parse_aspects(value: object) -> list[str]:
    """ai_analysis.csv 里 aspects 以字符串形式存储（如 "['抽卡']"），安全解析回列表。"""
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str) and value.strip().startswith("["):
        try:
            raw = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raw = [value]
    elif isinstance(value, str) and value.strip():
        raw = [value]
    else:
        raw = []
    return [a for a in raw if a in config.LLM_ASPECT_LABELS]


def load_labeled(path: str | Path = DEFAULT_SOURCE) -> pd.DataFrame:
    """读取 LLM 标注 CSV，标准化列名。"""
    df = pd.read_csv(path)
    required = {"clean_text", "llm_sentiment"}
    if not required.issubset(df.columns):
        raise KeyError(f"标注文件缺少必要列 {required - set(df.columns)}")
    return df


def to_alpaca_records(
    df: pd.DataFrame, *, min_reason_len: int = 4, min_text_len: int = 2
) -> list[dict[str, str]]:
    """过滤高置信样本并转成 alpaca 记录（instruction/input/output）。"""
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        text = str(row.get("clean_text", "")).strip()
        sentiment = row.get("llm_sentiment")
        if len(text) < min_text_len or sentiment not in config.LLM_SENTIMENT_LABELS:
            continue
        reason = str(row.get("llm_reason", "") or "")
        if len(reason.strip()) < min_reason_len:
            continue
        aspects = _parse_aspects(row.get("llm_aspects")) or ["其他"]
        output = json.dumps({"sentiment": sentiment, "aspects": aspects}, ensure_ascii=False)
        records.append({"instruction": INSTRUCTION, "input": text, "output": output})
    logger.info("高置信样本 %d/%d 条", len(records), len(df))
    return records


def write_jsonl(records: list[dict[str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def write_dataset_info(jsonl_name: str, path: str | Path) -> Path:
    """写出可直接注册进 LLaMA-Factory 的 dataset_info.json 片段。"""
    info = {
        DATASET_NAME: {
            "file_name": jsonl_name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_dataset(
    source: str | Path = DEFAULT_SOURCE,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    min_reason_len: int = 4,
) -> dict[str, object]:
    """端到端：读标注 -> 过滤 -> 写 JSONL + dataset_info.json。"""
    out_dir = Path(out_dir)
    df = load_labeled(source)
    records = to_alpaca_records(df, min_reason_len=min_reason_len)
    if not records:
        raise RuntimeError("没有满足高置信条件的样本，无法构建微调集（放宽过滤或扩充标注）。")
    jsonl_path = write_jsonl(records, out_dir / f"{DATASET_NAME}.jsonl")
    info_path = write_dataset_info(jsonl_path.name, out_dir / "dataset_info.json")
    return {"n_records": len(records), "jsonl": jsonl_path, "dataset_info": info_path}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构建 LLaMA-Factory 微调数据集")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-reason-len", type=int, default=4)
    args = parser.parse_args()
    res = format_dataset(args.source, args.out_dir, min_reason_len=args.min_reason_len)
    print(f"已生成 {res['n_records']} 条样本 -> {res['jsonl']}")
    print(f"dataset_info.json -> {res['dataset_info']}")


if __name__ == "__main__":
    main()
