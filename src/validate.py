"""数据契约校验：在加载阶段就近失败，而不是让脏数据流入分析层。"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "videos": {
        "Topic_Cluster", "Amount_View", "Amount_Like", "Amount_Favourite",
        "Publish_Date", "Video_Title", "TimeInSeconds", "Video_Length_Type",
    },
    "comments": {"Cluster", "Post_ID"},
    "posts": {"Post_ID", "Publish_Date"},
}


class SchemaError(ValueError):
    """数据缺少分析所需的列，或关键列全为空。"""


def check_columns(df: pd.DataFrame, kind: str) -> None:
    required = REQUIRED_COLUMNS.get(kind)
    if required is None:
        return
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"[{kind}] 缺少必需列: {sorted(missing)}")


def check_not_empty(df: pd.DataFrame, kind: str) -> None:
    if len(df) == 0:
        raise SchemaError(f"[{kind}] 数据为空，请检查文件是否正确下载")


def check_date_parse_rate(df: pd.DataFrame, column: str, kind: str, min_rate: float = 0.9) -> None:
    """日期解析失败率过高通常意味着源文件格式变化，需要提前预警而非悄悄丢数据。"""
    if column not in df.columns:
        return
    rate = df[column].notna().mean()
    if rate < min_rate:
        raise SchemaError(
            f"[{kind}] 字段 {column} 日期解析成功率仅 {rate:.1%}（<{min_rate:.0%}），"
            "请检查原始日期格式是否变更"
        )
