from __future__ import annotations

import pandas as pd
import pytest

from src import validate


def test_check_columns_passes_when_present(videos_df):
    validate.check_columns(videos_df, "videos")  # should not raise


def test_check_columns_raises_when_missing():
    df = pd.DataFrame({"Amount_View": [1, 2]})
    with pytest.raises(validate.SchemaError, match="缺少必需列"):
        validate.check_columns(df, "videos")


def test_check_not_empty_raises_on_empty_df():
    with pytest.raises(validate.SchemaError):
        validate.check_not_empty(pd.DataFrame(), "videos")


def test_check_date_parse_rate_raises_when_mostly_unparsed():
    df = pd.DataFrame({"Publish_Date": pd.to_datetime([None, None, None, "2024-01-01"])})
    with pytest.raises(validate.SchemaError, match="日期解析成功率"):
        validate.check_date_parse_rate(df, "Publish_Date", "videos", min_rate=0.9)


def test_check_date_parse_rate_passes_when_mostly_parsed():
    df = pd.DataFrame({"Publish_Date": pd.to_datetime(["2024-01-01"] * 9 + [None])})
    validate.check_date_parse_rate(df, "Publish_Date", "videos", min_rate=0.9)
