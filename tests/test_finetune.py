"""finetune.dataset_formatter 单测：纯数据变换，不加载任何大模型/依赖。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.finetune import dataset_formatter as df


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_text": ["抽卡又歪了", "剧情很好", "短", "服务器崩了"],
            "llm_sentiment": ["负面", "正面", "负面", "狂喜"],  # 末条情感非法
            "llm_aspects": ["['抽卡']", "['剧情']", "['其他']", "['运营']"],
            "llm_reason": ["保底歪了体验差", "角色塑造到位", "x", ""],  # 第3条理由过短、第4条空
        }
    )


def test_parse_aspects_handles_str_list_and_invalid():
    assert df._parse_aspects("['抽卡']") == ["抽卡"]
    assert df._parse_aspects(["剧情", "天气"]) == ["剧情"]  # 过滤取值域外
    assert df._parse_aspects("") == []


def test_to_alpaca_records_filters_low_confidence():
    records = df.to_alpaca_records(_sample_df())
    # 仅前两条满足：合法情感 + 理由足够长 + 文本不过短
    assert len(records) == 2
    assert all(set(r) == {"instruction", "input", "output"} for r in records)
    out0 = json.loads(records[0]["output"])
    assert out0["sentiment"] == "负面"
    assert out0["aspects"] == ["抽卡"]


def test_to_alpaca_records_output_is_valid_json():
    for r in df.to_alpaca_records(_sample_df()):
        parsed = json.loads(r["output"])
        assert parsed["sentiment"] in {"正面", "中性", "负面"}


def test_format_dataset_writes_jsonl_and_info(tmp_path):
    src = tmp_path / "ai_analysis.csv"
    _sample_df().to_csv(src, index=False)
    res = df.format_dataset(src, tmp_path / "out")
    assert res["n_records"] == 2

    jsonl_path = res["jsonl"]
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["instruction"]

    info = json.loads(res["dataset_info"].read_text(encoding="utf-8"))
    assert df.DATASET_NAME in info
    assert info[df.DATASET_NAME]["file_name"] == "genshin_sentiment.jsonl"


def test_format_dataset_raises_when_no_high_confidence(tmp_path):
    bad = pd.DataFrame(
        {"clean_text": ["短"], "llm_sentiment": ["负面"], "llm_aspects": ["['其他']"],
         "llm_reason": [""]}
    )
    src = tmp_path / "bad.csv"
    bad.to_csv(src, index=False)
    with pytest.raises(RuntimeError):
        df.format_dataset(src, tmp_path / "out")


def test_load_labeled_requires_columns(tmp_path):
    src = tmp_path / "x.csv"
    pd.DataFrame({"foo": [1]}).to_csv(src, index=False)
    with pytest.raises(KeyError):
        df.load_labeled(src)
