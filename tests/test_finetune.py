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


# ---- 训练/评估切分 ----


def test_split_records_preserves_all_and_no_overlap():
    df_big = pd.DataFrame(
        {
            "clean_text": [f"评论内容{i}" for i in range(10)],
            "llm_sentiment": (["负面", "正面"] * 5),
            "llm_aspects": ["['抽卡']"] * 10,
            "llm_reason": ["理由足够长用于通过过滤"] * 10,
        }
    )
    records = df.to_alpaca_records(df_big)
    train, ev = df.split_records(records, eval_ratio=0.2)
    assert len(train) + len(ev) == len(records)
    assert len(ev) >= 1
    train_inputs = {r["input"] for r in train}
    assert all(r["input"] not in train_inputs for r in ev)  # 无重叠


# ---- 第 3 步：评估与错例分析 ----


def test_evaluate_perfect_predictor():
    from src.finetune.evaluate import evaluate

    texts = ["剧情好", "抽卡歪了"]
    gold = ["正面", "负面"]
    rep = evaluate(lambda ts: ["正面", "负面"], texts, gold)
    assert rep.accuracy == 1.0
    assert rep.errors == []  # 全对，无错例
    # macro_f1 在固定三标签集上对缺席的「中性」会计 0，这是 sklearn 既定行为，不强求 1.0


def test_evaluate_finds_errors_and_flags_irony():
    from src.finetune.evaluate import evaluate

    texts = ["这运营真是好家伙", "剧情很好"]
    gold = ["负面", "正面"]
    rep = evaluate(lambda ts: ["正面", "正面"], texts, gold)  # 第一条判错
    assert rep.accuracy == 0.5
    assert len(rep.errors) == 1
    assert rep.errors[0]["irony"] is True  # “好家伙”被标为疑似反讽


def test_load_eval_set_roundtrip(tmp_path):
    from src.finetune.evaluate import load_eval_set

    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"instruction":"x","input":"抽卡歪了","output":"{\\"sentiment\\": \\"负面\\"}"}\n',
        encoding="utf-8",
    )
    texts, gold = load_eval_set(p)
    assert texts == ["抽卡歪了"]
    assert gold == ["负面"]
