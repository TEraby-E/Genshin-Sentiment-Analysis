from __future__ import annotations

import json

import pandas as pd

from src.finetune import analysis


def test_load_genshin_sentiment_jsonl_parses_outputs(tmp_path):
    path = tmp_path / "genshin_sentiment.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instruction": "x",
                        "input": "剧情很好",
                        "output": json.dumps({"sentiment": "正面", "aspects": ["剧情"]}, ensure_ascii=False),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "instruction": "x",
                        "input": "抽卡又歪了",
                        "output": json.dumps({"sentiment": "负面", "aspects": ["抽卡", "运营"]}, ensure_ascii=False),
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    frame = analysis.load_genshin_sentiment_jsonl(path)
    assert list(frame["text"]) == ["剧情很好", "抽卡又歪了"]
    assert list(frame["sentiment"]) == ["正面", "负面"]
    assert list(frame["aspects"]) == [["剧情"], ["抽卡", "运营"]]
    assert list(frame["aspect_count"]) == [1, 2]


def test_build_sentiment_dataset_report_computes_negative_rate():
    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "sentiment": ["正面", "负面", "负面"],
            "aspects": [["剧情"], ["抽卡"], ["抽卡", "运营"]],
            "aspect_count": [1, 1, 2],
            "is_negative": [False, True, True],
        }
    )

    report = analysis.build_sentiment_dataset_report(frame)
    aspect = next(row for row in report.aspect_rows if row.aspect == "抽卡")
    assert report.n_comments == 3
    assert report.negative_rate == 2 / 3
    assert aspect.negative_rate == 1.0
    assert aspect.negative_share_of_dataset == 2 / 3
    assert report.overall.startswith("整体负面率")


def test_build_sentiment_dataset_report_handles_empty_aspects():
    frame = pd.DataFrame(
        {
            "text": ["a"],
            "sentiment": ["中性"],
            "aspects": [[]],
            "aspect_count": [0],
            "is_negative": [False],
        }
    )

    report = analysis.build_sentiment_dataset_report(frame)
    assert report.aspect_rows == []
    assert report.insights == []
