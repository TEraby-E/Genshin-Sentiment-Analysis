"""舆情打标训练测试：训练/评估/存取/推理在合成标注上跑通，不触网；LLM 标注用 fake client。"""

from __future__ import annotations

import pandas as pd
import pytest

from src import sentiment_train


def _synthetic_corpus():
    texts = (
        ["这剧情真好看哭了", "角色塑造太棒了", "二创质量高社区氛围好"] * 8
        + ["抽卡又歪了氪金体验差", "服务器卡顿客服补偿少", "数值不平衡太折磨"] * 8
        + ["今天天气不错", "签到领原石", "更新公告看一下"] * 8
    )
    labels = ["正面"] * 24 + ["负面"] * 24 + ["中性"] * 24
    return texts, labels


def test_train_classifier_metrics_and_predict():
    texts, labels = _synthetic_corpus()
    res = sentiment_train.train_classifier(texts, labels, random_state=0)
    m = res["metrics"]
    assert m["n_train"] + m["n_test"] == len(texts)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert "agreement_with_teacher" in m
    # 学生模型应能区分明显的正负面
    preds = sentiment_train.predict(res["model"], ["抽卡歪了好气", "剧情太感人了"])
    assert preds[0] == "负面" and preds[1] == "正面"


def test_train_classifier_rejects_single_class():
    with pytest.raises(ValueError):
        sentiment_train.train_classifier(["a", "b", "c"], ["正面", "正面", "正面"])


def test_save_and_load_roundtrip(tmp_path):
    texts, labels = _synthetic_corpus()
    res = sentiment_train.train_classifier(texts, labels, random_state=0)
    path = sentiment_train.save_model(res["model"], tmp_path / "clf.joblib")
    assert path.exists()
    loaded = sentiment_train.load_model(path)
    assert sentiment_train.predict(loaded, ["抽卡歪了"]) == ["负面"]


def test_build_training_data_uses_llm_labels(monkeypatch):
    """build_training_data 应把 analyze_comments 的 llm_sentiment 转成 (text,label) 训练集。"""
    from src import text_pipeline

    def fake_analyze(comments, *, text_column="Comment_Content", sample=None, client=None):
        return pd.DataFrame(
            {"clean_text": ["抽卡歪了", "剧情好"], "llm_sentiment": ["负面", "正面"]}
        )

    monkeypatch.setattr(text_pipeline, "analyze_comments", fake_analyze)
    df = sentiment_train.build_training_data(
        pd.DataFrame({"Comment_Content": ["x", "y"]}), sample=2
    )
    assert list(df.columns) == ["text", "label"]
    assert list(df["label"]) == ["负面", "正面"]
