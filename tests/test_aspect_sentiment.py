from __future__ import annotations

import pytest

from src import aspect_sentiment, config


def test_tag_aspects_matches_known_keywords():
    assert "抽卡" in aspect_sentiment.tag_aspects("这次抽卡保底又歪了")
    assert "剧情" in aspect_sentiment.tag_aspects("剧情和主线写得不错")
    assert aspect_sentiment.tag_aspects("") == []
    assert aspect_sentiment.tag_aspects(None) == []


def test_tag_aspects_can_hit_multiple_aspects():
    hits = aspect_sentiment.tag_aspects("活动剧情写得不错，但抽卡保底歪了")
    assert {"活动", "剧情", "抽卡"} <= set(hits)


def test_aspect_breakdown_requires_is_neg_column(comments_df):
    with pytest.raises(KeyError, match="is_neg"):
        aspect_sentiment.aspect_breakdown(comments_df)


def test_aspect_breakdown_computes_neg_rate_per_aspect(comments_df):
    df = comments_df.copy()
    df["is_neg"] = df["cluster_name"].isin(config.NEGATIVE_CLUSTERS)
    breakdown = aspect_sentiment.aspect_breakdown(df)
    assert "mentions" in breakdown.columns
    assert "neg_rate" in breakdown.columns
    assert ((breakdown["neg_rate"] >= 0) & (breakdown["neg_rate"] <= 100)).all()


def test_agreement_with_cluster_reports_coverage(comments_df):
    coverage = aspect_sentiment.agreement_with_cluster(comments_df)
    assert coverage["total_comments"] == len(comments_df)
    assert 0 <= coverage["coverage_rate"] <= 100


def test_classify_with_llm_delegates_to_pipeline(monkeypatch):
    """classify_with_llm 现已落地：用 monkeypatch 替换真实 API 调用，验证它把
    text_pipeline 的结构化结果转换成与 tag_aspects 一致的方面标签列表。"""
    from src import text_pipeline

    def fake(texts, **kw):
        return [{"aspects": ["抽卡"], "sentiment": "负面", "reason": ""} for _ in texts]

    monkeypatch.setattr(text_pipeline, "classify_with_llm", fake)
    out = aspect_sentiment.classify_with_llm(["抽卡又歪了"])
    assert out == [["抽卡"]]
