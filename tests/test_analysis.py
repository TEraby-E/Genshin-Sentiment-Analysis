from __future__ import annotations

from src import analysis, config


def test_data_quality_report_shape(videos_df):
    report = analysis.data_quality_report(videos_df)
    assert set(report) == {"noise_pct", "median_view", "max_view", "top1pct_view_share"}
    assert 0 <= report["noise_pct"] <= 100


def test_ecosystem_by_topic_sorted_descending(videos_df):
    eco = analysis.ecosystem_by_topic(videos_df)
    assert list(eco["median_view"]) == sorted(eco["median_view"], reverse=True)
    assert eco["video_count"].sum() == len(videos_df)


def test_sentiment_trend_filters_low_volume_months(comments_df, posts_df):
    monthly = analysis.sentiment_trend(comments_df, posts_df)
    assert (monthly["total"] >= config.MIN_MONTHLY_COMMENTS).all()
    assert "neg_rate" in monthly.columns
    assert ((monthly["neg_rate"] >= 0) & (monthly["neg_rate"] <= 100)).all()


def test_alert_threshold_contains_significance_and_zscore(comments_df, posts_df):
    monthly = analysis.sentiment_trend(comments_df, posts_df)
    alert = analysis.alert_threshold(monthly)
    assert alert["threshold"] >= alert["baseline"]
    assert "significance" in alert and "p_value" in alert["significance"]
    assert "zscore_anomaly_months" in alert


def test_hit_prediction_returns_valid_auc(videos_df):
    result = analysis.hit_prediction(videos_df, random_state=0)
    assert 0.0 <= result["auc"] <= 1.0
    assert set(result["feature_importance"]) == {
        "TimeInSeconds", "Topic_Cluster", "hour", "dayofweek", "title_len", "len_type_enc",
    }


def test_train_hit_model_returns_usable_model(videos_df):
    """看板的单视频打分器依赖这个返回的 model + length_encoder，验证可直接 predict。"""
    trained = analysis.train_hit_model(videos_df, random_state=0)
    assert hasattr(trained["model"], "predict_proba")
    assert set(trained["features"]) == set(analysis.HIT_FEATURES)
    enc = trained["length_encoder"]

    import pandas as pd

    x = pd.DataFrame(
        [{
            "TimeInSeconds": 300, "Topic_Cluster": 0, "hour": 20,
            "dayofweek": 5, "title_len": 18,
            "len_type_enc": int(enc.transform([enc.classes_[0]])[0]),
        }]
    )[trained["features"]]
    prob = trained["model"].predict_proba(x)[0, 1]
    assert 0.0 <= prob <= 1.0
