"""核心分析逻辑：数据质量审查、玩家生态、舆情监控、爆款预测。"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config, stats_tests


def data_quality_report(videos: pd.DataFrame) -> dict:
    """数据质量审查：噪声占比、长尾集中度。"""
    noise_pct = (videos["Topic_Cluster"] == config.NOISE_TOPIC_CLUSTER).mean() * 100
    q99 = videos["Amount_View"].quantile(0.99)
    top_share = (
        videos.loc[videos["Amount_View"] >= q99, "Amount_View"].sum()
        / videos["Amount_View"].sum()
        * 100
    )
    return {
        "noise_pct": round(noise_pct, 1),
        "median_view": int(videos["Amount_View"].median()),
        "max_view": int(videos["Amount_View"].max()),
        "top1pct_view_share": round(top_share, 1),
    }


def ecosystem_by_topic(videos: pd.DataFrame) -> pd.DataFrame:
    """各主题热度：视频数、中位播放、中位点赞，按中位播放降序。"""
    return (
        videos.groupby("topic")
        .agg(
            video_count=("Amount_View", "size"),
            median_view=("Amount_View", "median"),
            median_like=("Amount_Like", "median"),
        )
        .sort_values("median_view", ascending=False)
    )


def sentiment_trend(comments: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    """负面舆情月度趋势：关联评论与帖子时间，按月统计负面占比。"""
    comments = comments.copy()
    comments["is_neg"] = comments["cluster_name"].isin(config.NEGATIVE_CLUSTERS)
    comments["date"] = comments["Post_ID"].map(
        dict(zip(posts["Post_ID"], posts["Publish_Date"]))
    )
    comments = comments.dropna(subset=["date"])
    comments["ym"] = comments["date"].dt.to_period("M").astype(str)

    monthly = comments.groupby("ym").agg(
        total=("is_neg", "size"), neg=("is_neg", "sum")
    )
    monthly = monthly[monthly["total"] >= config.MIN_MONTHLY_COMMENTS]
    monthly["neg_rate"] = monthly["neg"] / monthly["total"] * 100
    return monthly


def alert_threshold(monthly: pd.DataFrame) -> dict:
    """根据基线计算舆情预警线，并标出超线月份。

    固定倍数阈值简单直观，但不回答"这个月的上升是否在统计上显著，
    而不只是样本噪声"——用 stats_tests.peak_vs_baseline_significance 补充这一点，
    并用 rolling_zscore_alert 提供一个能适应基线漂移的替代预警口径。
    """
    baseline = monthly["neg_rate"].median()
    threshold = baseline * config.ALERT_MULTIPLIER
    breaches = monthly[monthly["neg_rate"] > threshold]
    significance = stats_tests.peak_vs_baseline_significance(monthly)
    zscore_monthly = stats_tests.rolling_zscore_alert(monthly)
    return {
        "baseline": round(baseline, 1),
        "threshold": round(threshold, 1),
        "peak_month": monthly["neg_rate"].idxmax(),
        "peak_rate": round(monthly["neg_rate"].max(), 1),
        "breach_months": list(breaches.index),
        "significance": significance,
        "zscore_anomaly_months": list(zscore_monthly[zscore_monthly["is_anomaly"]].index),
    }


HIT_FEATURES = ["TimeInSeconds", "Topic_Cluster", "hour", "dayofweek", "title_len", "len_type_enc"]


def engineer_hit_features(videos: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """为爆款模型构造特征列与 is_hit 标签，返回处理后的 df 与长度类型编码器。

    抽成独立函数，让训练（train_hit_model）与交互式单视频打分（看板）共用同一套特征工程，
    避免两处口径漂移。
    """
    df = videos.copy()
    hit_floor = df["Amount_View"].quantile(config.HIT_QUANTILE)
    df["is_hit"] = (df["Amount_View"] >= hit_floor).astype(int)
    df["hour"] = df["Publish_Date"].dt.hour
    df["dayofweek"] = df["Publish_Date"].dt.dayofweek
    df["title_len"] = df["Video_Title"].astype(str).str.len()
    encoder = LabelEncoder()
    df["len_type_enc"] = encoder.fit_transform(df["Video_Length_Type"].astype(str))
    return df, encoder


def train_hit_model(videos: pd.DataFrame, random_state: int = 42) -> dict:
    """训练爆款预测模型：留出集评估 AUC，再在全量上重新拟合一个可部署/可交互打分的模型。

    返回 model（全量拟合）、auc（留出集诚实评估）、feature_importance、length_encoder，
    供看板的单视频爆款概率打分器直接复用。
    """
    df, encoder = engineer_hit_features(videos)
    X, y = df[HIT_FEATURES].fillna(0), df["is_hit"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    eval_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1
    )
    eval_model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, eval_model.predict_proba(X_te)[:, 1])

    full_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1
    )
    full_model.fit(X, y)
    importance = dict(
        sorted(
            zip(HIT_FEATURES, [round(float(x), 3) for x in full_model.feature_importances_]),
            key=lambda x: -x[1],
        )
    )
    return {
        "model": full_model,
        "auc": round(float(auc), 3),
        "feature_importance": importance,
        "length_encoder": encoder,
        "features": HIT_FEATURES,
    }


def hit_prediction(videos: pd.DataFrame, random_state: int = 42) -> dict:
    """爆款预测：用元数据特征训练随机森林，返回 AUC 与特征重要性。"""
    trained = train_hit_model(videos, random_state=random_state)
    return {"auc": trained["auc"], "feature_importance": trained["feature_importance"]}
