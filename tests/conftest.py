"""共享的合成测试数据：不依赖 Kaggle 原始数据，保证测试可在 CI 中独立运行。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def videos_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2024-07-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Topic_Cluster": rng.integers(0, 8, n),
            "topic": rng.choice(["角色专题", "二创剪辑", "攻略解析", "混合内容"], n),
            "Amount_View": rng.lognormal(mean=8, sigma=1.5, size=n).astype(int),
            "Amount_Like": rng.lognormal(mean=6, sigma=1.5, size=n).astype(int),
            "Amount_Favourite": rng.lognormal(mean=5, sigma=1.5, size=n).astype(int),
            "Publish_Date": dates,
            "Video_Title": [f"标题{i}" * (i % 3 + 1) for i in range(n)],
            "TimeInSeconds": rng.integers(30, 1200, n),
            "Video_Length_Type": rng.choice(["短视频", "中视频", "长视频"], n),
            "Author": rng.choice([f"up_{i}" for i in range(40)], n),
        }
    )


@pytest.fixture
def posts_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    months = pd.date_range("2024-07-01", periods=12, freq="MS")
    per_month = 8
    dates = [m + pd.Timedelta(days=int(d)) for m in months for d in rng.integers(0, 27, per_month)]
    n = len(dates)
    return pd.DataFrame(
        {
            "Post_ID": range(1, n + 1),
            "Publish_Date": dates,
            "Like_Count": rng.lognormal(mean=9, sigma=0.8, size=n).astype(int),
            "Comment_Count": rng.lognormal(mean=6, sigma=0.8, size=n).astype(int),
            "Collaboration_Flag": [i % per_month == 0 for i in range(n)],
        }
    )


@pytest.fixture
def comments_df(posts_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 2400
    cluster_names = [
        "Character Appreciation & Fandom Memes",
        "Story Criticism & Operational Skepticism",
        "Player Daily Life & Social Interaction",
        "Community Controversy & Negative Sentiment",
        "Game Mechanics & Optimization Feedback",
        "Rewards & Resource Acquisition",
    ]
    texts = [
        "这次剧情写得真好，角色塑造很到位",
        "抽卡保底又歪了，氪金体验很差",
        "活动关卡设计太烂了，运营听劝",
        "圣遗物词条数值机制不平衡，建议加强",
        "服务器卡顿，客服补偿太少",
        "二创同人质量很高，社区氛围不错",
    ]
    return pd.DataFrame(
        {
            "Post_ID": rng.choice(posts_df["Post_ID"], n),
            "Cluster": rng.integers(0, 6, n),
            "cluster_name": rng.choice(cluster_names, n),
            "Comment_Content": rng.choice(texts, n),
        }
    )
