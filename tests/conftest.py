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


# ---- RAG 测试夹具：用「假嵌入」让检索器在无 GPU / 无外网的 CI 里也能跑 ----


@pytest.fixture
def fake_embedding_fn():
    """确定性假嵌入：满足 EmbeddingFunction 协议，按字符命中给固定维度打分。

    不依赖任何模型/网络，但能让「字面相近」的文本向量也相近，足以驱动检索断言。
    """
    vocab = "歪保底深渊螺旋圣遗物词条原石抽卡剧情活动运营"
    dim = len(vocab)

    def _embed(texts):
        vecs = []
        for t in texts:
            v = [float(t.count(ch)) for ch in vocab]
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs

    _embed.dim = dim  # type: ignore[attr-defined]
    return _embed


@pytest.fixture
def lore_documents() -> list[str]:
    """小型「梗 & 设定词典」语料，覆盖几条典型黑话释义。"""
    return [
        "歪了：指抽卡时没有抽到 UP 角色，而是出了常驻角色，玩家表达失望。",
        "保底：抽卡达到一定次数后必定获得高星角色的机制。",
        "深渊螺旋：高难度周期性副本，考验角色练度与配队深度。",
        "圣遗物词条：装备上的副属性，词条歪表示出的属性不理想。",
        "原石：游戏内可用于抽卡的核心资源。",
    ]


@pytest.fixture
def rag_retriever(fake_embedding_fn, lore_documents):
    """基于假嵌入与内存向量库构建的混合检索器，供 RAG 单测使用。"""
    from src.rag.retriever import HybridRetriever

    return HybridRetriever.from_documents(
        lore_documents, embedding_fn=fake_embedding_fn
    )


# ---- 路由 Agent 测试夹具：可编程的假轨道，确定性驱动路由/升档逻辑 ----


@pytest.fixture
def make_fake_track():
    """工厂：造一个满足 TaggingTrack 协议的假轨道，行为完全由参数控制。

    classify 用 sentiment_fn(text)->str 决定输出，便于断言路由把哪条评论送到哪个轨道、
    以及校验失败后是否升级到更高档轨道。
    """
    from src.agents.base import TagResult

    def _make(name, cost, *, available=True, sentiment_fn=None, aspects=None, conf=0.6):
        sent_fn = sentiment_fn or (lambda _t: "中性")

        class _FakeTrack:
            def __init__(self):
                self.name = name
                self.cost = cost
                self.calls: list[str] = []

            def is_available(self) -> bool:
                return available

            def classify(self, texts):
                self.calls.extend(texts)
                return [
                    TagResult(
                        text=t,
                        sentiment=sent_fn(t),
                        aspects=list(aspects or []),
                        reason=f"{name} 判定",
                        confidence=conf,
                        track=name,
                    )
                    for t in texts
                ]

        return _FakeTrack()

    return _make
