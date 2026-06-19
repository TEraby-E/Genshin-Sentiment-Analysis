"""合成演示数据：当 data/ 下没有真实 Kaggle CSV 时，让看板仍能开箱即跑。

与 tests/conftest.py 的 fixture 形状一致，但规模更大、文本更贴近真实评论，
方便演示「作品打标 / 作者分类与增长 / 爆点预测与总结」三个工具。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TOPICS = ["角色专题", "二创剪辑", "攻略解析", "剧情解读", "音乐MMD", "新版本前瞻", "混合内容"]
_TITLE_TEMPLATES = [
    "{role}全方位培养攻略，零氪也能毕业",
    "【原神】{role}个人MMD/二创合集",
    "{patch}版本前瞻全解读，这次的福利有点猛",
    "深度剧情解析：{role}背后的世界观伏笔",
    "{role}抽卡复盘，保底到底歪不歪",
    "新活动副本速通教学，奖励一个不落",
    "盘点那些让人破防的名场面",
]
_ROLES = ["纳西妲", "雷电将军", "钟离", "胡桃", "枫原万叶", "神里绫华", "夜兰"]
_PATCHES = ["4.8", "5.0", "5.1", "5.2", "5.3"]

_COMMENT_TEXTS = [
    "这次剧情写得真好，角色塑造很到位，看哭了",
    "抽卡保底又歪了，氪了好几百还没出，氪金体验真的差",
    "活动关卡设计太肝了，奖励还少，运营听劝啊",
    "圣遗物词条又是暴击双爆歪到防御，数值机制太折磨",
    "服务器卡顿闪退，客服补偿就给五个原石，离谱",
    "二创质量越来越高了，社区氛围是真的好，梗也好玩",
    "前瞻直播信息量好大，新角色立绘绝了",
    "这版本平衡性有问题，新角色强度碾压老角色",
    "求求优化一下大世界探索体验吧，跑图太累了",
    "音乐做得太顶了，原神还是有点东西的",
]


def generate_videos(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2024-07-01") + pd.to_timedelta(rng.integers(0, 480, n), unit="D")
    clusters = rng.integers(0, len(_TOPICS), n)
    titles = []
    for _ in range(n):
        tpl = rng.choice(_TITLE_TEMPLATES)
        titles.append(tpl.format(role=rng.choice(_ROLES), patch=rng.choice(_PATCHES)))
    return pd.DataFrame(
        {
            "Topic_Cluster": clusters,
            "topic": [_TOPICS[c] for c in clusters],
            "Amount_View": rng.lognormal(mean=8, sigma=1.6, size=n).astype(int),
            "Amount_Like": rng.lognormal(mean=6, sigma=1.5, size=n).astype(int),
            "Amount_Favourite": rng.lognormal(mean=5, sigma=1.5, size=n).astype(int),
            "Publish_Date": dates,
            "Video_Title": titles,
            "TimeInSeconds": rng.integers(30, 1500, n),
            "Video_Length_Type": rng.choice(["短视频", "中视频", "长视频"], n),
            "Author": rng.choice([f"UP主_{i}" for i in range(220)], n),
        }
    )


def generate_posts(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2024-07-01", periods=16, freq="MS")
    per_month = 10
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


def generate_comments(posts: pd.DataFrame, n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cluster_names = [
        "Character Appreciation & Fandom Memes",
        "Story Criticism & Operational Skepticism",
        "Player Daily Life & Social Interaction",
        "Community Controversy & Negative Sentiment",
        "Game Mechanics & Optimization Feedback",
        "Rewards & Resource Acquisition",
    ]
    return pd.DataFrame(
        {
            "Post_ID": rng.choice(posts["Post_ID"], n),
            "Cluster": rng.integers(0, len(cluster_names), n),
            "cluster_name": rng.choice(cluster_names, n),
            "Comment_Content": rng.choice(_COMMENT_TEXTS, n),
        }
    )


def load_demo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    videos = generate_videos()
    posts = generate_posts()
    comments = generate_comments(posts)
    return videos, comments, posts
