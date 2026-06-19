"""数据加载与基础清洗：读取五个 CSV 并补充派生字段。"""

from __future__ import annotations

import logging

import pandas as pd

from . import config, validate

logger = logging.getLogger(__name__)


def _path(key: str):
    return config.DATA_DIR / config.FILES[key]


def load_videos() -> pd.DataFrame:
    """视频数据 + 主题名映射 + 时间解析。"""
    videos = pd.read_csv(_path("videos"))
    vkey = pd.read_csv(_path("video_keywords"))
    validate.check_not_empty(videos, "videos")
    validate.check_columns(videos, "videos")

    # 关键词表中 Cluster_Name 含 NaN 行，去掉后再建映射，避免映射不全
    vkey_clean = vkey.dropna(subset=["Cluster_Name"]).drop_duplicates("Topic_Cluster")
    name_map = dict(zip(vkey_clean["Topic_Cluster"], vkey_clean["Cluster_Name"]))

    videos["topic"] = videos["Topic_Cluster"].map(name_map)
    unmapped = videos["topic"].isna().sum()
    if unmapped:
        logger.warning("%d 条视频的 Topic_Cluster 未在关键词表中找到对应主题名", unmapped)

    videos["Publish_Date"] = pd.to_datetime(videos["Publish_Date"], errors="coerce")
    validate.check_date_parse_rate(videos, "Publish_Date", "videos")
    return videos


def load_comments() -> pd.DataFrame:
    """评论数据 + 情感/主题簇名映射。"""
    comments = pd.read_csv(_path("comments"))
    ckey = pd.read_csv(_path("comment_keywords"))
    validate.check_not_empty(comments, "comments")
    validate.check_columns(comments, "comments")

    comments["cluster_name"] = comments["Cluster"].map(
        dict(zip(ckey["Cluster"], ckey["Cluster_Name"]))
    )
    return comments


def load_posts() -> pd.DataFrame:
    """官方帖子数据 + 时间解析。"""
    posts = pd.read_csv(_path("posts"))
    validate.check_not_empty(posts, "posts")
    validate.check_columns(posts, "posts")

    posts["Publish_Date"] = pd.to_datetime(posts["Publish_Date"], errors="coerce")
    validate.check_date_parse_rate(posts, "Publish_Date", "posts")
    return posts


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_videos(), load_comments(), load_posts()
