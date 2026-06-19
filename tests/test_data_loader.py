from __future__ import annotations

import pandas as pd
import pytest

from src import config, data_loader, validate


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch, videos_df, comments_df, posts_df):
    videos_df.to_csv(tmp_path / config.FILES["videos"], index=False)
    comments_df.drop(columns=["cluster_name"]).to_csv(
        tmp_path / config.FILES["comments"], index=False
    )
    posts_df.to_csv(tmp_path / config.FILES["posts"], index=False)

    topic_names = [f"主题{i}" for i in range(8)]
    keywords = pd.DataFrame({"Topic_Cluster": range(8), "Cluster_Name": topic_names})
    keywords.to_csv(tmp_path / config.FILES["video_keywords"], index=False)

    ckeywords = pd.DataFrame({"Cluster": range(6), "Cluster_Name": [f"簇{i}" for i in range(6)]})
    ckeywords.to_csv(tmp_path / config.FILES["comment_keywords"], index=False)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_load_videos_maps_topic_and_parses_date(tmp_data_dir):
    videos = data_loader.load_videos()
    assert "topic" in videos.columns
    assert pd.api.types.is_datetime64_any_dtype(videos["Publish_Date"])


def test_load_comments_maps_cluster_name(tmp_data_dir):
    comments = data_loader.load_comments()
    assert "cluster_name" in comments.columns
    assert comments["cluster_name"].notna().any()


def test_load_posts_parses_date(tmp_data_dir):
    posts = data_loader.load_posts()
    assert pd.api.types.is_datetime64_any_dtype(posts["Publish_Date"])


def test_load_all_returns_three_frames(tmp_data_dir):
    videos, comments, posts = data_loader.load_all()
    assert len(videos) > 0 and len(comments) > 0 and len(posts) > 0


def test_load_videos_raises_on_missing_columns(tmp_path, monkeypatch, videos_df):
    videos_df.drop(columns=["Amount_View"]).to_csv(
        tmp_path / config.FILES["videos"], index=False
    )
    topic_names = [f"主题{i}" for i in range(8)]
    pd.DataFrame({"Topic_Cluster": range(8), "Cluster_Name": topic_names}).to_csv(
        tmp_path / config.FILES["video_keywords"], index=False
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(validate.SchemaError):
        data_loader.load_videos()
