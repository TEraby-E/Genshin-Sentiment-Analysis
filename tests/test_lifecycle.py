from __future__ import annotations

from src import lifecycle


def test_creator_lifecycle_classifies_single_post_creators(videos_df):
    creators = lifecycle.creator_lifecycle(videos_df)
    assert "stage" in creators.columns
    assert set(creators["stage"]) <= {"单发", "成长期", "稳定期", "沉寂期"}


def test_creator_lifecycle_single_video_author_is_danfa(videos_df):
    df = videos_df.copy()
    df["Author"] = ["author_solo"] + ["author_repeat"] * (len(df) - 1)
    creators = lifecycle.creator_lifecycle(df)
    assert creators.loc["author_solo", "stage"] == "单发"
    assert creators.loc["author_solo", "n_videos"] == 1


def test_creator_stage_summary_columns(videos_df):
    creators = lifecycle.creator_lifecycle(videos_df)
    summary = lifecycle.creator_stage_summary(creators)
    assert {"creator_count", "total_videos", "median_view_per_video"} <= set(summary.columns)
    assert summary["total_videos"].sum() == len(videos_df)


def test_topic_lifecycle_labels_known_categories(videos_df):
    result = lifecycle.topic_lifecycle(videos_df)
    assert set(result["stage"]) <= {"增长期", "衰退期", "平稳期", "新兴/数据不足"}
    assert "growth_rate" in result.columns
