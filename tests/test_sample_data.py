"""合成演示数据测试：保证看板在无真实数据时仍能跑通下游分析函数。"""

from __future__ import annotations

from src import analysis, lifecycle, sample_data


def test_load_demo_shapes_and_columns():
    videos, comments, posts = sample_data.load_demo()
    assert len(videos) > 0 and len(comments) > 0 and len(posts) > 0
    # 下游分析依赖的关键列必须齐全
    for col in ["Topic_Cluster", "topic", "Amount_View", "Publish_Date", "Author"]:
        assert col in videos.columns
    assert "Comment_Content" in comments.columns
    assert "Post_ID" in posts.columns


def test_demo_data_feeds_downstream_analysis():
    """演示数据要能直接喂给爆款模型与生命周期分群而不报错。"""
    videos, _, _ = sample_data.load_demo()
    trained = analysis.train_hit_model(videos, random_state=0)
    assert 0.0 <= trained["auc"] <= 1.0
    creators = lifecycle.creator_lifecycle(videos)
    assert "stage" in creators.columns
    assert creators["stage"].notna().all()
