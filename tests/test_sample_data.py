"""合成演示数据测试：保证看板在无真实数据时仍能拿到可打标的评论文本。"""

from __future__ import annotations

from src import sample_data


def test_load_demo_shapes_and_columns():
    videos, comments, posts = sample_data.load_demo()
    assert len(videos) > 0 and len(comments) > 0 and len(posts) > 0
    for col in ["Topic_Cluster", "topic", "Amount_View", "Publish_Date", "Author"]:
        assert col in videos.columns
    assert "Comment_Content" in comments.columns
    assert "Post_ID" in posts.columns


def test_demo_comment_texts_available_for_tagging():
    """看板的智能路由打标需要现成的评论文本作为演示输入。"""
    assert sample_data._COMMENT_TEXTS
    assert all(isinstance(t, str) and t.strip() for t in sample_data._COMMENT_TEXTS)
