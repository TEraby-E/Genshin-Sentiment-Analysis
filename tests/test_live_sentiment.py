"""最新评论舆情聚合测试。"""

from __future__ import annotations

import pandas as pd

from src.agents.router import RouterAgent
from src.live_sentiment import analyze_recent_comments, report_to_dict


def test_analyze_recent_comments_tags_and_summarizes(make_fake_track):
    track = make_fake_track(
        "keyword",
        0,
        sentiment_fn=lambda t: "负面" if "歪" in t or "烂" in t else "正面",
        aspects=["抽卡"],
        conf=0.8,
    )
    router = RouterAgent([track])
    comments = pd.DataFrame(
        {
            "Comment_Content": ["抽卡又歪了，真烂", "剧情很好"],
            "bvid": ["BV1", "BV2"],
        }
    )

    tagged, report = analyze_recent_comments(comments, router=router)

    assert list(tagged["sentiment"]) == ["负面", "正面"]
    assert list(tagged["track"]) == ["keyword", "keyword"]
    assert report.n_comments == 2
    assert report.sentiment_counts["负面"] == 1
    assert report.aspect_counts["抽卡"] == 2
    assert report.negative_examples == ["抽卡又歪了，真烂"]
    assert report_to_dict(report)["n_comments"] == 2


def test_analyze_recent_comments_requires_text_column(make_fake_track):
    router = RouterAgent([make_fake_track("keyword", 0)])
    try:
        analyze_recent_comments(pd.DataFrame({"text": ["x"]}), router=router)
    except KeyError as e:
        assert "Comment_Content" in str(e)
    else:
        raise AssertionError("expected KeyError")
