"""B 站抓取适配层测试：用 fake transport，不触网。"""

from __future__ import annotations

from pathlib import Path

from src.scraper.bilibili import BiliClient, VideoRecord, crawl_recent_comments


def test_search_videos_normalizes_search_response(tmp_path: Path):
    def fake_transport(url, _headers):
        assert "search/type" in url
        return {
            "data": {
                "result": [
                    {
                        "aid": 123,
                        "bvid": "BV123",
                        "title": '<em class="keyword">原神</em> 新版本',
                        "author": "up主",
                        "pubdate": 1710000000,
                        "play": "1000",
                        "like": "88",
                    }
                ]
            }
        }

    client = BiliClient(cache_dir=tmp_path, transport=fake_transport, sleep_seconds=0)
    videos = client.search_videos("原神")
    assert videos == [
        VideoRecord(
            aid=123,
            bvid="BV123",
            title="原神 新版本",
            up="up主",
            pubdate=1710000000,
            view=1000,
            like=88,
        )
    ]


def test_fetch_comments_normalizes_comment_response(tmp_path: Path):
    def fake_transport(url, _headers):
        assert "reply/main" in url
        return {
            "data": {
                "replies": [
                    {
                        "rpid": 1,
                        "ctime": 1710000001,
                        "like": 10,
                        "content": {"message": "抽卡又歪了，真难受"},
                        "member": {"uname": "玩家A"},
                    }
                ],
                "cursor": {"is_end": True},
            }
        }

    client = BiliClient(cache_dir=tmp_path, transport=fake_transport, sleep_seconds=0)
    video = VideoRecord(aid=123, bvid="BV123", title="原神 新版本", up="up主")
    comments = client.fetch_comments(video)
    assert len(comments) == 1
    assert comments[0].Comment_Content == "抽卡又歪了，真难受"
    assert comments[0].bvid == "BV123"
    assert comments[0].user == "玩家A"


def test_crawl_recent_comments_returns_model_ready_dataframe(tmp_path: Path):
    def fake_transport(url, _headers):
        if "search/type" in url:
            return {
                "data": {
                    "result": [
                        {
                            "aid": 123,
                            "bvid": "BV123",
                            "title": "原神 新版本",
                            "author": "up主",
                            "pubdate": 1710000000,
                        }
                    ]
                }
            }
        return {
            "data": {
                "replies": [
                    {
                        "rpid": 1,
                        "ctime": 1710000001,
                        "like": 10,
                        "content": {"message": "剧情很好"},
                        "member": {"uname": "玩家A"},
                    }
                ],
                "cursor": {"is_end": True},
            }
        }

    client = BiliClient(cache_dir=tmp_path, transport=fake_transport, sleep_seconds=0)
    df = crawl_recent_comments(max_videos=1, comments_per_video=5, client=client)
    assert list(df["Comment_Content"]) == ["剧情很好"]
    assert list(df["bvid"]) == ["BV123"]
    assert "video_pubdate" in df.columns
