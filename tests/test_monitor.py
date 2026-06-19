"""monitor 测试：监控逻辑与降级用 fake client 注入，不触网。"""

from __future__ import annotations

import pandas as pd

from src import monitor
from src.scraper import ScrapeBlockedError


class _FakeClient:
    def __init__(self, ranking=None, search=None, raise_on=None):
        self._ranking = ranking or []
        self._search = search or {}
        self._raise_on = raise_on or set()

    def get_ranking(self, rid=0, type_="all"):
        if "ranking" in self._raise_on:
            raise ScrapeBlockedError("blocked")
        return self._ranking

    def search_videos(self, keyword, page=1):
        if "search" in self._raise_on:
            raise ScrapeBlockedError("blocked")
        return self._search.get(keyword, [])


# ---- 榜单 ----


def test_fetch_ranking_parses_live():
    raw = [
        {"title": "<em>原神</em>视频", "owner": {"name": "up1"},
         "stat": {"view": 100, "like": 10, "coin": 5}, "bvid": "BV1"},
    ]
    df, live = monitor.fetch_ranking(_FakeClient(ranking=raw))
    assert live is True
    assert df.iloc[0]["title"] == "原神视频"  # HTML 标签被清洗
    assert df.iloc[0]["view"] == 100


def test_fetch_ranking_falls_back_when_blocked():
    df, live = monitor.fetch_ranking(_FakeClient(raise_on={"ranking"}))
    assert live is False
    assert len(df) > 0  # 演示数据兜底
    assert set(["rank", "title", "view", "bvid"]).issubset(df.columns)


def test_diff_rankings_detects_changes():
    old = pd.DataFrame({"bvid": ["A", "B", "C"], "rank": [1, 2, 3], "title": ["a", "b", "c"]})
    new = pd.DataFrame({"bvid": ["B", "A", "D"], "rank": [1, 2, 3], "title": ["b", "a", "d"]})
    diff = monitor.diff_rankings(old, new)
    assert [n["title"] for n in diff["newcomers"]] == ["d"]  # D 新上榜
    assert diff["dropped_count"] == 1  # C 跌出
    # B 从 2→1 上升 1，A 从 1→2 下降 1
    changes = {m["title"]: m["change"] for m in diff["movements"]}
    assert changes["b"] == 1 and changes["a"] == -1


def test_save_and_list_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "SNAPSHOT_DIR", tmp_path)
    df = pd.DataFrame({"rank": [1], "title": ["x"], "bvid": ["B1"]})
    path = monitor.save_snapshot(df, "ranking")
    assert path.exists()
    assert path in monitor.list_snapshots("ranking")


# ---- 竞品 ----


def test_track_competitors_aggregates():
    search = {
        "原神": [{"title": "原神a", "play": 100}, {"title": "原神b", "play": 300}],
        "鸣潮": [{"title": "鸣潮a", "play": 50}],
    }
    df, live = monitor.track_competitors(["原神", "鸣潮"], client=_FakeClient(search=search))
    assert live is True
    genshin = df[df["竞品"] == "原神"].iloc[0]
    assert genshin["搜索命中视频数"] == 2
    assert genshin["总播放"] == 400
    # 按总播放降序，原神在前
    assert df.iloc[0]["竞品"] == "原神"


def test_track_competitors_falls_back_when_all_blocked():
    df, live = monitor.track_competitors(["原神"], client=_FakeClient(raise_on={"search"}))
    assert live is False
    assert len(df) > 0  # 演示数据兜底
