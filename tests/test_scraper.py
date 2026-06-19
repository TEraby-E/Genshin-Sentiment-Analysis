"""BilibiliClient 测试：WBI 签名为纯函数可确定性断言；网络层用 fake session 注入，不触网。"""

from __future__ import annotations

import pytest

from src.scraper import ScrapeBlockedError
from src.scraper.bilibili import BilibiliClient


class _FakeResp:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status


class _FakeSession:
    """按 url 子串匹配返回预设响应，模拟 requests.Session。"""

    def __init__(self, routes: dict[str, _FakeResp]):
        self.routes = routes
        self.headers: dict = {}
        self.calls: list[str] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return _FakeResp("{}")


def _client_with(routes: dict[str, _FakeResp]) -> BilibiliClient:
    c = BilibiliClient(cache_dir=None, min_interval=0)
    c._session = _FakeSession(routes)  # 绕过真实 cookie 初始化
    return c


# ---- WBI 签名：纯函数确定性 ----


def test_wbi_sign_is_deterministic_and_adds_fields(monkeypatch):
    c = BilibiliClient(cache_dir=None, min_interval=0)
    # 固定 WBI 原始密钥，绕过 nav 网络请求
    keys = ("7cd084941338484aae1ad9425b84077c", "4932caff0ff746eab6f01bf08b70ac45")
    monkeypatch.setattr(c, "_wbi_keys", lambda: keys)
    mk = c._mixin_key()
    assert len(mk) == 32

    monkeypatch.setattr("src.scraper.bilibili.time.time", lambda: 1_700_000_000)
    signed = c._sign({"keyword": "原神", "page": 1})
    assert signed["wts"] == 1_700_000_000
    assert len(signed["w_rid"]) == 32  # md5 十六进制
    # 相同输入 + 相同时间戳 → 相同签名
    assert c._sign({"keyword": "原神", "page": 1})["w_rid"] == signed["w_rid"]


# ---- 网络层：风控/解析 ----


def test_get_json_raises_blocked_on_html():
    c = _client_with({"ranking": _FakeResp("<!DOCTYPE html><html>风控</html>")})
    with pytest.raises(ScrapeBlockedError):
        c._get_json(c.RANKING_URL)


def test_get_json_raises_blocked_on_code_412():
    c = _client_with({"ranking": _FakeResp('{"code": -412, "message": "blocked"}')})
    with pytest.raises(ScrapeBlockedError):
        c._get_json(c.RANKING_URL)


def test_get_ranking_parses_list():
    payload = '{"code":0,"data":{"list":[{"title":"a","bvid":"BV1"},{"title":"b","bvid":"BV2"}]}}'
    c = _client_with({"ranking": _FakeResp(payload)})
    out = c.get_ranking()
    assert [v["bvid"] for v in out] == ["BV1", "BV2"]


def test_search_videos_parses_result():
    payload = '{"code":0,"data":{"result":[{"title":"<em>原神</em>","play":100}]}}'
    c = _client_with({"search/type": _FakeResp(payload)})
    # 绕过 WBI（_sign 需要 nav），直接 stub
    c._sign = lambda p: p  # type: ignore[method-assign]
    out = c.search_videos("原神")
    assert out[0]["play"] == 100
