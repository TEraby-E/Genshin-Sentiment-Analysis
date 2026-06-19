"""B 站公开 Web API 客户端（含 WBI 签名）。

「逆向」部分指复刻 B 站前端公开接口的 WBI 签名算法——这是公开文档化的、
调用其搜索/空间等接口的必要步骤，并非破解登录或私密数据。本客户端：

- 只请求公开数据：排行榜 ranking、搜索 search、公开视频统计 view；
- 自带请求限速（min_interval）、磁盘缓存（避免重复请求）、有限重试；
- 命中风控（返回 HTML 拦截页 / -412）时抛出 ScrapeBlockedError，由上层降级到缓存/演示数据，
  而不是反复重试或试图绕过——尊重对方的反爬策略。
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
import urllib.parse
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# WBI mixin 重排表（B 站公开前端逻辑，固定常量）
_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61,
    26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36,
    20, 34, 44, 52,
]

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
}


class ScrapeBlockedError(RuntimeError):
    """被风控拦截或返回非 JSON 时抛出，提示上层降级到缓存/演示数据。"""


class BilibiliClient:
    """B 站公开数据抓取客户端。

    Args:
        min_interval: 相邻请求最小间隔秒数（限速，默认 1.5s）。
        cache_dir: 磁盘缓存目录；None 表示不缓存。
        cache_ttl: 缓存有效期秒数（默认 1 小时）。
        timeout: 单请求超时秒数。
    """

    NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
    RANKING_URL = "https://api.bilibili.com/x/web-interface/ranking/v2"
    SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"
    VIEW_URL = "https://api.bilibili.com/x/web-interface/view"

    def __init__(
        self,
        *,
        min_interval: float = 1.5,
        cache_dir: str | Path | None = "data/cache/bili",
        cache_ttl: float = 3600,
        timeout: float = 12,
    ):
        self.min_interval = min_interval
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self._session: Any | None = None

    # ---- 会话与限速 ----
    def _get_session(self) -> Any:
        if self._session is None:
            import requests

            s = requests.Session()
            s.headers.update(_DEFAULT_HEADERS)
            # 访问首页领取 buvid3 等基础 cookie（搜索接口需要）
            try:
                s.get("https://www.bilibili.com", timeout=self.timeout)
                s.get("https://api.bilibili.com/x/frontend/finger/spi", timeout=self.timeout)
            except Exception as e:  # noqa: BLE001
                logger.warning("初始化 cookie 失败（不影响排行榜接口）：%s", e)
            self._session = s
        return self._session

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()

    def _get_json(self, url: str, params: dict | None = None, *, retries: int = 2) -> dict:
        session = self._get_session()
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            self._throttle()
            try:
                resp = session.get(url, params=params, timeout=self.timeout)
                text = resp.text.lstrip()
                if not text.startswith("{"):
                    # 返回 HTML 拦截页 → 风控，不再重试
                    raise ScrapeBlockedError(
                        f"接口返回非 JSON（疑似风控拦截），HTTP {resp.status_code}"
                    )
                data = json.loads(text)
                code = data.get("code")
                if code not in (0, None):
                    # -412 = 请求被拦截；-799 = 频繁
                    if code in (-412, -799):
                        raise ScrapeBlockedError(f"接口 code={code}（风控/频繁），建议降级")
                    logger.warning("接口 %s 返回 code=%s: %s", url, code, data.get("message"))
                return data
            except ScrapeBlockedError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("请求失败 %d/%d：%s", attempt, retries, e)
                if attempt < retries:
                    time.sleep(2**attempt)
        raise ScrapeBlockedError(f"请求最终失败：{last_err}")

    # ---- 缓存 ----
    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _cached(self, key: str, fetch) -> dict:
        path = self._cache_path(key)
        if path and path.exists() and (time.time() - path.stat().st_mtime) < self.cache_ttl:
            return json.loads(path.read_text(encoding="utf-8"))
        data = fetch()
        if path:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    # ---- WBI 签名 ----
    @functools.lru_cache(maxsize=1)
    def _wbi_keys(self) -> tuple[str, str]:
        data = self._get_json(self.NAV_URL)
        img = data["data"]["wbi_img"]
        img_key = img["img_url"].rsplit("/", 1)[1].split(".")[0]
        sub_key = img["sub_url"].rsplit("/", 1)[1].split(".")[0]
        return img_key, sub_key

    def _mixin_key(self) -> str:
        img_key, sub_key = self._wbi_keys()
        raw = img_key + sub_key
        return "".join(raw[i] for i in _MIXIN_KEY_ENC_TAB)[:32]

    def _sign(self, params: dict) -> dict:
        """给参数加 wts + w_rid 签名（WBI）。"""
        signed = dict(params)
        signed["wts"] = int(time.time())
        query = "&".join(
            f"{k}={urllib.parse.quote(str(signed[k]), safe='')}" for k in sorted(signed)
        )
        signed["w_rid"] = hashlib.md5((query + self._mixin_key()).encode()).hexdigest()
        return signed

    # ---- 公开数据接口 ----
    def get_ranking(self, rid: int = 0, type_: str = "all") -> list[dict]:
        """全站/分区排行榜（榜单监控）。rid=0 为全站，type_='all'。无需 WBI，最稳定。"""
        def fetch():
            return self._get_json(self.RANKING_URL, {"rid": rid, "type": type_})

        data = self._cached(f"ranking:{rid}:{type_}", fetch)
        return data.get("data", {}).get("list", [])

    def search_videos(self, keyword: str, page: int = 1) -> list[dict]:
        """视频搜索（竞品动态抓取）。需要 WBI 签名 + buvid cookie。"""
        def fetch():
            params = self._sign({"search_type": "video", "keyword": keyword, "page": page})
            return self._get_json(self.SEARCH_URL, params)

        data = self._cached(f"search:{keyword}:{page}", fetch)
        return data.get("data", {}).get("result", []) or []

    def get_video_stat(self, bvid: str) -> dict:
        """单个公开视频的统计数据（播放/点赞/投币等）。"""
        def fetch():
            return self._get_json(self.VIEW_URL, {"bvid": bvid})

        data = self._cached(f"view:{bvid}", fetch)
        return data.get("data", {}).get("stat", {})
