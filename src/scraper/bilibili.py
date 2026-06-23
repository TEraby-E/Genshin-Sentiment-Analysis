"""B 站公开视频与评论抓取。

默认使用 B 站公开 Web API，输出统一的 DataFrame 字段：
`Comment_Content` 是下游模型使用的文本列，其他字段保留视频和评论元数据。
可通过环境变量 `BILI_COOKIE` 提供登录态 cookie，提高接口可访问性与评论召回。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config

JsonTransport = Callable[[str, dict[str, str]], dict[str, Any]]


@dataclass
class VideoRecord:
    """搜索结果中的视频元数据。"""

    aid: int
    bvid: str
    title: str
    up: str = ""
    pubdate: int | None = None
    view: int | None = None
    like: int | None = None


@dataclass
class CommentRecord:
    """规范化后的 B 站评论。字段名兼容既有 `Comment_Content` 下游。"""

    Comment_Content: str
    bvid: str
    aid: int
    video_title: str
    up: str = ""
    rpid: int | None = None
    ctime: int | None = None
    like: int | None = None
    user: str = ""


def _strip_html(text: str) -> str:
    """搜索接口的 title 可能带 `<em>` 高亮标签，抓取阶段直接去掉。"""
    return (
        str(text)
        .replace("<em class=\"keyword\">", "")
        .replace("</em>", "")
        .replace("<em>", "")
    )


def _default_transport(url: str, headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - 用户主动抓取公开 API
        return json.loads(resp.read().decode("utf-8", errors="replace"))


class BiliClient:
    """极薄的 B 站公开 API 客户端，支持测试注入 fake transport。"""

    def __init__(
        self,
        *,
        cookie: str | None = None,
        cache_dir: str | Path | None = None,
        transport: JsonTransport | None = None,
        sleep_seconds: float = 0.4,
    ) -> None:
        self.cookie = cookie if cookie is not None else os.getenv("BILI_COOKIE", "")
        self.cache_dir = Path(cache_dir) if cache_dir else config.DATA_DIR / "cache" / "bili"
        self.transport = transport or _default_transport
        self.sleep_seconds = sleep_seconds

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        }
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    def _get_json(self, url: str, *, cache: bool = True) -> dict[str, Any]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.md5(url.encode("utf-8")).hexdigest()  # noqa: S324 - cache key only
        path = self.cache_dir / f"{key}.json"
        if cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        data = self.transport(url, self._headers())
        if cache:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return data

    def search_videos(
        self,
        keyword: str = "原神",
        *,
        page: int = 1,
        page_size: int = 20,
        order: str = "pubdate",
    ) -> list[VideoRecord]:
        """按关键词搜索近期视频，默认按发布时间排序。"""
        query = urllib.parse.urlencode(
            {
                "search_type": "video",
                "keyword": keyword,
                "page": page,
                "page_size": page_size,
                "order": order,
            }
        )
        url = f"https://api.bilibili.com/x/web-interface/search/type?{query}"
        raw = self._get_json(url)
        result = (raw.get("data") or {}).get("result") or []
        videos: list[VideoRecord] = []
        for item in result:
            aid = int(item.get("aid") or 0)
            bvid = str(item.get("bvid") or "")
            if not aid and not bvid:
                continue
            videos.append(
                VideoRecord(
                    aid=aid,
                    bvid=bvid,
                    title=_strip_html(str(item.get("title") or "")),
                    up=str(item.get("author") or item.get("up") or ""),
                    pubdate=int(item["pubdate"]) if item.get("pubdate") else None,
                    view=int(item["play"]) if str(item.get("play", "")).isdigit() else None,
                    like=int(item["like"]) if str(item.get("like", "")).isdigit() else None,
                )
            )
        return videos

    def resolve_aid(self, bvid: str) -> int:
        """评论接口需要 aid；搜索结果缺 aid 时用 bvid 查询视频详情补齐。"""
        query = urllib.parse.urlencode({"bvid": bvid})
        raw = self._get_json(f"https://api.bilibili.com/x/web-interface/view?{query}")
        aid = (raw.get("data") or {}).get("aid")
        return int(aid or 0)

    def fetch_comments(
        self,
        video: VideoRecord,
        *,
        max_comments: int = 100,
        page_size: int = 20,
    ) -> list[CommentRecord]:
        """抓取单个视频的楼层评论。"""
        aid = video.aid or self.resolve_aid(video.bvid)
        if not aid:
            return []

        comments: list[CommentRecord] = []
        next_page = 0
        while len(comments) < max_comments:
            query = urllib.parse.urlencode(
                {"type": 1, "oid": aid, "mode": 3, "ps": page_size, "next": next_page}
            )
            raw = self._get_json(f"https://api.bilibili.com/x/v2/reply/main?{query}")
            data = raw.get("data") or {}
            replies = data.get("replies") or []
            for item in replies:
                content = (item.get("content") or {}).get("message") or ""
                if not str(content).strip():
                    continue
                member = item.get("member") or {}
                comments.append(
                    CommentRecord(
                        Comment_Content=str(content).strip(),
                        bvid=video.bvid,
                        aid=aid,
                        video_title=video.title,
                        up=video.up,
                        rpid=int(item["rpid"]) if item.get("rpid") else None,
                        ctime=int(item["ctime"]) if item.get("ctime") else None,
                        like=int(item["like"]) if item.get("like") is not None else None,
                        user=str(member.get("uname") or ""),
                    )
                )
                if len(comments) >= max_comments:
                    break

            cursor = data.get("cursor") or {}
            if not replies or bool(cursor.get("is_end")):
                break
            next_page = int(cursor.get("next") or next_page + 1)
        return comments


def crawl_recent_comments(
    *,
    keyword: str = "原神",
    max_videos: int = 10,
    comments_per_video: int = 50,
    client: BiliClient | None = None,
) -> pd.DataFrame:
    """搜索近期视频并抓取评论，返回可直接送入模型的评论 DataFrame。"""
    client = client or BiliClient()
    videos = client.search_videos(keyword=keyword, page_size=max_videos)[:max_videos]
    rows: list[dict[str, Any]] = []
    for video in videos:
        for comment in client.fetch_comments(video, max_comments=comments_per_video):
            row = asdict(comment)
            if video.pubdate is not None:
                row["video_pubdate"] = video.pubdate
            if video.view is not None:
                row["video_view"] = video.view
            if video.like is not None:
                row["video_like"] = video.like
            rows.append(row)
    return pd.DataFrame(rows)
