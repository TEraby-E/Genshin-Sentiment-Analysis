"""B 站数据抓取适配层。

爬虫模块只负责把外部接口返回规范化成项目内部表结构；舆情打标与总结放在
`src.live_sentiment`，避免抓取逻辑和模型逻辑耦合。
"""

from __future__ import annotations

from .bilibili import BiliClient, CommentRecord, VideoRecord, crawl_recent_comments

__all__ = ["BiliClient", "CommentRecord", "VideoRecord", "crawl_recent_comments"]
