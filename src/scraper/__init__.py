"""B 站公开数据抓取（竞品动态、榜单监控）。

只抓取公开、非个人数据（排行榜、搜索结果、公开视频统计），用于市场/竞品分析。
内置限速、本地缓存与优雅降级，避免高频请求；不绕过登录墙、不抓私密数据。
"""

from .bilibili import BilibiliClient, ScrapeBlockedError

__all__ = ["BilibiliClient", "ScrapeBlockedError"]
