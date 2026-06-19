"""竞品动态 & 榜单数据监控逻辑。

把抓取到的原始数据整理成可对比、可追踪的结构化快照，为业务决策提供数据参考：
- 榜单监控：抓全站排行榜 → 落地带时间戳的快照 → 与上次快照对比，标出新上榜/排名变化；
- 竞品监测：对一组竞品关键词搜索 → 聚合各竞品的内容产出量、热度、Top 作品。

所有抓取失败（无网络 / 被风控）时优雅降级到内置演示数据，保证界面/脚本始终可用。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .scraper import BilibiliClient, ScrapeBlockedError

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = config.DATA_DIR / "snapshots"

# 原神在 B 站的主要竞品/对标二游（用于竞品动态搜索）
DEFAULT_COMPETITORS = ["原神", "崩坏：星穹铁道", "绝区零", "鸣潮", "明日方舟"]

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_title(title: str) -> str:
    return _TAG_RE.sub("", str(title)).replace("&amp;", "&").strip()


# ---- 榜单监控 ----


def fetch_ranking(client: BilibiliClient | None = None) -> tuple[pd.DataFrame, bool]:
    """抓取全站排行榜，整理为 DataFrame。返回 (df, is_live)；失败则回退演示数据。"""
    client = client or BilibiliClient()
    try:
        raw = client.get_ranking()
        rows = [
            {
                "rank": i + 1,
                "title": _clean_title(v.get("title", "")),
                "up": v.get("owner", {}).get("name", ""),
                "view": v.get("stat", {}).get("view", 0),
                "like": v.get("stat", {}).get("like", 0),
                "coin": v.get("stat", {}).get("coin", 0),
                "bvid": v.get("bvid", ""),
            }
            for i, v in enumerate(raw)
        ]
        if rows:
            return pd.DataFrame(rows), True
        raise ScrapeBlockedError("排行榜为空")
    except (ScrapeBlockedError, Exception) as e:  # noqa: BLE001
        logger.warning("榜单抓取失败，回退演示数据：%s", e)
        return _demo_ranking(), False


def diff_rankings(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """对比两份榜单快照，识别新上榜、跌出、排名变化。"""
    old_map = dict(zip(old["bvid"], old["rank"]))
    new_map = dict(zip(new["bvid"], new["rank"]))
    title_map = dict(zip(new["bvid"], new["title"]))

    newcomers = [
        {"title": title_map[b], "rank": new_map[b]} for b in new_map if b not in old_map
    ]
    dropped = [b for b in old_map if b not in new_map]
    movements = []
    for b in new_map:
        if b in old_map:
            delta = old_map[b] - new_map[b]  # 正数=排名上升
            if delta != 0:
                movements.append(
                    {"title": title_map[b], "from": old_map[b], "to": new_map[b], "change": delta}
                )
    movements.sort(key=lambda x: -abs(x["change"]))
    return {
        "newcomers": sorted(newcomers, key=lambda x: x["rank"]),
        "dropped_count": len(dropped),
        "movements": movements,
    }


def save_snapshot(df: pd.DataFrame, kind: str) -> Path:
    """把快照写入 data/snapshots/{kind}_{timestamp}.csv，用于后续时序对比。"""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"{kind}_{ts}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def list_snapshots(kind: str) -> list[Path]:
    """按时间排序列出某类快照文件（最新在后）。"""
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob(f"{kind}_*.csv"))


# ---- 竞品监测 ----


def track_competitors(
    keywords: list[str] | None = None, client: BilibiliClient | None = None, top_n: int = 3
) -> tuple[pd.DataFrame, bool]:
    """对每个竞品关键词搜索视频，聚合内容产出与热度。返回 (汇总df, is_live)。"""
    keywords = keywords or DEFAULT_COMPETITORS
    client = client or BilibiliClient()
    rows = []
    live = True
    for kw in keywords:
        try:
            results = client.search_videos(kw)
        except (ScrapeBlockedError, Exception) as e:  # noqa: BLE001
            logger.warning("竞品 %s 搜索失败：%s", kw, e)
            live = False
            results = []
        if not results:
            continue
        views = [int(r.get("play", 0) or 0) for r in results]
        top = sorted(results, key=lambda r: int(r.get("play", 0) or 0), reverse=True)[:top_n]
        rows.append(
            {
                "竞品": kw,
                "搜索命中视频数": len(results),
                "总播放": int(np.sum(views)),
                "中位播放": int(np.median(views)) if views else 0,
                "Top作品": " / ".join(_clean_title(t.get("title", "")) for t in top),
            }
        )

    if not rows:
        logger.warning("竞品搜索全部失败，回退演示数据")
        return _demo_competitors(keywords), False
    return pd.DataFrame(rows).sort_values("总播放", ascending=False), live


# ---- 演示数据兜底 ----


def _demo_ranking(seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    titles = [
        "《原神》至冬生态短片", "崩坏：星穹铁道 新版本PV", "绝区零 角色演示",
        "鸣潮 主线剧情解读", "原神 五星角色抽卡实录", "明日方舟 周年庆活动",
        "二游音乐MMD合集", "原神 大世界探索攻略", "星铁 模拟宇宙速通", "鸣潮 BOSS无伤教学",
    ]
    n = len(titles)
    views = sorted(rng.integers(80_0000, 600_0000, n), reverse=True)
    return pd.DataFrame(
        {
            "rank": range(1, n + 1),
            "title": titles,
            "up": [f"UP主_{i}" for i in range(n)],
            "view": views,
            "like": [int(v * rng.uniform(0.03, 0.08)) for v in views],
            "coin": [int(v * rng.uniform(0.01, 0.03)) for v in views],
            "bvid": [f"BV_demo_{i}" for i in range(n)],
        }
    )


def _demo_competitors(keywords: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for kw in keywords:
        cnt = int(rng.integers(15, 30))
        views = rng.integers(5_0000, 300_0000, cnt)
        rows.append(
            {
                "竞品": kw,
                "搜索命中视频数": cnt,
                "总播放": int(views.sum()),
                "中位播放": int(np.median(views)),
                "Top作品": f"{kw} 热门作品示例 / {kw} 新版本前瞻 / {kw} 角色解析",
            }
        )
    return pd.DataFrame(rows).sort_values("总播放", ascending=False)
