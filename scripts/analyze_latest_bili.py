"""抓取最新 B 站评论并用现有路由模型生成舆情报告。

示例：
    .venv/Scripts/python.exe scripts/analyze_latest_bili.py --keyword 原神
    python scripts/analyze_latest_bili.py --keyword 原神 --videos 5 --comments-per-video 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.live_sentiment import analyze_recent_comments, report_to_dict
from src.scraper.bilibili import crawl_recent_comments


def main() -> int:
    parser = argparse.ArgumentParser(description="最新 B 站评论舆情分析")
    parser.add_argument("--keyword", default="原神", help="B 站搜索关键词")
    parser.add_argument("--videos", type=int, default=8, help="抓取多少个近期视频")
    parser.add_argument(
        "--comments-per-video", type=int, default=40, help="每个视频最多抓取多少条评论"
    )
    parser.add_argument("--out-md", default="outputs/latest_bili_sentiment.md")
    parser.add_argument("--out-csv", default="outputs/latest_bili_tagged.csv")
    parser.add_argument("--out-json", default="outputs/latest_bili_sentiment.json")
    args = parser.parse_args()

    comments = crawl_recent_comments(
        keyword=args.keyword,
        max_videos=args.videos,
        comments_per_video=args.comments_per_video,
    )
    if comments.empty:
        raise SystemExit("没有抓到评论：可能是接口风控、关键词无结果，或需要设置 BILI_COOKIE。")

    tagged, report = analyze_recent_comments(comments)

    md_path = Path(args.out_md)
    csv_path = Path(args.out_csv)
    json_path = Path(args.out_json)
    for path in (md_path, csv_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(report.to_markdown(), encoding="utf-8")
    tagged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(report.to_markdown())
    print(f"\n已写入：{md_path} / {csv_path} / {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
