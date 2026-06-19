"""命令行入口：跑一轮竞品 + 榜单监控，落地快照，并对比上一次榜单变化。

用法：
    uv sync --extra scrape
    uv run python scripts/monitor_run.py
    uv run python scripts/monitor_run.py --keywords 原神 鸣潮 绝区零 --interval 2
被风控/无网时自动降级到演示数据，流程不中断。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from src import monitor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="竞品 + 榜单监控")
    parser.add_argument("--keywords", nargs="*", default=None, help="竞品关键词（默认内置一组）")
    parser.add_argument("--interval", type=float, default=1.5, help="请求最小间隔秒数（限速）")
    args = parser.parse_args()

    client = monitor.BilibiliClient(min_interval=args.interval)

    print("=" * 60)
    print("榜单数据监控")
    print("=" * 60)
    ranking, live = monitor.fetch_ranking(client)
    tag = "实时抓取" if live else "演示数据(降级)"
    print(f"数据源：{tag}　全站排行榜 Top{len(ranking)}")
    print(ranking.head(10)[["rank", "title", "up", "view"]].to_string(index=False))

    # 与上一份榜单快照对比
    prev = monitor.list_snapshots("ranking")
    if prev:
        old = pd.read_csv(prev[-1])
        diff = monitor.diff_rankings(old, ranking)
        print(f"\n较上次（{prev[-1].name}）变化：")
        print(f"  新上榜 {len(diff['newcomers'])} 条，跌出 {diff['dropped_count']} 条")
        for m in diff["movements"][:5]:
            arrow = "↑" if m["change"] > 0 else "↓"
            print(f"  {arrow}{abs(m['change'])}  {m['title'][:30]}（{m['from']}→{m['to']}）")
    snap = monitor.save_snapshot(ranking, "ranking")
    print(f"快照已保存：{snap.name}")

    print("\n" + "=" * 60)
    print("竞品动态监测")
    print("=" * 60)
    comp, clive = monitor.track_competitors(args.keywords, client=client)
    print(f"数据源：{'实时抓取' if clive else '演示数据(降级)'}")
    print(comp.to_string(index=False))
    monitor.save_snapshot(comp, "competitors")


if __name__ == "__main__":
    main()
