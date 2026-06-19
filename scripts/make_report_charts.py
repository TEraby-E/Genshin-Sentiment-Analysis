"""为 HR 项目展示书生成简单易懂的单图表（区别于 outputs/ 下面向技术读者的六联图面板）。

每张图只讲一件事，配色统一，字号更大，供非技术读者阅读。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.pyplot as plt

from src import (
    ab_test,
    analysis,
    aspect_sentiment,
    causal_inference,
    config,
    data_loader,
    lifecycle,
    monitor,
)

matplotlib.rcParams["font.sans-serif"] = config.CJK_FONTS
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 13

IMG_DIR = Path(__file__).resolve().parents[1] / "Business-report-1" / "image"
IMG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#000060"
RED = "#E74C3C"
GREEN = "#2ECC71"
GRAY = "#888888"
TEAL = "#4ECDC4"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}")


def chart_data_scale(videos, comments, posts):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["视频", "评论", "官方帖子"]
    values = [len(videos), len(comments), len(posts)]
    bars = ax.bar(labels, values, color=[NAVY, TEAL, GRAY], width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("数量（对数刻度）")
    ax.set_title("数据规模总览", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "data_scale.png")


def chart_topic_ecosystem(videos):
    eco = analysis.ecosystem_by_topic(videos).head(8).sort_values("median_view")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(range(len(eco)), eco["median_view"] / 1e4, color=NAVY)
    ax.set_yticks(range(len(eco)))
    ax.set_yticklabels([str(t)[:26] for t in eco.index], fontsize=12)
    ax.set_xlabel("单视频中位播放量（万）")
    ax.set_title("玩家最关注的内容主题 Top 8", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "topic_ecosystem.png")


def chart_sentiment_trend(comments, posts):
    monthly = analysis.sentiment_trend(comments, posts)
    alert = analysis.alert_threshold(monthly)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(monthly))
    ax.plot(x, monthly["neg_rate"], marker="o", color=RED, lw=2.2, label="月度负面占比")
    ax.axhline(alert["baseline"], ls="--", color=GRAY, label=f"基线 {alert['baseline']}%")
    ax.axhline(alert["threshold"], ls=":", color=NAVY, label=f"预警线 {alert['threshold']}%")
    ax.fill_between(x, monthly["neg_rate"], alpha=0.12, color=RED)
    peak_idx = list(monthly.index).index(alert["peak_month"])
    ax.annotate(
        f"峰值 {alert['peak_month']}\n{alert['peak_rate']}%（统计显著, p<0.05）",
        xy=(peak_idx, alert["peak_rate"]),
        xytext=(peak_idx, alert["peak_rate"] + 4),
        ha="center", fontsize=11, fontweight="bold", color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY),
    )
    ax.set_xticks(list(x)[::2])
    ax.set_xticklabels(list(monthly.index)[::2], rotation=45, fontsize=10)
    ax.set_ylabel("负面评论占比 (%)")
    ax.set_title("舆情负面情绪月度走势与预警", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "sentiment_trend.png")


def chart_aspect_breakdown(comments):
    df = comments.copy()
    df["is_neg"] = df["cluster_name"].isin(config.NEGATIVE_CLUSTERS)
    breakdown = aspect_sentiment.aspect_breakdown(df).sort_values("mentions")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [RED if r > breakdown["neg_rate"].median() else TEAL for r in breakdown["neg_rate"]]
    bars = ax.barh(range(len(breakdown)), breakdown["mentions"], color=colors)
    ax.set_yticks(range(len(breakdown)))
    ax.set_yticklabels(breakdown.index, fontsize=12)
    for b, rate in zip(bars, breakdown["neg_rate"]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  负面占比 {rate}%",
                va="center", fontsize=10, color=GRAY)
    ax.set_xlabel("提及评论数")
    ax.set_title("玩家具体在吐槽什么方面", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "aspect_breakdown.png")


def chart_hit_prediction(videos):
    result = analysis.hit_prediction(videos)
    importance = result["feature_importance"]
    label_map = {
        "TimeInSeconds": "视频时长", "title_len": "标题长度", "hour": "发布时段",
        "Topic_Cluster": "内容主题", "dayofweek": "发布星期", "len_type_enc": "视频类型",
    }
    items = sorted(importance.items(), key=lambda kv: kv[1])
    labels = [label_map.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = [NAVY if lb in ("发布时段", "发布星期") else TEAL for lb in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("特征重要性")
    ax.set_title(f"爆款预测：什么因素更重要？(AUC={result['auc']})", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "hit_prediction.png")


def chart_creator_lifecycle(videos):
    creators = lifecycle.creator_lifecycle(videos)
    summary = lifecycle.creator_stage_summary(creators)
    order = ["单发", "成长期", "稳定期", "沉寂期"]
    summary = summary.reindex(order).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    colors = [GRAY, TEAL, NAVY, RED]

    axes[0].bar(summary.index, summary["creator_count"], color=colors)
    axes[0].set_title("各阶段创作者数量", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("创作者数")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(summary.index, summary["median_view_per_video"] / 1e4, color=colors)
    axes[1].set_title("各阶段单视频中位播放（万）", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("中位播放量（万）")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "创作者生命周期分群：沉寂期创作者历史表现不输稳定期", fontsize=14, fontweight="bold"
    )
    save(fig, "creator_lifecycle.png")


def chart_causal_inference(posts):
    result = causal_inference.matched_control_effect(posts, metric="Like_Count")
    treated_mean = posts.loc[posts["Collaboration_Flag"], "Like_Count"].mean()
    control_mean = treated_mean / (1 + result["relative_lift_pct"] / 100)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(["同月普通帖子\n(对照组)", "跨界联动帖子\n(处理组)"],
                   [control_mean, treated_mean], color=[GRAY, NAVY], width=0.5)
    for b, v in zip(bars, [control_mean, treated_mean]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("平均点赞数")
    ax.set_title(
        f"跨界联动效果：点赞数提升 {result['relative_lift_pct']}%\n"
        f"(n={result['n_treated']}, p={result['p_value']}, "
        f"{'显著' if result['significant'] else '不显著'})",
        fontsize=13, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "causal_inference.png")


def chart_ab_test_demo():
    result = ab_test.simulate_campaign_ab_demo()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    rates = [result["control_rate"] * 100, result["treatment_rate"] * 100]
    bars = ax.bar(["对照组\n(原方案)", "实验组\n(新方案)"], rates, color=[GRAY, TEAL], width=0.5)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("点击率 (%)")
    sig_desc = "显著" if result["significant"] else "不显著"
    ax.set_title(
        f"A/B 实验评估框架演示（模拟数据）\n"
        f"样本量 {result['sizing']['required_n_per_group']:,}/组, "
        f"绝对提升 {result['absolute_lift']*100:.2f}pp, p={result['p_value']} → {sig_desc}",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "ab_test_demo.png")


def chart_competitor_monitor():
    """竞品动态监测：各竞品在 B 站的内容产出热度对比（实时抓取，失败则降级演示数据）。"""
    df, live = monitor.track_competitors()
    df = df.sort_values("总播放")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = [NAVY if k == "原神" else TEAL for k in df["竞品"]]

    axes[0].barh(range(len(df)), df["总播放"] / 1e4, color=colors)
    axes[0].set_yticks(range(len(df)))
    axes[0].set_yticklabels(df["竞品"], fontsize=12)
    axes[0].set_xlabel("搜索结果总播放（万）")
    axes[0].set_title("各竞品内容总热度", fontsize=13, fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].barh(range(len(df)), df["搜索命中视频数"], color=colors)
    axes[1].set_yticks(range(len(df)))
    axes[1].set_yticklabels(df["竞品"], fontsize=12)
    axes[1].set_xlabel("搜索命中视频数")
    axes[1].set_title("各竞品内容产出量", fontsize=13, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    src = "实时抓取" if live else "演示数据"
    fig.suptitle(f"竞品动态监测（数据源：{src}）", fontsize=14, fontweight="bold")
    save(fig, "competitor_monitor.png")


def main():
    videos, comments, posts = data_loader.load_all()
    chart_data_scale(videos, comments, posts)
    chart_topic_ecosystem(videos)
    chart_sentiment_trend(comments, posts)
    chart_aspect_breakdown(comments)
    chart_hit_prediction(videos)
    chart_creator_lifecycle(videos)
    chart_causal_inference(posts)
    chart_ab_test_demo()
    chart_competitor_monitor()


if __name__ == "__main__":
    main()
