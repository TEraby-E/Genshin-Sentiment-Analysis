"""可视化：生成六图分析面板。"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

matplotlib.rcParams["font.sans-serif"] = config.CJK_FONTS
matplotlib.rcParams["axes.unicode_minus"] = False


def make_dashboard(
    videos: pd.DataFrame,
    comments: pd.DataFrame,
    posts: pd.DataFrame,
    monthly: pd.DataFrame,
    save_path=None,
):
    """生成六图分析面板并保存。"""
    if save_path is None:
        save_path = config.OUTPUT_DIR / "genshin_analysis.png"

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")

    # 图1：各主题中位播放
    ax1 = fig.add_subplot(2, 3, 1)
    agg = videos.groupby("topic")["Amount_View"].median().sort_values()
    ax1.barh(range(len(agg)), agg.values / 1e4, color="#4ECDC4")
    ax1.set_yticks(range(len(agg)))
    ax1.set_yticklabels([str(t)[:22] for t in agg.index], fontsize=7)
    ax1.set_xlabel("中位播放量(万)")
    ax1.set_title("各主题视频热度", fontweight="bold")

    # 图2：播放量长尾分布
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(np.log10(videos["Amount_View"] + 1), bins=50, color="#FF6B6B", alpha=0.8)
    ax2.set_xlabel("log10(播放量)")
    ax2.set_ylabel("视频数")
    ax2.set_title("播放量长尾分布", fontweight="bold")

    # 图3：UGC 月度产出
    ax3 = fig.add_subplot(2, 3, 3)
    mv = videos.groupby(videos["Publish_Date"].dt.to_period("M").astype(str)).size()
    ax3.plot(range(len(mv)), mv.values, marker="o", color="#5B8FF9")
    ax3.set_xticks(range(0, len(mv), 2))
    ax3.set_xticklabels(mv.index[::2], rotation=45, fontsize=7)
    ax3.set_title("UGC 月度产出趋势", fontweight="bold")
    ax3.set_ylabel("视频数")

    # 图4：负面舆情月度趋势（核心）
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(range(len(monthly)), monthly["neg_rate"].values, marker="s", color="#E74C3C", lw=2)
    med = monthly["neg_rate"].median()
    ax4.axhline(med, ls="--", color="gray", label=f"基线 {med:.1f}%")
    ax4.fill_between(range(len(monthly)), monthly["neg_rate"].values, alpha=0.15, color="#E74C3C")
    ax4.set_xticks(range(0, len(monthly), 2))
    ax4.set_xticklabels(monthly.index[::2], rotation=45, fontsize=7)
    ax4.set_title("★负面舆情占比月度趋势(舆情预警)", fontweight="bold")
    ax4.set_ylabel("负面占比%")
    ax4.legend(fontsize=8)

    # 图5：评论情感/主题构成
    ax5 = fig.add_subplot(2, 3, 5)
    sent_label = {
        "Character Appreciation & Fandom Memes": "正面-角色喜爱",
        "Story Criticism & Operational Skepticism": "负面-剧情/运营批评",
        "Player Daily Life & Social Interaction": "正面-社交",
        "Community Controversy & Negative Sentiment": "负面-社区争议",
        "Game Mechanics & Optimization Feedback": "诉求-机制反馈",
        "Rewards & Resource Acquisition": "诉求-奖励资源",
    }
    cc = comments["cluster_name"].value_counts().head(6)
    labels = [sent_label.get(k, str(k)[:12]) for k in cc.index]
    colors = [
        "#2ECC71" if "正面" in lb else "#E74C3C" if "负面" in lb else "#F39C12"
        for lb in labels
    ]
    ax5.barh(range(len(cc)), cc.values / 1e3, color=colors)
    ax5.set_yticks(range(len(cc)))
    ax5.set_yticklabels(labels, fontsize=7)
    ax5.set_xlabel("评论数(千)")
    ax5.set_title("评论情感/主题构成", fontweight="bold")

    # 图6：内容价值象限（点赞率 vs 收藏率）
    ax6 = fig.add_subplot(2, 3, 6)
    vt = videos.groupby("topic").apply(
        lambda g: pd.Series(
            {
                "lr": (g["Amount_Like"] / g["Amount_View"]).median() * 100,
                "fr": (g["Amount_Favourite"] / g["Amount_View"]).median() * 100,
            }
        ),
        include_groups=False,
    )
    ax6.scatter(vt["lr"], vt["fr"], s=80, color="#9B59B6", alpha=0.7)
    for t, r in vt.iterrows():
        ax6.annotate(str(t)[:10], (r["lr"], r["fr"]), fontsize=6)
    ax6.set_xlabel("点赞率%")
    ax6.set_ylabel("收藏率%")
    ax6.set_title("内容价值象限(认同 vs 收藏)", fontweight="bold")

    plt.suptitle("原神 B 站玩家生态与舆情分析报告", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path
