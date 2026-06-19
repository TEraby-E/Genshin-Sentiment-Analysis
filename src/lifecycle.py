"""创作者与内容主题生命周期分析。

数据集是公开内容数据（视频/评论/帖子），没有第一方的玩家账号行为流水，
因此无法做严格意义上的"玩家"生命周期分群。这里用能拿到的两个最接近的代理维度：

1. UP 主（创作者）生命周期：基于发布频次与活跃时长，划分新晋/成长/稳定/沉寂四个阶段，
   对应市场策略中"哪类创作者值得长期合作、哪类只适合一次性投放"的判断。
2. 内容主题生命周期：基于月度产出量的近期变化率，判断主题处于增长/平稳/衰退，
   对应"内容偏好"维度——告诉运营该把资源往哪类主题倾斜。

两者都是观察性的描述统计代理指标，不是个体玩家的真实生命周期，这一点在报告中需要明确说明。
"""

from __future__ import annotations

import pandas as pd


def creator_lifecycle(videos: pd.DataFrame, recent_days: int = 60) -> pd.DataFrame:
    """按 UP 主聚合，划分生命周期阶段。

    阶段定义：
    - 单发：只发过一条视频，无法判断是否会持续产出；
    - 成长期：发布数 >= 2 且最近一条视频在 recent_days 内，仍在活跃产出；
    - 沉寂期：曾发布 >= 2 条，但最近一条视频超过 recent_days 未更新；
    - 稳定期：发布数 >= 5 且最近仍活跃，是可持续合作的核心创作者。
    """
    df = videos.dropna(subset=["Publish_Date"]).copy()
    max_date = df["Publish_Date"].max()

    agg = df.groupby("Author").agg(
        n_videos=("Amount_View", "size"),
        first_post=("Publish_Date", "min"),
        last_post=("Publish_Date", "max"),
        median_view=("Amount_View", "median"),
        median_like=("Amount_Like", "median"),
    )
    agg["days_since_last"] = (max_date - agg["last_post"]).dt.days
    agg["active_span_days"] = (agg["last_post"] - agg["first_post"]).dt.days

    def stage(row):
        if row["n_videos"] == 1:
            return "单发"
        is_recent = row["days_since_last"] <= recent_days
        if row["n_videos"] >= 5 and is_recent:
            return "稳定期"
        if is_recent:
            return "成长期"
        return "沉寂期"

    agg["stage"] = agg.apply(stage, axis=1)
    return agg


def creator_stage_summary(creator_df: pd.DataFrame) -> pd.DataFrame:
    """各生命周期阶段的创作者数量与产出表现，用于判断资源该投向哪个阶段。"""
    return creator_df.groupby("stage").agg(
        creator_count=("n_videos", "size"),
        total_videos=("n_videos", "sum"),
        median_view_per_video=("median_view", "median"),
    ).sort_values("total_videos", ascending=False)


def topic_lifecycle(videos: pd.DataFrame, recent_months: int = 3) -> pd.DataFrame:
    """按主题计算最近 N 个月相对前 N 个月的产出增长率，划分增长/平稳/衰退。

    用产出量（视频数）而非播放量做生命周期判断，因为播放量会被算法推荐策略和爆款波动干扰，
    产出量更直接反映创作者"是否还在持续生产这类内容"——这才是生命周期的本质信号。
    """
    df = videos.dropna(subset=["Publish_Date", "topic"]).copy()
    df["ym"] = df["Publish_Date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    if len(months) < 2 * recent_months:
        recent_months = max(1, len(months) // 2)

    recent_window = set(months[-recent_months:])
    prior_window = set(months[-2 * recent_months : -recent_months])

    counts = df.groupby(["topic", "ym"]).size().unstack(fill_value=0)
    recent = counts[[m for m in recent_window if m in counts.columns]].sum(axis=1)
    prior = counts[[m for m in prior_window if m in counts.columns]].sum(axis=1)

    out = pd.DataFrame({"recent_count": recent, "prior_count": prior})
    prior_safe = out["prior_count"].replace(0, pd.NA)
    out["growth_rate"] = ((out["recent_count"] - out["prior_count"]) / prior_safe * 100).round(1)

    def label(rate):
        if pd.isna(rate):
            return "新兴/数据不足"
        if rate > 15:
            return "增长期"
        if rate < -15:
            return "衰退期"
        return "平稳期"

    out["stage"] = out["growth_rate"].map(label)
    return out.sort_values("recent_count", ascending=False)
