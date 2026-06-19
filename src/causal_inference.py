"""跨界联动效果的因果推断：跨界联动（Collaboration_Flag）是市场策略中的核心场景之一，
这里回答"联动公告相比同期普通公告，互动效果有没有真实的增量"，而不是只看联动帖子的绝对数字
（绝对数字会被发布时间、版本节奏等混杂因素干扰）。

数据集中联动帖子样本量极小（n=6），不足以支撑严格的双重差分面板回归。
这里用更适合小样本的设计：按月匹配同期普通帖子作为对照组，用置换检验（permutation test）
估计处理效应，而不依赖大样本下才成立的正态近似——这是小样本因果推断里更稳健的做法。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def matched_control_effect(
    posts: pd.DataFrame,
    metric: str = "Like_Count",
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict:
    """估计跨界联动帖子相对同月普通帖子在某互动指标上的处理效应。

    匹配设计：每个联动帖子的对照组是同一个公历月内的所有普通帖子，
    用"同月"控制版本节奏、活动密度等随时间变化的混杂因素。
    """
    df = posts.dropna(subset=["Publish_Date", metric]).copy()
    df["ym"] = pd.to_datetime(df["Publish_Date"]).dt.to_period("M")

    treated = df[df["Collaboration_Flag"]]
    if len(treated) == 0:
        raise ValueError("数据中没有标记为跨界联动的帖子")

    rng = np.random.default_rng(random_state)
    observed_diffs = []
    control_means = []

    for _, row in treated.iterrows():
        same_month_control = df[(df["ym"] == row["ym"]) & (~df["Collaboration_Flag"])][metric]
        if len(same_month_control) == 0:
            continue
        observed_diffs.append(row[metric] - same_month_control.mean())
        control_means.append(same_month_control.mean())

    if not observed_diffs:
        raise ValueError("没有找到任何同月对照组，无法估计处理效应")

    observed_effect = float(np.mean(observed_diffs))
    baseline_mean = float(np.mean(control_means))
    n_treated = len(treated)

    # 置换检验：每次随机挑选等量的"伪处理组"，按同样的同月匹配逻辑重新估计效应，
    # 用这个效应的经验分布回答"观测到的效应有多大概率只是随机抽样造成的"，
    # 而不是套用大样本下才成立的正态近似（处理组只有 n_treated 个事件，正态近似并不可靠）。
    all_index = df.index.to_numpy()
    permuted_effects = np.empty(n_permutations)
    for i in range(n_permutations):
        pseudo_treated_idx = rng.choice(all_index, size=n_treated, replace=False)
        diffs = []
        for idx in pseudo_treated_idx:
            row = df.loc[idx]
            same_month = df[(df["ym"] == row["ym"]) & (~df.index.isin(pseudo_treated_idx))][metric]
            if len(same_month) == 0:
                continue
            diffs.append(row[metric] - same_month.mean())
        permuted_effects[i] = np.mean(diffs) if diffs else 0.0

    p_value = float(np.mean(np.abs(permuted_effects) >= abs(observed_effect)))

    return {
        "metric": metric,
        "n_treated": int(n_treated),
        "n_matched": len(observed_diffs),
        "observed_effect": round(observed_effect, 1),
        "relative_lift_pct": round(observed_effect / baseline_mean * 100, 1),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "caveat": (
            f"处理组样本量仅 n={n_treated}，置信区间天然较宽，"
            "结论需结合更大样本或A/B实验验证"
        ),
    }
