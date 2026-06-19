"""A/B 实验评估体系：样本量/检验功效计算 + 双样本显著性检验。

本数据集是观察性的公开内容数据，不包含真实的随机分流实验记录，
所以这里实现的是一套通用、可复用的评估框架（这正是 JD 里"熟悉 A/B 实验评估体系"
要考察的能力本身），用模拟数据演示其正确性；应用到真实业务时，
只需要把模拟的转化人数换成真实埋点统计的转化人数即可直接复用。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def required_sample_size(
    baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> dict:
    """计算双样本比例检验所需的最小样本量（每组）。

    baseline_rate: 对照组的基线转化率（如点击率）
    mde: 最小可检测效应（绝对值，如 0.02 代表 2 个百分点的提升）
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    p1, p2 = baseline_rate, baseline_rate + mde
    p_bar = (p1 + p2) / 2

    term_alpha = z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
    term_power = z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    numerator = (term_alpha + term_power) ** 2
    n = numerator / (mde**2)
    return {
        "baseline_rate": baseline_rate,
        "mde": mde,
        "alpha": alpha,
        "power": power,
        "required_n_per_group": int(np.ceil(n)),
    }


def evaluate_ab_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> dict:
    """双样本比例 z 检验：判断实验组相对对照组的提升是否统计显著，并给出效应的置信区间。"""
    p1 = control_conversions / control_n
    p2 = treatment_conversions / treatment_n
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (p2 - p1) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(p1 * (1 - p1) / control_n + p2 * (1 - p2) / treatment_n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    return {
        "control_rate": round(p1, 4),
        "treatment_rate": round(p2, 4),
        "absolute_lift": round(diff, 4),
        "relative_lift_pct": round(diff / p1 * 100, 1) if p1 > 0 else None,
        "ci_95": (round(float(ci[0]), 4), round(float(ci[1]), 4)),
        "z": round(float(z), 3),
        "p_value": round(float(p_value), 4),
        "significant": p_value < alpha,
    }


def simulate_campaign_ab_demo(random_state: int = 42) -> dict:
    """方法演示：模拟"两套版本预热海报"在点击率上的 A/B 测试，串联功效计算与结果评估。

    场景设定贴近营销实际：对照组海报基线点击率 8%，预期新方案带来 2 个百分点的提升，
    先算出在该效应量下需要多大样本才能稳定检出，再用模拟的真实流量数据验证评估流程本身的正确性。
    """
    baseline_rate, mde = 0.08, 0.02
    sizing = required_sample_size(baseline_rate, mde)

    rng = np.random.default_rng(random_state)
    n = sizing["required_n_per_group"]
    control_conversions = int(rng.binomial(n, baseline_rate))
    treatment_conversions = int(rng.binomial(n, baseline_rate + mde))

    result = evaluate_ab_test(control_conversions, n, treatment_conversions, n)
    result["sizing"] = sizing
    return result
