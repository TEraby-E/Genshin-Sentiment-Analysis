"""统计显著性检验：原版预警阈值是经验倍数（1.5×基线），缺乏统计依据。
这里补充双比例 z 检验，回答"某月负面占比的上升是否显著，而非样本波动"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def two_proportion_ztest(neg_a: int, total_a: int, neg_b: int, total_b: int) -> dict:
    """检验两个月份的负面占比是否存在显著差异（双侧 z 检验）。"""
    p_a, p_b = neg_a / total_a, neg_b / total_b
    p_pool = (neg_a + neg_b) / (total_a + total_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))
    z = (p_a - p_b) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "rate_a": round(p_a * 100, 2),
        "rate_b": round(p_b * 100, 2),
        "z": round(float(z), 3),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
    }


def peak_vs_baseline_significance(monthly: pd.DataFrame) -> dict:
    """检验舆情峰值月份相对基线月份是否存在统计显著的负面占比上升。

    基线取负面占比最接近中位数的月份，而非简单用中位数本身（中位数不对应具体样本量）。
    """
    median_rate = monthly["neg_rate"].median()
    baseline_month = (monthly["neg_rate"] - median_rate).abs().idxmin()
    peak_month = monthly["neg_rate"].idxmax()

    base = monthly.loc[baseline_month]
    peak = monthly.loc[peak_month]
    result = two_proportion_ztest(
        int(peak["neg"]), int(peak["total"]), int(base["neg"]), int(base["total"])
    )
    result["baseline_month"] = baseline_month
    result["peak_month"] = peak_month
    return result


def rolling_zscore_alert(
    monthly: pd.DataFrame, window: int = 3, z_threshold: float = 2.0
) -> pd.DataFrame:
    """用滚动窗口 z-score 替代固定倍数阈值：能适应舆情基线本身的缓慢漂移。

    固定倍数（如 1.5×全期中位数）在版本初期/后期基线本身变化时会漏报或误报，
    滚动窗口把"异常"定义为相对近期波动的偏离，而不是相对全期一个静态数字的偏离。
    """
    out = monthly.copy()
    # shift(1) 排除当月自身，否则异常月会把自己拉进基线，稀释自己的 z-score
    prior = out["neg_rate"].shift(1)
    roll_mean = prior.rolling(window, min_periods=2).mean()
    roll_std = prior.rolling(window, min_periods=2).std()
    out["zscore"] = (out["neg_rate"] - roll_mean) / roll_std.replace(0, np.nan)
    out["is_anomaly"] = out["zscore"] > z_threshold
    return out
