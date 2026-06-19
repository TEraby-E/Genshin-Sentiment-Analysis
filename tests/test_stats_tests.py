from __future__ import annotations

import pandas as pd

from src import stats_tests


def test_two_proportion_ztest_identical_rates_not_significant():
    result = stats_tests.two_proportion_ztest(50, 500, 50, 500)
    assert result["z"] == 0.0
    assert not result["significant"]


def test_two_proportion_ztest_large_gap_is_significant():
    result = stats_tests.two_proportion_ztest(300, 1000, 100, 1000)
    assert result["significant"]
    assert result["p_value"] < 0.05


def test_peak_vs_baseline_significance_picks_extremes():
    monthly = pd.DataFrame(
        {
            "total": [1000, 1000, 1000, 1000],
            "neg": [100, 105, 95, 400],
        },
        index=["2024-01", "2024-02", "2024-03", "2024-04"],
    )
    monthly["neg_rate"] = monthly["neg"] / monthly["total"] * 100
    result = stats_tests.peak_vs_baseline_significance(monthly)
    assert result["peak_month"] == "2024-04"
    assert result["significant"]


def test_rolling_zscore_alert_flags_spike():
    monthly = pd.DataFrame(
        {
            "neg_rate": [10.0, 10.5, 9.8, 10.2, 35.0],
            "total": [1000] * 5,
            "neg": [100, 105, 98, 102, 350],
        },
        index=[f"2024-0{i}" for i in range(1, 6)],
    )
    out = stats_tests.rolling_zscore_alert(monthly, window=3, z_threshold=2.0)
    assert out.loc["2024-05", "is_anomaly"]
    assert not out.loc["2024-02", "is_anomaly"]
