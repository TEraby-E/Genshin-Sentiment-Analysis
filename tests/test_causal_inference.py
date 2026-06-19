from __future__ import annotations

import pandas as pd
import pytest

from src import causal_inference


def test_matched_control_effect_returns_expected_keys(posts_df):
    result = causal_inference.matched_control_effect(
        posts_df, metric="Like_Count", n_permutations=200
    )
    expected_keys = {"observed_effect", "relative_lift_pct", "p_value", "significant", "n_treated"}
    assert expected_keys <= set(result)
    assert 0 <= result["p_value"] <= 1
    assert result["n_treated"] == posts_df["Collaboration_Flag"].sum()


def test_matched_control_effect_raises_without_treated_posts(posts_df):
    df = posts_df.copy()
    df["Collaboration_Flag"] = False
    with pytest.raises(ValueError, match="没有标记为跨界联动"):
        causal_inference.matched_control_effect(df)


def test_matched_control_effect_detects_strong_injected_lift():
    rng_df = pd.DataFrame(
        {
            "Publish_Date": pd.date_range("2024-01-01", periods=40, freq="3D"),
            "Like_Count": [1000] * 40,
            "Collaboration_Flag": [i % 10 == 0 for i in range(40)],
        }
    )
    rng_df.loc[rng_df["Collaboration_Flag"], "Like_Count"] = 5000
    result = causal_inference.matched_control_effect(
        rng_df, metric="Like_Count", n_permutations=500
    )
    assert result["relative_lift_pct"] > 100
    assert result["significant"]
