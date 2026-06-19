from __future__ import annotations

from src import ab_test


def test_required_sample_size_decreases_with_larger_mde():
    small_mde = ab_test.required_sample_size(0.08, 0.01)
    large_mde = ab_test.required_sample_size(0.08, 0.05)
    assert large_mde["required_n_per_group"] < small_mde["required_n_per_group"]


def test_evaluate_ab_test_detects_no_difference():
    result = ab_test.evaluate_ab_test(100, 1000, 100, 1000)
    assert result["absolute_lift"] == 0.0
    assert not result["significant"]


def test_evaluate_ab_test_detects_large_difference():
    result = ab_test.evaluate_ab_test(80, 1000, 200, 1000)
    assert result["significant"]
    assert result["absolute_lift"] > 0


def test_simulate_campaign_ab_demo_runs_end_to_end():
    result = ab_test.simulate_campaign_ab_demo()
    assert "sizing" in result
    assert result["sizing"]["required_n_per_group"] > 0
    assert 0 <= result["p_value"] <= 1
