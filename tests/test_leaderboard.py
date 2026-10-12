"""
tests/test_leaderboard.py -- unit tests for harness/leaderboard.py, with
hand-computable worked examples pinning down the Track A/B/C percentile
logic and the retention-ratio construction of Track C.
"""
import math

from harness.leaderboard import (
    MAX_BONUS_WEIGHT,
    TRACK_A_MAP_WEIGHT,
    TRACK_A_NDCG_WEIGHT,
    TRACK_A_WEIGHT,
    TRACK_B_WEIGHT,
    TRACK_C_WEIGHT,
    bonus_component,
    retention_ratio,
    track_a_component,
    track_b_component,
    track_c_component,
)


def test_track_a_lowest_and_highest_in_class():
    team_ndcg = {"low": 0.10, "mid": 0.30, "high": 0.50}
    team_map = {"low": 0.05, "mid": 0.15, "high": 0.25}
    result = track_a_component(team_ndcg, team_map)

    assert math.isclose(result["low"], 0.0, abs_tol=1e-9)
    assert math.isclose(result["high"], TRACK_A_WEIGHT, rel_tol=1e-9)
    expected_mid = TRACK_A_NDCG_WEIGHT * 0.5 + TRACK_A_MAP_WEIGHT * 0.5
    assert math.isclose(result["mid"], expected_mid, rel_tol=1e-9)


def test_track_a_narrow_raw_spread_still_uses_full_weight_range():
    # Same lesson as the earlier project's leaderboard fix: a tightly clustered
    # raw metric must still spread its BEST/WORST teams across the full
    # weight range under percentile ranking.
    team_ndcg = {"a": 0.1674, "b": 0.1990, "c": 0.2276}
    team_map = {"a": 0.0898, "b": 0.1072, "c": 0.1276}
    result = track_a_component(team_ndcg, team_map)

    assert math.isclose(result["a"], 0.0, abs_tol=1e-9)
    assert math.isclose(result["c"], TRACK_A_WEIGHT, rel_tol=1e-9)
    assert result["a"] < result["b"] < result["c"]


def test_track_b_single_metric_percentile():
    team_clean_ndcg = {"low": 0.2, "mid": 0.4, "high": 0.6}
    result = track_b_component(team_clean_ndcg)

    assert math.isclose(result["low"], 0.0, abs_tol=1e-9)
    assert math.isclose(result["high"], TRACK_B_WEIGHT, rel_tol=1e-9)
    assert math.isclose(result["mid"], TRACK_B_WEIGHT * 0.5, rel_tol=1e-9)


def test_retention_ratio_full_retention():
    assert math.isclose(retention_ratio(0.5, [0.5, 0.5, 0.5]), 1.0, rel_tol=1e-9)


def test_retention_ratio_partial_degradation():
    # Clean nDCG@10 = 0.8; noisy conditions average 0.4 -> half retained.
    assert math.isclose(retention_ratio(0.8, [0.5, 0.3]), 0.5, rel_tol=1e-9)


def test_retention_ratio_zero_clean_score_is_zero_not_a_crash():
    assert retention_ratio(0.0, [0.1, 0.2]) == 0.0


def test_retention_ratio_no_noisy_values_is_zero():
    assert retention_ratio(0.5, []) == 0.0


def test_retention_ratio_is_capped_at_one():
    # A lucky noisy-condition fluctuation on a tiny query set should not
    # read as "better than fully retained".
    assert retention_ratio(0.4, [0.9]) == 1.0


def test_track_c_rewards_the_most_robust_team_even_if_its_raw_accuracy_is_lower():
    # "sturdy" has a lower clean score than "fragile" but barely degrades
    # under noise; "fragile" has a higher clean score but collapses.
    # Track C must rank "sturdy" above "fragile" -- that is the entire
    # point of scoring retention rather than raw noisy-condition accuracy.
    sturdy_retention = retention_ratio(clean_ndcg=0.4, noisy_ndcg_values=[0.38, 0.36])
    fragile_retention = retention_ratio(clean_ndcg=0.9, noisy_ndcg_values=[0.10, 0.05])
    assert sturdy_retention > fragile_retention

    team_retention = {"sturdy": sturdy_retention, "fragile": fragile_retention}
    result = track_c_component(team_retention)
    assert result["sturdy"] > result["fragile"]
    assert math.isclose(result["sturdy"], TRACK_C_WEIGHT, rel_tol=1e-9)
    assert math.isclose(result["fragile"], 0.0, abs_tol=1e-9)


def test_bonus_component_bounds_and_absent_teams_get_nothing():
    team_bonus_raw = {"attacker": 0.9, "defender": 0.1}
    result = bonus_component(team_bonus_raw)
    assert math.isclose(result["attacker"], MAX_BONUS_WEIGHT, rel_tol=1e-9)
    assert math.isclose(result["defender"], 0.0, abs_tol=1e-9)
    # A team with no entry in team_bonus_raw (didn't submit an
    # adversarial_set.json) simply is not a key here -- callers must
    # treat a missing key as +0.0, never a penalty.
    assert "did_not_participate" not in result


def test_bonus_component_empty_input():
    assert bonus_component({}) == {}


def test_single_team_class_gets_full_marks_on_every_track():
    # Matches instructor-tools/grading_rubric.py's treatment of a lone
    # baseline-beater: nothing to be ranked against, so it gets the
    # ceiling on every track.
    assert track_a_component({"solo": 0.3}, {"solo": 0.1})["solo"] == TRACK_A_WEIGHT
    assert track_b_component({"solo": 0.3})["solo"] == TRACK_B_WEIGHT
    assert track_c_component({"solo": 0.9})["solo"] == TRACK_C_WEIGHT
