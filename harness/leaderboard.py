"""
harness/leaderboard.py -- combine raw metrics into the weighted
leaderboard score defined in the spec (Section 7):

    Track A (base LM accuracy)        30%
  + Track B (clean feedback accuracy) 25%
  + Track C (drift robustness)        35%
  + Bonus (adversarial round-robin)   up to +10%, additive only

As in the earlier project, every track above is computed relative to the class
(percentile rank), not from raw metric values -- see the earlier project's
harness/leaderboard.py module docstring, "Why accuracy is class-relative
too", which applies unchanged here: a fixed weight only translates into
that much real influence on rank order if the underlying raw values are
actually spread out across the class, which you cannot guarantee in
advance. A single team running this harness locally cannot compute the
real leaderboard_score in isolation -- course staff assemble it when
aggregating the full class leaderboard (see the instructor-tools
repository's aggregate_leaderboard.py).

Track C is NOT a raw noisy-condition score -- it is a *retention ratio*:
(mean nDCG@10 across the noisy conditions) / (your own clean-condition
nDCG@10, i.e. your Track B raw score), percentile-ranked. See the
spec, Section 7, "A note on why Track C is a ratio, not a raw
score", for why: scoring raw noisy-condition accuracy alone would just
reward whoever already had the best Track A/B accuracy (correlated
signal, not new information); dividing out your own clean-condition score
isolates how much you specifically lose to noise, independent of how
strong you already were.
"""
from typing import Dict, List, Optional

TRACK_A_NDCG_WEIGHT = 0.24
TRACK_A_MAP_WEIGHT = 0.06
TRACK_A_WEIGHT = TRACK_A_NDCG_WEIGHT + TRACK_A_MAP_WEIGHT  # 0.30

TRACK_B_WEIGHT = 0.25
TRACK_C_WEIGHT = 0.35

MAX_BONUS_WEIGHT = 0.10


def _percentile_ranks(values: List[float]) -> List[float]:
    """Given a list of raw values, returns a same-length list of
    percentile ranks in [0.0, 1.0]: 0.0 for the single lowest value, 1.0
    for the single highest, linear in between. Tied values receive the
    same (averaged-rank) percentile. A class of size 1 maps its lone
    value to 1.0 (nothing to be ranked against, so it gets the ceiling --
    matches instructor-tools/grading_rubric.py's treatment of a lone
    baseline-beater). Identical logic and behaviour to the earlier project's
    harness/leaderboard.py._percentile_ranks(); duplicated here rather
    than imported so this repository has no dependency on the
    the earlier project repository.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    percentiles = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        percentile = avg_rank / (n - 1)
        for k in range(i, j + 1):
            percentiles[order[k]] = percentile
        i = j + 1
    return percentiles


def track_a_component(team_ndcg: Dict[str, float], team_map: Dict[str, float]) -> Dict[str, float]:
    """team_ndcg / team_map: {team: raw score_candidates() nDCG@10 / MAP@10}
    for every OK-status team on the set being scored. Returns
    {team: track_a_component} in [0.0, TRACK_A_WEIGHT]."""
    teams = list(team_ndcg.keys())
    ndcg_pct = _percentile_ranks([team_ndcg[t] for t in teams])
    map_pct = _percentile_ranks([team_map[t] for t in teams])
    return {
        teams[i]: TRACK_A_NDCG_WEIGHT * ndcg_pct[i] + TRACK_A_MAP_WEIGHT * map_pct[i]
        for i in range(len(teams))
    }


def track_b_component(team_clean_ndcg: Dict[str, float]) -> Dict[str, float]:
    """team_clean_ndcg: {team: raw relevance_model_feedback() nDCG@10 at
    0% noise}. Returns {team: track_b_component} in [0.0, TRACK_B_WEIGHT]."""
    teams = list(team_clean_ndcg.keys())
    pct = _percentile_ranks([team_clean_ndcg[t] for t in teams])
    return {teams[i]: TRACK_B_WEIGHT * pct[i] for i in range(len(teams))}


def retention_ratio(clean_ndcg: float, noisy_ndcg_values: List[float]) -> float:
    """(mean of noisy_ndcg_values) / clean_ndcg, clamped to [0.0, 1.0] --
    "how much of your clean-condition accuracy survives the noisy
    conditions". If clean_ndcg is 0 (nothing to retain in the first
    place), returns 0.0 rather than dividing by zero: a team that scored
    zero on the clean condition has nothing meaningful to say about
    robustness. Not clamped above 1.0 in principle a team could score
    *better* under noise by chance on a tiny query set, but that is
    capped at 1.0 here so a lucky noisy-condition fluctuation cannot
    read as more than "fully retained"."""
    if clean_ndcg <= 0:
        return 0.0
    if not noisy_ndcg_values:
        return 0.0
    mean_noisy = sum(noisy_ndcg_values) / len(noisy_ndcg_values)
    return max(0.0, min(1.0, mean_noisy / clean_ndcg))


def track_c_component(team_retention: Dict[str, float]) -> Dict[str, float]:
    """team_retention: {team: raw retention_ratio(...) value}. Returns
    {team: track_c_component} in [0.0, TRACK_C_WEIGHT]."""
    teams = list(team_retention.keys())
    pct = _percentile_ranks([team_retention[t] for t in teams])
    return {teams[i]: TRACK_C_WEIGHT * pct[i] for i in range(len(teams))}


def bonus_component(team_bonus_raw: Dict[str, float]) -> Dict[str, float]:
    """team_bonus_raw: {team: raw combined attack+defense score from the
    round-robin} (see instructor-tools' round_robin_bonus.py; teams that
    did not submit an adversarial_set.json are simply absent from this
    dict, not present with a 0). Percentile-ranked among only the teams
    that participated, scaled to [0.0, MAX_BONUS_WEIGHT] -- purely
    additive; a team absent from this dict gets +0.0, never a penalty."""
    teams = list(team_bonus_raw.keys())
    if not teams:
        return {}
    pct = _percentile_ranks([team_bonus_raw[t] for t in teams])
    return {teams[i]: MAX_BONUS_WEIGHT * pct[i] for i in range(len(teams))}


def summarize(
    score_candidates_metrics: Dict[str, float],
    clean_feedback_metrics: Dict[str, float],
    noisy_feedback_ndcg_by_level: Dict[float, float],
    baseline_score_candidates_metrics: Optional[Dict[str, float]] = None,
) -> Dict:
    """Local, single-team sanity-check summary -- NOT the real leaderboard
    score (see module docstring). score_candidates_metrics /
    clean_feedback_metrics: the `aggregate` dicts from
    harness.metrics.evaluate_run for your score_candidates() run and your
    relevance_model_feedback() run at 0% noise, respectively.
    noisy_feedback_ndcg_by_level: {noise_fraction: aggregate nDCG@10} for
    each practice noise level you swept (see harness/noise_injection.py).
    """
    clean_ndcg = clean_feedback_metrics["ndcg@10"]
    noisy_values = [v for level, v in noisy_feedback_ndcg_by_level.items() if level > 0.0]
    retention = retention_ratio(clean_ndcg, noisy_values) if noisy_values else None

    result = {
        "score_candidates_ndcg@10": score_candidates_metrics["ndcg@10"],
        "score_candidates_map@10": score_candidates_metrics["map@10"],
        "clean_feedback_ndcg@10": clean_ndcg,
        "noisy_feedback_ndcg@10_by_level": dict(noisy_feedback_ndcg_by_level),
        "practice_retention_ratio": retention,
        "note": (
            "These are your RAW numbers -- Track A/B/C of the real "
            "leaderboard_score percentile-rank each of these against the "
            "whole class before applying the 30%/25%/35% weights (plus "
            "up to +10% bonus), exactly as the earlier project's accuracy "
            "components do, and for the same reason: a fixed weight only "
            "means what it says if the underlying values are spread out "
            "across the class, which no single team can know in "
            "isolation. The practice_retention_ratio above also uses the "
            "PUBLIC practice noise levels (harness/noise_injection.py), "
            "not the undisclosed grading recipe (spec Section 6) -- "
            "treat it as a useful local signal, not your real Track C "
            "score. None of this is computable without seeing every "
            "team's numbers -- course staff apply all of it when "
            "aggregating the real leaderboard."
        ),
    }
    if baseline_score_candidates_metrics is not None:
        result["baseline_score_candidates_ndcg@10"] = baseline_score_candidates_metrics["ndcg@10"]
        result["beats_baseline_score_candidates_ndcg@10"] = (
            score_candidates_metrics["ndcg@10"] > baseline_score_candidates_metrics["ndcg@10"]
        )
    return result
