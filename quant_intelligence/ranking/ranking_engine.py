"""
Strategy Ranking & NO-TRADE Engine.

Transparent scoring: combines conditional expected R, probability of positive
return, confidence/sample quality, drawdown/risk character, and a cost
penalty into one visible score per strategy. Every component is exposed so
the UI can show exactly why a strategy ranked where it did - nothing is a
black box.

NO-TRADE is a first-class outcome, chosen when:
  - No strategy clears a minimum expected-edge bar after costs.
  - The best strategy's confidence is LOW/INSUFFICIENT_DATA.
  - The gap between the top two strategies is not practically significant.
  - Data quality is not OK.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CONFIDENCE_WEIGHTS = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3, "INSUFFICIENT_DATA": 0.0}

MIN_EXPECTED_R_AFTER_COSTS = 0.05  # minimum edge (in R) required to even consider trading
MIN_CONFIDENCE_TO_TRADE = "MEDIUM"
MIN_SCORE_GAP_FOR_SIGNIFICANCE = 0.08  # top-2 score gap below this => ambiguous, prefer NO TRADE

CONFIDENCE_RANK = {"INSUFFICIENT_DATA": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass
class StrategyScore:
    strategy_name: str
    score: float
    expected_r: float | None
    prob_positive_return: float | None
    confidence_label: str
    sample_size: int
    max_drawdown: float | None
    eligible: bool
    ineligibility_reason: str | None = None
    components: dict = field(default_factory=dict)


def score_strategy(conditional_metrics: dict, max_drawdown: float | None, cost_penalty: float = 0.0) -> StrategyScore:
    name = conditional_metrics.get("strategy_name", "unknown")
    confidence = conditional_metrics.get("confidence_label", "INSUFFICIENT_DATA")
    sample_size = conditional_metrics.get("sample_size", 0)
    expected_r = conditional_metrics.get("expected_r")
    prob_pos = conditional_metrics.get("prob_positive_return")

    eligible = True
    reason = None

    if confidence == "INSUFFICIENT_DATA" or expected_r is None:
        eligible = False
        reason = f"Insufficient historical analogues ({sample_size}) to estimate conditional edge"
        return StrategyScore(name, 0.0, expected_r, prob_pos, confidence, sample_size, max_drawdown, eligible, reason)

    conf_weight = CONFIDENCE_WEIGHTS.get(confidence, 0.0)
    drawdown_penalty = min(abs(max_drawdown) / 10.0, 1.0) if max_drawdown else 0.0

    expected_r_component = expected_r
    prob_component = (prob_pos or 0.5) - 0.5  # center at 0

    score = (
        0.5 * expected_r_component
        + 0.3 * prob_component
        + 0.3 * conf_weight
        - 0.2 * drawdown_penalty
        - cost_penalty
    )

    if expected_r < MIN_EXPECTED_R_AFTER_COSTS:
        eligible = False
        reason = f"Conditional expected R ({expected_r:.3f}) below minimum edge threshold ({MIN_EXPECTED_R_AFTER_COSTS})"
    elif CONFIDENCE_RANK.get(confidence, 0) < CONFIDENCE_RANK.get(MIN_CONFIDENCE_TO_TRADE, 2):
        eligible = False
        reason = f"Confidence '{confidence}' below minimum required '{MIN_CONFIDENCE_TO_TRADE}'"

    return StrategyScore(
        strategy_name=name,
        score=score,
        expected_r=expected_r,
        prob_positive_return=prob_pos,
        confidence_label=confidence,
        sample_size=sample_size,
        max_drawdown=max_drawdown,
        eligible=eligible,
        ineligibility_reason=reason,
        components={
            "expected_r_component": 0.5 * expected_r_component,
            "prob_component": 0.3 * prob_component,
            "confidence_component": 0.3 * conf_weight,
            "drawdown_penalty": -0.2 * drawdown_penalty,
            "cost_penalty": -cost_penalty,
        },
    )


@dataclass
class RankingDecision:
    ranked: list[StrategyScore]
    selected_strategy: str | None
    is_no_trade: bool
    reason: str


def rank_and_select(scores: list[StrategyScore], data_quality_status: str = "OK") -> RankingDecision:
    ranked = sorted(scores, key=lambda s: s.score, reverse=True)

    if data_quality_status != "OK":
        return RankingDecision(ranked, None, True, f"Data quality is {data_quality_status}: NO TRADE forced")

    eligible = [s for s in ranked if s.eligible]

    if not eligible:
        return RankingDecision(ranked, None, True, "No strategy clears minimum edge/confidence thresholds")

    top = eligible[0]
    if len(eligible) > 1:
        gap = top.score - eligible[1].score
        if gap < MIN_SCORE_GAP_FOR_SIGNIFICANCE:
            return RankingDecision(
                ranked,
                None,
                True,
                f"Top strategies ({top.strategy_name} vs {eligible[1].strategy_name}) are statistically/"
                f"practically indistinguishable (score gap {gap:.3f} < {MIN_SCORE_GAP_FOR_SIGNIFICANCE})",
            )

    return RankingDecision(ranked, top.strategy_name, False, f"{top.strategy_name} has the strongest validated conditional edge")
