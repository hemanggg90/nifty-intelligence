from quant_intelligence.ranking.ranking_engine import rank_and_select, score_strategy


def test_insufficient_data_marks_ineligible():
    metrics = {"strategy_name": "X", "confidence_label": "INSUFFICIENT_DATA", "sample_size": 3, "expected_r": None, "prob_positive_return": None}
    score = score_strategy(metrics, max_drawdown=-100)
    assert not score.eligible
    assert "Insufficient" in score.ineligibility_reason


def test_strong_edge_is_eligible():
    metrics = {"strategy_name": "X", "confidence_label": "HIGH", "sample_size": 150, "expected_r": 0.5, "prob_positive_return": 0.6}
    score = score_strategy(metrics, max_drawdown=-100)
    assert score.eligible


def test_weak_edge_below_threshold_is_ineligible():
    metrics = {"strategy_name": "X", "confidence_label": "HIGH", "sample_size": 150, "expected_r": 0.01, "prob_positive_return": 0.51}
    score = score_strategy(metrics, max_drawdown=-100)
    assert not score.eligible


def test_low_confidence_is_ineligible_even_with_good_expected_r():
    metrics = {"strategy_name": "X", "confidence_label": "LOW", "sample_size": 15, "expected_r": 0.5, "prob_positive_return": 0.6}
    score = score_strategy(metrics, max_drawdown=-100)
    assert not score.eligible


def test_no_eligible_strategy_forces_no_trade():
    scores = [
        score_strategy({"strategy_name": "A", "confidence_label": "LOW", "sample_size": 5, "expected_r": 0.5, "prob_positive_return": 0.6}, -50),
        score_strategy({"strategy_name": "B", "confidence_label": "INSUFFICIENT_DATA", "sample_size": 2, "expected_r": None, "prob_positive_return": None}, -50),
    ]
    decision = rank_and_select(scores)
    assert decision.is_no_trade
    assert decision.selected_strategy is None


def test_bad_data_quality_forces_no_trade_even_with_good_scores():
    scores = [score_strategy({"strategy_name": "A", "confidence_label": "HIGH", "sample_size": 200, "expected_r": 1.0, "prob_positive_return": 0.7}, -10)]
    decision = rank_and_select(scores, data_quality_status="FAIL")
    assert decision.is_no_trade
    assert "quality" in decision.reason.lower()


def test_ambiguous_top_two_forces_no_trade():
    scores = [
        score_strategy({"strategy_name": "A", "confidence_label": "HIGH", "sample_size": 200, "expected_r": 0.5, "prob_positive_return": 0.55}, -10),
        score_strategy({"strategy_name": "B", "confidence_label": "HIGH", "sample_size": 200, "expected_r": 0.5, "prob_positive_return": 0.55}, -10),
    ]
    decision = rank_and_select(scores)
    assert decision.is_no_trade


def test_clear_winner_is_selected():
    scores = [
        score_strategy({"strategy_name": "A", "confidence_label": "HIGH", "sample_size": 200, "expected_r": 1.0, "prob_positive_return": 0.7}, -5),
        score_strategy({"strategy_name": "B", "confidence_label": "LOW", "sample_size": 15, "expected_r": 0.1, "prob_positive_return": 0.51}, -5),
    ]
    decision = rank_and_select(scores)
    assert decision.selected_strategy == "A"
    assert not decision.is_no_trade
