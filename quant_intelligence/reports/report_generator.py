"""
Research Report Generator.

Builds a structured research report entirely from actual system data (the
pipeline output, backtest results, and analogue evidence). No numbers are
invented here - this module only formats what the engine already computed.
An AI copilot (if enabled later) may only summarize this content, never
replace or fabricate figures within it.
"""
from __future__ import annotations

import datetime as dt
from quant_intelligence.utils.timeutil import now_ist


def generate_report(pipeline_output) -> dict:
    o = pipeline_output
    strategy_sections = []
    for si in o.strategy_intel:
        strategy_sections.append(
            {
                "strategy": si.strategy_name,
                "global_metrics": si.global_metrics,
                "conditional_metrics": si.conditional_metrics,
                "score": si.score.score,
                "eligible": si.score.eligible,
                "ineligibility_reason": si.score.ineligibility_reason,
                "n_analogues": len(si.analogues),
            }
        )

    caveats = [
        "Historical performance does not guarantee future results.",
        "Conditional expectancy estimates are based on a finite historical sample from the "
        "configured lookback window and may not generalize to unseen market conditions.",
        "Synthetic or CSV research data may not perfectly reflect live market microstructure "
        "(fills, latency, slippage) - paper-trading results will differ from backtests.",
    ]
    low_conf = [s for s in o.strategy_intel if s.conditional_metrics.get("confidence_label") in ("LOW", "INSUFFICIENT_DATA")]
    if low_conf:
        caveats.append(
            f"{len(low_conf)} strategy(ies) have LOW/INSUFFICIENT_DATA confidence in the current "
            "regime - their conditional estimates should not be relied upon."
        )

    content = {
        "generated_at": str(now_ist()),
        "instrument": o.instrument,
        "as_of_timestamp": str(o.timestamp),
        "data_quality_status": o.data_quality_status,
        "regime": {"label": o.regime_label, "confidence": o.regime_confidence, "probabilities": o.regime_probabilities},
        "ranking": {
            "selected_strategy": o.ranking.selected_strategy,
            "is_no_trade": o.ranking.is_no_trade,
            "reason": o.ranking.reason,
        },
        "strategies": strategy_sections,
        "caveats": caveats,
    }

    markdown = _to_markdown(content)
    return {"content_json": content, "content_markdown": markdown}


def _to_markdown(content: dict) -> str:
    lines = [
        f"# Research Report - {content['instrument']}",
        f"Generated: {content['generated_at']}  |  As of: {content['as_of_timestamp']}",
        "",
        f"**Data Quality:** {content['data_quality_status']}",
        f"**Regime:** {content['regime']['label']} ({content['regime']['confidence']*100:.0f}% confidence)",
        "",
        "## Decision",
        f"**{'NO TRADE' if content['ranking']['is_no_trade'] else content['ranking']['selected_strategy']}**",
        f"Reason: {content['ranking']['reason']}",
        "",
        "## Strategy Evidence",
    ]
    for s in content["strategies"]:
        lines.append(f"### {s['strategy']}")
        lines.append(f"- Score: {s['score']:.4f} | Eligible: {s['eligible']}")
        if not s["eligible"]:
            lines.append(f"- Ineligibility reason: {s['ineligibility_reason']}")
        cm = s["conditional_metrics"]
        lines.append(
            f"- Conditional expected R: {cm.get('expected_r')}, confidence: {cm.get('confidence_label')}, "
            f"sample size: {cm.get('sample_size')} (n analogues used: {s['n_analogues']})"
        )
        gm = s["global_metrics"]
        lines.append(f"- Global (unconditional): {gm.get('n_trades', 0)} trades, win rate {gm.get('win_rate')}")
        lines.append("")

    lines.append("## Caveats")
    for c in content["caveats"]:
        lines.append(f"- {c}")

    return "\n".join(lines)


def persist_report(instrument: str, report: dict) -> None:
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import ResearchReport

        with get_session() as session:
            session.add(
                ResearchReport(
                    instrument=instrument,
                    report_type="COMMAND_CENTER_SNAPSHOT",
                    content_json=report["content_json"],
                    content_markdown=report["content_markdown"],
                )
            )
    except Exception:
        pass
