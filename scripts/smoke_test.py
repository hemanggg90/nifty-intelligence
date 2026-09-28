"""Quick end-to-end smoke test: run the full pipeline once on synthetic data."""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_intelligence.database.db import init_db
from quant_intelligence.research.pipeline import run_pipeline

init_db()

end = dt.datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
start = end - dt.timedelta(days=60)

result = run_pipeline("NIFTY", "5min", start, end)

print("Instrument:", result.instrument)
print("Timestamp:", result.timestamp)
print("Data quality:", result.data_quality_status)
print("Regime:", result.regime_label, "confidence:", round(result.regime_confidence, 3))
print()
for si in result.strategy_intel:
    print(f"--- {si.strategy_name} ---")
    print("  Global metrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in si.global_metrics.items()})
    print("  Conditional:", si.conditional_metrics)
    print("  Score:", round(si.score.score, 4), "eligible:", si.score.eligible, si.score.ineligibility_reason)
print()
print("RANKING DECISION:")
print("  Selected:", result.ranking.selected_strategy)
print("  No trade:", result.ranking.is_no_trade)
print("  Reason:", result.ranking.reason)
print("\nSMOKE TEST PASSED")
