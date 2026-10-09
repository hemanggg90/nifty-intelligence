"""NSE Heatmap: what is moving, and which movers the auto paper traders should scan.

The map shows every watchlist index, stock and commodity coloured by % change. The Selection panel picks the biggest
gainers and losers per group and tells the NSE and commodity auto-traders to scan only those, taking only bullish
setups on up-movers and bearish ones on down-movers. Prices come from the candle cache (no Dhan request).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.engine import COMMODITY_RUNNER, ENGINE
from quant_intelligence.execution.mover_selection import DOWN, UP, MoverConfig, compute_selection
from quant_intelligence.ui.charts import heatmap_treemap
from quant_intelligence.ui.components import chip, empty_state, page_header
from quant_intelligence.ui.heatmap_data import GROUPS, WINDOWS, build_heatmap_frame, groups_of
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.timeutil import now_ist

_CFG = {"displayModeBar": False}
_ACTION = {UP: "▲ calls only", DOWN: "▼ puts only"}
RUNNERS = (("NSE auto-trader (indices + stocks)", ENGINE, ("Index", "Stock")),
           ("Commodity auto-trader (MCX)", COMMODITY_RUNNER, ("Commodity",)))

apply_theme()
init_session_state()
page_header(
    "NSE Heatmap",
    "Everything on the watchlist at a glance, coloured by % change. Pick the biggest gainers and losers and let the "
    "auto paper traders scan only those - up-movers take calls, down-movers take puts.",
)


def _selection_panel() -> None:
    st.markdown("#### Select movers for paper trading")
    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    n = c1.slider("Gainers and losers per group", 1, 15, SETTINGS.mover_top_n, key="hm_n",
                  help="Per group: NSE indices, NSE stocks, commodities. Up to N gainers and N losers each.")
    min_abs = c2.number_input("Minimum move (%)", 0.0, 5.0, float(SETTINGS.mover_min_abs_pct), 0.1, key="hm_min",
                              help="A smaller move is treated as noise and not selected, so a quiet day selects fewer than N.")
    auto = c3.checkbox("Keep refreshing the selection every scan cycle", value=True, key="hm_auto",
                       help="On: the list follows the market through the day. Off: the list is fixed at the moment you press Apply.")
    config = MoverConfig(n=int(n), min_abs_pct=float(min_abs), max_age_minutes=SETTINGS.mover_max_age_minutes, auto_refresh=bool(auto))
    preview = compute_selection(groups_of(), config)

    if preview.usable:
        rows = [{"Symbol": s, "Group": preview.group_of[s], "Change today": preview.change_pct[s], "Allowed": _ACTION[preview.picks[s]]}
                for s in sorted(preview.picks, key=lambda s: -abs(preview.change_pct[s]))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(38 * (len(rows) + 1) + 3, 420),
                     column_config={"Change today": st.column_config.NumberColumn(format="%+.2f%%")})
        st.caption(f"Preview: {preview.summary()}. {preview.reason}".strip())
    else:
        st.info(preview.reason or "Nothing to select right now.")

    b1, b2, b3 = st.columns(3)
    if b1.button("Apply to NSE auto-trader", type="primary", width="stretch", key="hm_apply_nse"):
        ENGINE.set_mover_config(config, preview.subset(("Index", "Stock")))
        st.toast("NSE auto-trader will scan only the selected movers")
    if b2.button("Apply to commodity auto-trader", type="primary", width="stretch", key="hm_apply_mcx"):
        COMMODITY_RUNNER.set_mover_config(config, preview.subset(("Commodity",)))
        st.toast("Commodity auto-trader will scan only the selected movers")
    if b3.button("Clear (scan everything)", width="stretch", key="hm_clear"):
        ENGINE.clear_mover_config()
        COMMODITY_RUNNER.clear_mover_config()
        st.toast("Both auto-traders scan every instrument again")

    for title, runner, _ in RUNNERS:
        cfg, sel = runner.mover_config, runner.mover_selection
        if cfg is None:
            st.markdown(f"{chip('OFF', 'muted')} **{title}** scans every instrument.", unsafe_allow_html=True)
        else:
            used = f" - last scan used: {sel.summary()}" if sel is not None and sel.usable else (f" - {sel.reason}" if sel is not None and sel.reason else "")
            st.markdown(f"{chip('ON', 'good')} **{title}** scans the top {cfg.n} gainers and losers per group "
                        f"(min {cfg.min_abs_pct:g}%, {'refreshed every cycle' if cfg.auto_refresh else 'fixed'}){used}", unsafe_allow_html=True)
    st.caption(
        "Selection only narrows which instruments are scanned and which direction may be taken; the strategies still decide "
        "whether to enter, and the risk engine and loss limits apply as always. Positions already open are still monitored and "
        "exited. Setups against the move are skipped and counted in the daily report. Written (SELL) options are not filtered. "
        "There is no evidence yet that following the day's move improves results - check the Daily Report after a few sessions."
    )


@st.fragment(run_every="30s")
def _map() -> None:
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
    window = c1.selectbox("Change over", list(WINDOWS), key="hm_window", help="'Today' is against the previous session's close.")
    shown = c2.multiselect("Show", list(GROUPS), default=list(GROUPS), key="hm_groups")
    size_by = c3.selectbox("Tile size", ["Equal", "Day range (busier = bigger)"], key="hm_size")
    df = build_heatmap_frame(window)
    df = df[df["group"].isin(shown)]

    with_data = df[df["has_data"]]
    if with_data.empty:
        empty_state("No candle data yet", "The data keeper fills the candle cache once a Dhan token is set; until then there is nothing to map.")
        return
    newest = with_data["as_of"].max()
    if with_data["fresh"].sum() == 0:
        st.warning(f"The newest candle is from {newest:%d %b %H:%M} IST - the market is closed or the feed is not refreshing, "
                   "so this shows the last session, not live moves. Nothing is selected for trading from it.")
    else:
        st.caption(f"Candles as of {newest:%H:%M} IST · {int(with_data['fresh'].sum())} of {len(df)} instruments current")

    limit = max(1.0, float(with_data["change_pct"].abs().quantile(0.9))) if window != "Today" else 3.0
    st.plotly_chart(heatmap_treemap(df, "day_range_pct" if size_by.startswith("Day") else None, color_limit=limit),
                    width="stretch", config=_CFG)
    missing = df[~df["has_data"]]["symbol"].tolist()
    if missing:
        st.caption(f"No candle data for: {', '.join(missing)} (shown nowhere on the map rather than as 0%).")

    t1, t2 = st.columns(2)
    cols = {"symbol": "Symbol", "group": "Group", "change_pct": "Change", "last": "Last"}
    show = with_data.sort_values("change_pct")[list(cols)].rename(columns=cols)
    cfg = {"Change": st.column_config.NumberColumn(format="%+.2f%%"), "Last": st.column_config.NumberColumn(format="%.2f")}
    with t1:
        st.markdown("**Top gainers**")
        st.dataframe(show[show["Change"] > 0].tail(8).iloc[::-1], hide_index=True, width="stretch", column_config=cfg)
    with t2:
        st.markdown("**Top losers**")
        st.dataframe(show[show["Change"] < 0].head(8), hide_index=True, width="stretch", column_config=cfg)
    with st.expander("All instruments (table view)"):
        full = df.sort_values("change_pct", ascending=False)[["symbol", "name", "group", "sector", "change_pct", "last", "day_range_pct", "as_of"]]
        st.dataframe(full, hide_index=True, width="stretch", column_config={
            "change_pct": st.column_config.NumberColumn("Change", format="%+.2f%%"), "last": st.column_config.NumberColumn("Last", format="%.2f"),
            "day_range_pct": st.column_config.NumberColumn("Day range", format="%.2f%%"), "as_of": st.column_config.DatetimeColumn("Candle", format="DD MMM HH:mm")})


_map()
st.divider()
_selection_panel()
st.caption(f"Page rendered {now_ist():%H:%M:%S} IST. Prices are the last closed 5-minute bar from the candle cache.")
