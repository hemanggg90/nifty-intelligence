# Quant Strategy Intelligence & Execution System (`nifty-intelligence`)

A research-first trading terminal for Indian **index options (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX)**, **stock options**
(a fixed 15-stock watchlist) and **MCX commodity options**, built as a Streamlit dashboard on top of
the Dhan broker API.

It answers one question, over and over, for every instrument it watches:

> **Which validated strategy has the strongest statistically validated edge under today's market
> conditions - and is there a valid setup right now? If not, do nothing.**

Everything is paper-traded by default. Real orders require a deliberate, double-gated opt-in (see
[Safety model](#safety-model)). **This is research software, not financial advice - nothing here is a
promise of profit** (read [Known limitations](#known-limitations-read-this-before-trusting-a-signal)).

---

## Table of contents

1. [What it does, in one picture](#what-it-does-in-one-picture)
2. [Core ideas](#core-ideas)
3. [How a trade happens, end to end](#how-a-trade-happens-end-to-end)
4. [Dashboard pages](#dashboard-pages)
5. [The strategy library](#the-strategy-library)
6. [Market data](#market-data)
7. [Research layer: features, regimes, backtests, analogues, ranking](#research-layer)
8. [Options layer](#options-layer)
9. [Risk engine and position sizing](#risk-engine-and-position-sizing)
10. [Execution: paper, auto-trading, live](#execution-paper-auto-trading-live)
11. [Safety model](#safety-model)
12. [Quick start](#quick-start)
13. [Configuration reference](#configuration-reference)
14. [Deploying on Streamlit Community Cloud](#deploying-on-streamlit-community-cloud)
15. [Project layout](#project-layout)
16. [Database](#database)
17. [Testing](#testing)
18. [Troubleshooting](#troubleshooting)
19. [Extending it](#extending-it)
20. [Known limitations](#known-limitations-read-this-before-trusting-a-signal)

---

## What it does, in one picture

```
 Dhan API / your CSVs  (real data only - no synthetic fallback)
          |
          v
   DataManager ---- quality gate (gaps, bad OHLC, out-of-session bars, staleness) ----> FAIL/DEGRADED = NO TRADE
          |
          v
   Feature engine (trend, ATR, VWAP, opening range, RSI, Bollinger, pivots ...) -- no look-ahead
          |
          +--> Regime engine (10 regimes, probabilities, not a single label)
          |
          v
   For each of the 17 strategies:  historical backtest  ->  "analogues" (30 most similar past setups)
                                   ->  conditional expected R, win probability, confidence
          |
          v
   Ranking engine ----> best strategy, or  NO TRADE  (first-class outcome)
          |
          v
   Setup detector: does the chosen strategy's entry trigger on the latest bar?   (WAITING / TRIGGERED)
          |
          v
   Option selector (live Dhan option chain)  ->  contract (CE/PE, strike, expiry)
          |
          v
   Premium model: underlying stop/target  ->  option-premium stop/target
          |
          v
   Position sizing (risk % and capital cap)  ->  RISK ENGINE (can VETO)  ->  Paper broker  (or Dhan, if LIVE-authorised)
          |
          v
   Monitor stop/target on live premium, square off before the close, record everything in the database
```

## Core ideas

- **Research first, execution second.** A strategy must *earn* the right to trade today by having
  historically worked in conditions similar to today's. Picking a strategy and *entering a trade* are
  separate steps: a selected strategy sits in `WAITING_FOR_SETUP` until its own entry conditions fire.
- **NO TRADE is a result, not a failure.** The system refuses to trade when data quality is not OK,
  no strategy clears a minimum edge, confidence is too low, or the top two strategies are
  indistinguishable.
- **Real data only.** There is no synthetic or random fallback. If Dhan (or your CSVs) can't supply
  candles, the system raises `DataUnavailableError` and says why, rather than inventing prices.
- **Everything is traded as options.** Indices, stocks and commodities all resolve to an option
  contract from the live chain. (Strategies are *researched* on the underlying's price action and
  *executed* in option premium - see limitations.)
- **A deterministic risk engine has the last word.** It does not depend on the research layer and
  cannot be overridden by it.
- **Transparent, not a black box.** Scores expose their components, analogues are inspectable, regimes
  are rule-based with visible probabilities, and every order, fill, position and risk decision is
  stored with an ID.

## How a trade happens, end to end

1. **Scan.** For each instrument the pipeline loads ~60 days (configurable, 10-180) of 5-minute
   candles, validates them, computes features and classifies the regime.
2. **Rank.** Every strategy is back-tested over that history. The 30 historical setups most similar to
   *now* give each strategy a conditional expected R, win probability and confidence. The ranking
   engine picks a winner or says NO TRADE.
3. **Detect.** The setup detector checks whether the winner's entry condition is true on the latest
   bar (some strategies also need the live option chain).
4. **Resolve a contract.** The live Dhan option chain is fetched; the option selector maps
   LONG -> call, SHORT -> put (BUY) or the mirrored SELL leg, picks the strike (ATM by default,
   configurable moneyness offset) and expiry (nearest by default).
5. **Translate to premium.** The underlying's stop/target distance is scaled by the option's live
   delta into a premium stop/target. A SELL *requires* a valid premium stop (unbounded risk otherwise).
6. **Size.** Quantity (whole lots) is the smaller of a risk-based size (90% of the max risk per trade)
   and a capital cap (max % of equity in one trade). A BUY that costs more than the available cash is
   refused ("Insufficient funds").
7. **Approve.** The risk engine runs its checks in order and either approves or vetoes (logged with a
   decision ID and reason).
8. **Execute.** Paper broker simulates a fill with adverse slippage (or Dhan places a real order, only
   if LIVE is fully authorised).
9. **Manage.** Each cycle, every open option's live premium is compared with its stop/target; positions
   are squared off shortly before the session closes.

## Dashboard pages

Run the app and use the sidebar page list. The main page (`app.py`) is the command center.

| Page | Purpose |
|---|---|
| **Main (`app.py`)** | Instrument/timeframe/lookback controls, Dhan API-key panel, trading-mode badge, the current decision, strategy ranking table and regime probabilities. |
| 01 Market State | The latest feature vector (every feature is documented) and data-quality status. |
| 02 Strategy Intelligence | Global vs conditional (current-state) performance for every strategy. |
| 03 Strategy Library | Each strategy's description, parameters and required features. |
| 04 Backtest Lab | Event-driven backtests with in-sample / out-of-sample split and walk-forward folds; editable parameters. |
| 05 Regime Analysis | Regime probabilities and classifier confidence over time. |
| 06 Historical Analogues | The exact past setups used to score each strategy today. |
| **07 Paper Trading** | A trading terminal for one instrument: quote header (last price, day change, range), candlestick chart with VWAP/EMA overlays and the setup's entry/stop/target lines, **option-chain ladder** (calls | strike | puts, OI, change in OI, IV, ATM highlighted) with an open-interest chart, PCR, max pain and support/resistance, the **strategy leaderboard**, a decision card (regime, selected strategy, setup status) and an **order ticket** showing premium, quantity, amount needed, max loss/profit **after charges**, risk:reward and a funds meter before you submit. "All instruments" scope hands over to the background scanner. |
| **08 Positions & Orders** | A broker-style view of the paper account. Live KPIs (day P&L net of charges, capital used/available, win rate); tabs for **open positions** (cards with live P&L, stop-to-target progress bar, charges and a manual **Exit** / **Exit all** with confirmation), today's trades, filterable trade history, order book, fills (with slippage cost) and **analytics** (equity curve, profit factor, expectancy, payoff, drawdown, streaks, charges breakdown, P&L by strategy/instrument). CSV downloads on every table. |
| 09 Risk Control | Account state, configured limits, the **emergency kill switch**, recent risk events. |
| 10 Research Reports | Structured reports generated only from stored system data. |
| 11 System Logs & Health | Dhan connectivity/credentials status and system logs. |
| **12 Live Options Trading** | Real orders via Dhan - disabled unless fully authorised (see Safety model). |
| **13 Auto Multi-Instrument Trading** | Background auto-trader over NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX and the 15-stock watchlist: control bar, live account/market tiles, a **market watch** (last price, change %, day-range bar, sparkline and the last scan result per instrument), and tabs for positions, scan results, an activity feed (orders, risk vetoes, exits) and today's performance. |
| **14 Commodity Auto Trading (MCX)** | The same layout for a second, independent background auto-trader for MCX commodities with its own session hours and square-off countdown. |

A sidebar indicator on every page shows when auto-trading is running and offers a Stop button.

**Look and feel.** Every trading page starts with a status bar (IST clock, NSE and MCX session open/closed with a
countdown, Dhan connection state, whether each auto-trader is running, PAPER/LIVE mode, kill switch). Numbers use
Indian grouping (lakh/crore) and rupee signs; P&L is always colour **and** triangle **and** sign, never colour alone.
Charts follow a validated colour palette (blue/orange/aqua for series; green/red reserved for good/bad), one axis per
chart, hairline grids and trading-hour gaps removed from intraday candles. The theme lives in `.streamlit/config.toml`
and `ui/theme.py`. Quotes in watchlists come from the on-disk candle cache, so they cost no Dhan requests.
Charges shown are estimates at Dhan's brokerage and the statutory option rates (display only - the paper broker books
gross premium P&L).

## The strategy library

17 strategies implement one interface (`strategies/base_strategy.py`): `generate_historical_setups`,
`check_setup` (live), and a bar-by-bar `simulate_trade` (stop checked before target within a bar -
conservative). Default exits are ATR-based.

| Strategy | Idea |
|---|---|
| Opening Range Breakout | Break of the first-30-minute range. |
| Momentum | Strong recent momentum confirmed by trend and volume. |
| VWAP Mean Reversion | Fade stretched moves away from session VWAP in weak trends. |
| RSI(2) Mean Reversion | Connors-style pullbacks in the direction of the EMA(200) trend. |
| Bollinger Band Mean Reversion | Fade closes outside the bands that reclaim them, in ranges. |
| Supertrend Trend Following | Supertrend(10, 3) direction flips, trailing the line. |
| EMA 50/200 Crossover | Golden/death cross trend following. |
| Donchian Channel Breakout | 20-bar breakout with cooldown. |
| MACD Signal Crossover | MACD/signal cross filtered by EMA(200) direction. |
| CPR Breakout | Break of the prior session's Central Pivot Range. |
| Camarilla Pivot Reversal | Fade prior-session R3/S3 touches toward the pivot. |
| Inside Bar Breakout | Break of an inside bar after a coil. |
| Gap Fill Reversion | Fade a moderate opening gap back toward the prior close. |
| PCR Extreme Contrarian * | Fade chain-wide put/call-ratio extremes with an RSI(2) reversal. |
| OI Buildup Confirmation * | Donchian breakout confirmed by rising near-ATM open interest. |
| IV Crush Premium Writer * | Sell an OTM option when ATM IV is rich vs realised vol. |
| Unusual Volume Fade * | Fade a single-strike volume spike near the money. |

`*` **Chain-aware** strategies need a live option chain. They return no historical setups (there is no
historical chain data), so they can fire live but are never selected by the ranker. See limitations.

## Market data

Preference order, per instrument: **CSV files you provide** (`data_cache/csv/<SYMBOL>_<TIMEFRAME>.csv`
with `timestamp,open,high,low,close,volume`) -> **Dhan** -> error. Results are cached as Parquet (and refreshed only when a new bar closes - see *Staying under Dhan's rate limits*) in
`data_cache/parquet_cache/`.

- **Instruments:** NSE equities, NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX (Dhan `INDEX` candles; SENSEX options route to `BSE_FNO`) and MCX futures underlyings.
- **Intraday intervals** from Dhan: 1, 5, 15, 25, 60 minutes, up to ~90 days per request; otherwise daily.
- **Timestamps:** Dhan returns UTC epoch seconds. The adapter proves the timezone from the data
  itself (bars must fall inside the trading session), shifts to naive IST, and refuses data that fits
  no timezone. All "now" values use an IST clock (`utils/timeutil.py`) because Streamlit Cloud runs in UTC.
- **Forming candle:** the last, still-open bar is dropped so signals never repaint.
- **Stray bars:** a few pre/post-close prints (e.g. zero-volume 15:30 bars) are dropped; if more than 2%
  of bars are out of session the data is **FAIL**.
- **Quality gate** (`data/quality.py`): duplicates, intra-session gaps, zero/negative prices, impossible
  OHLC, invalid volume, staleness against the real market calendar. `OK` is required to trade;
  `DEGRADED`/`FAIL` force NO TRADE and the risk engine vetoes anyway.
- **No volume on indices.** Dhan gives index candles no volume, so volume-dependent strategies
  (Momentum, VWAP Mean Reversion, Unusual Volume Fade) are skipped for NIFTY/BANKNIFTY.
- **Sessions:** NSE 09:15-15:30 IST, MCX 09:00-23:55 IST (`utils/market_profile.py`). Exchange holidays
  are not modelled (weekends only).
- `data_adapters/synthetic.py` still exists but is **used only by the test-suite**, never at runtime.

## Research layer

- **Features** (`features/feature_engine.py`): returns, trend slope, momentum, market structure, ATR and
  ATR percentile, realised vol and vol expansion, relative volume, session-reset VWAP and distance,
  opening range, gap, time since open, RSI(2/14), Bollinger bands, EMAs, MACD, Supertrend, Donchian,
  CPR and Camarilla pivots. Every feature uses only data up to its own bar (backward-looking by
  construction). Derivatives fields (OI, PCR, IV, futures basis) are NaN placeholders by design.
- **Regimes** (`regimes/regime_engine.py`): rule-based scores turned into a softmax probability
  distribution over TREND_UP, TREND_DOWN, RANGE, COMPRESSION, VOL_EXPANSION, VOL_CONTRACTION,
  HIGH_VOL, LOW_VOL, EVENT_DRIVEN, LIQUIDITY_STRESS. No machine learning.
- **Backtesting** (`backtesting/engine.py`): sequential, non-overlapping trades per strategy, 75-bar
  maximum hold. Costs follow how the trade is actually made - as an **option**: Dhan's Rs 20 per order
  (x2) plus STT (sell-side premium), exchange charge, SEBI fee, stamp duty and 18% GST on *premium*
  turnover, plus slippage; NSE and MCX use their own STT/exchange rates. Because the backtest only has
  underlying prices, the premium is an assumption (`OPTION_PREMIUM_PCT` of the underlying, `OPTION_DELTA`).
  Ranking works in **net-of-cost R**. Supports in-sample / out-of-sample splits and rolling walk-forward.
- **Analogues** (`analogues/analogue_engine.py`): a transparent k-NN. The current market state is
  compared (standardised distance) against the entry-time features of every past setup using six
  features - `trend_slope`, `momentum_20`, `atr_percentile_100`, `vol_expansion`, `relative_volume`,
  `vwap_distance_pct`. The 30 nearest give conditional expected R, win rate, median R and confidence.
- **Ranking** (`ranking/ranking_engine.py`, shared with the live pipeline through
  `research/ranking_core.py`): a visible score
  `0.5*edge + 0.3*(P(win)-0.5) + 0.3*confidence_weight - 0.2*drawdown_penalty - cost_penalty`, every
  component exposed. The `edge` is a **shrunk lower confidence bound** of the conditional net-of-cost
  expected R (shrunk toward the strategy's own average, then minus one standard error). Confidence is
  based on how many *independent trading days* the analogues span, and the standard error is
  cluster-robust by day. Drawdown is measured in R. A strategy is *eligible* only with a shrunk edge
  >= 0.05, confidence >= MEDIUM, and (with enough history) a positive edge in at least half of four
  chronological folds. NO TRADE is returned if data quality is not OK, nothing is eligible, or the top two
  are not statistically distinguishable (edge difference < 1 combined standard error).
- **Ranker evaluator** (`ranking/evaluate_ranker.py`): a walk-forward test of the ranker itself. At each
  decision point it ranks using only trades already closed, then compares the selected strategy's realised
  net R with picking a strategy at random. Run `python -m quant_intelligence.ranking.evaluate_ranker`
  (uses the candles already in `data_cache/`; no network).

## Options layer

- **Chain analytics** (`options/chain_analytics.py`): parses Dhan's chain into strikes with LTP, IV,
  OI, greeks, ATM strike, PCR and ATM IV.
- **Contract selection** (`options/option_selector.py`): indices are registered statically
  (NIFTY id 13, BANKNIFTY id 25, segment `IDX_I`); NSE F&O stocks and MCX underlyings resolve
  dynamically from Dhan's scrip master (lot size and strike step refreshed from it). MCX options sit on
  the front-month future, falling back to the next month when the front has no live expiry.
- **Premium model** (`options/premium_model.py`): scales underlying stop/target by live |delta|
  (0.5 if unknown) - a first-order approximation that ignores gamma/theta, so it is an estimate to be
  monitored against live premium, not a guarantee.
- **Staying under Dhan's rate limits.** Dhan's limits apply to the whole *account*, and exceeding them
  (HTTP 429, code 805) risks the account being blocked. Published limits: orders 10/s, data (candles) 5/s,
  quote (live prices) **1/s**, non-trading (funds, positions) 20/s, option chain 1 per 3 s. The app is built
  to stay inside them rather than recover after hitting them:
  - **One limiter for everything** (`brokers/dhan_rate_limit.py`): every Dhan call is classed into one of
    those categories and spaced ~30% under the limit, shared by both background runners and every browser tab.
  - **Chains only when needed:** the option chain is fetched only for an instrument whose strategy has a
    *triggered setup* (or a chain-aware strategy), not for all ~25 instruments every cycle.
  - **Shared single-flight caches** (`brokers/dhan_cache.py`): live prices (2 s) and option chains (15 s) are
    fetched once and shared, so ten open tabs cost the same as one. Expiry lists are cached for 30 minutes.
  - **Candles refresh on bar close:** an instrument is re-requested only when a new bar has closed (about one
    small tail request per instrument per bar), then merged into the cache.
  - **429 handling:** read calls honour `Retry-After` or back off (6/12 s) and retry up to 3 times. After
    repeated 429s a short circuit breaker (30 s, doubling to 2 min) fast-fails reads without touching the
    network and then resumes by itself. **Orders are throttled but never auto-retried** (a retry could
    duplicate an order) and are never blocked by the breaker, so exits can always go out.
  - **Rejected token (401):** nothing more is sent with that token for 5 minutes or until you paste a new
    one; hammering Dhan with a dead token on every cycle would itself risk a block.
  - **Visibility:** *System Logs & Health* shows calls per category over the last minute, 429 counts and any
    pause; a sidebar banner appears on every page while requests are paused.

  The app cannot see requests made by *other* processes on the same account (a second copy of the app, a
  local run plus a cloud deployment, scripts). **Run one instance at a time.**

## Risk engine and position sizing

`risk/risk_engine.py` is deterministic, independent of the research layer, and records every
approval/veto (with reason and per-check results) to the `risk_events` table. Checks run in this order:

1. Emergency kill switch  2. Broker connected  3. Data quality OK  4. Daily loss limit
5. Max drawdown from peak  6. Max trades per day  7. Max risk per trade
8. Max exposure per strategy  9. Max total portfolio exposure  10. Liquidity (relative volume >= 0.3)

Defaults (all overridable by environment variable): risk per trade **1%**, daily loss **3%**, drawdown
**8%**, trades per day **6**, per-strategy exposure **10%**, portfolio exposure **20%**, capital in one
trade **5%** of equity.

**Sizing** (`execution/auto_trader.py::size_position`): `quantity = floor(0.9 x risk budget / stop
distance)` in whole lots, then capped so premium x quantity <= the per-trade capital cap.

**Capital accounting** (`execution/capital.py`): the paper broker only moves cash when a position
*closes*, so capital used is derived from open BUY positions (entry premium x quantity) and
`available = cash - capital used`. Sold-option margin is **not** modelled; SELL premium is reported
separately.

## Execution: paper, auto-trading, live

- **Paper broker** (`brokers/paper_broker.py`): simulated fills with adverse slippage
  (`SLIPPAGE_TICKS x TICK_SIZE`), orders/fills/positions persisted.
- **Background engine** (`execution/engine.py`): a process-wide `TradingEngine` owns the shared broker,
  daily risk counters, the kill switch and the NSE scan loop; a second `ScanRunner` handles MCX
  commodities. Both run in daemon threads, **independent of any page or browser tab** - switching pages
  or closing the tab does not stop them; only **Stop**, a server restart or the host sleeping does. They
  share one cash pool, one risk-counter set and one kill switch.
- **Per cycle** (default every 60 s): scan every instrument -> pipeline -> chain -> setup -> size -> risk ->
  paper order; then check every open option's live premium against stop/target. A failed cycle is logged
  and retried; it never kills the worker.
- **Session handling:** by default trades only inside each market's session; positions are **squared off
  ~5 minutes before the close** at the latest price (also on the first cycle after a restart, if missed).
- **Restart recovery:** on start-up, today's positions, realised P&L and trade counts are rebuilt from the
  database; open rows from earlier days are marked `STALE`, never deleted.
- **Live trading** (`brokers/dhan_broker.py`, page 12): real orders through Dhan's v2 Orders API, only
  when fully authorised (below). It has been tested against Dhan's documented request shapes but **not
  against a real funded account**.

## Safety model

- **Paper by default.** `TRADING_MODE=PAPER`.
- **Live needs two independent switches**: `TRADING_MODE=LIVE` **and**
  `TRADING_LIVE_CONFIRM=YES_I_UNDERSTAND_THE_RISK`. `DhanBroker` re-checks this itself, so a bug elsewhere
  cannot enable real orders.
- **Kill switch** (page 09) blocks all new trades instantly.
- **Mandatory stop on every SELL;** BUY loss is capped at premium paid.
- **Risk engine veto** on every order; nothing downstream can override it.
- **Credentials:** the sidebar "API Keys" panel writes your token to a local `.env` (gitignored). On a
  shared/public deployment that panel would let any visitor change the token - keep such apps private
  and prefer platform Secrets.
- **Never commit `.env`.** It is in `.gitignore`.

## Quick start

Requires **Python 3.11+** (developed on 3.12) and a Dhan account with API access (Dhan Client ID and
access token). Check Dhan's current terms for what its market-data APIs require on your account.

```bash
git clone https://github.com/hemanggg90/nifty-intelligence.git
cd nifty-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create .env in the project root (see Configuration reference). Minimum:
#   DHAN_CLIENT_ID=your-client-id
#   DHAN_ACCESS_TOKEN=your-access-token

streamlit run quant_intelligence/app.py                # run from the project root
```

Open http://localhost:8501. You can also paste the Client ID/token into the sidebar's **API Keys**
panel instead of editing `.env`.

**First run walkthrough**

1. Open the dashboard; with valid credentials the pipeline runs on NIFTY and shows the ranking.
2. Go to **07 Paper Trading -> All instruments together -> Run one cycle now** to scan everything once.
3. Happy with the results? **Start auto trading**; watch **08 Positions & Orders** and **09 Risk Control**.

> Dhan **access tokens expire roughly every 24 hours**. Regenerate one in your Dhan account when you see
> `DH-901 / Invalid_Authentication`.

Helper scripts: `scripts/init_db.py` (create tables), `scripts/smoke_test.py` (run the pipeline once),
`scripts/ui_smoke_test.py` (headlessly execute every page and report exceptions).

## Configuration reference

Everything is an environment variable (loaded from `.env`); safe defaults let it run in PAPER mode with
SQLite even with no `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `TRADING_MODE` | `PAPER` | `PAPER` or `LIVE`. |
| `TRADING_LIVE_CONFIRM` | *(empty)* | Must equal `YES_I_UNDERSTAND_THE_RISK` for live orders. |
| `DATABASE_URL` | local SQLite in `data_cache/` | Any SQLAlchemy URL (e.g. Postgres). |
| `DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN` | *(empty)* | Dhan credentials. |
| `DHAN_BASE_URL` | `https://api.dhan.co/v2` | Dhan API base URL. |
| `OPTION_UNDERLYINGS` | `NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY,SENSEX` | Index underlyings scanned. |
| `OPTION_MONEYNESS_OFFSET` | `0` | 0 = ATM; +N = N strikes further OTM. |
| `OPTION_EXPIRY_PREFERENCE` | `NEAREST` | `NEAREST` or `NEAREST_MONTHLY`. |
| `OPTION_PRODUCT_TYPE` | `INTRADAY` | `INTRADAY`, `MARGIN` or `CNC` (order product type). |
| `DEFAULT_INSTRUMENT`, `DEFAULT_TIMEFRAME` | `NIFTY`, `5min` | Dashboard defaults. |
| `PAPER_STARTING_CAPITAL` | `1000000` | Paper account size (INR). |
| `AUTO_TRADE_REFRESH_SECONDS` | `60` | Pause between auto-trading cycles. |
| `LTP_REFRESH_SECONDS` | `5` | Live price refresh on the dashboard. |
| `EOD_SQUARE_OFF` | `true` | Close positions near each market's close. |
| `EOD_SQUARE_OFF_MINUTES` | `5` | How many minutes before the close. |
| `MAX_RISK_PER_TRADE_PCT` | `1.0` | Max loss at stop, % of equity. |
| `MAX_CAPITAL_PER_TRADE_PCT` | `5.0` | Max premium outlay in one trade, % of equity (0 disables). |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Daily loss limit. |
| `MAX_STRATEGY_EXPOSURE_PCT` | `10.0` | Risk concentrated in one strategy. |
| `MAX_PORTFOLIO_EXPOSURE_PCT` | `20.0` | Total capital at risk. |
| `MAX_TRADES_PER_DAY` | `6` | Daily trade count limit. |
| `MAX_DRAWDOWN_PCT` | `8.0` | Max drawdown from peak equity. |
| `BROKERAGE_PER_ORDER_INR` | `20` | Dhan's flat brokerage per executed F&O order. |
| `SLIPPAGE_TICKS`, `TICK_SIZE` | `1`, `0.05` | Slippage in premium units; also used for paper fills. |
| `OPT_STT_SELL_RATE_NSE`, `OPT_STT_SELL_RATE_MCX` | `0.0015`, `0.0005` | STT on sell-side option premium. |
| `OPT_TXN_RATE_NSE`, `OPT_TXN_RATE_MCX` | `0.0003553`, `0.000418` | Exchange charge on premium turnover. |
| `SEBI_RATE`, `STAMP_BUY_RATE`, `GST_RATE` | `0.000001`, `0.00003`, `0.18` | SEBI fee, buy-side stamp duty, GST on brokerage + exchange + SEBI. |
| `DHAN_MIN_INTERVAL_QUOTE_SEC`, `DHAN_MIN_INTERVAL_DATA_SEC`, `DHAN_MIN_INTERVAL_NONTRADING_SEC`, `DHAN_MIN_INTERVAL_ORDERS_SEC`, `DHAN_MIN_INTERVAL_CHAIN_SEC` | `1.25`, `0.35`, `0.1`, `0.17`, `3.5` | Minimum seconds between Dhan calls per category (set under Dhan's limits of 1/s, 5/s, 20/s, 10/s, 1 per 3 s). Lower them only if you are sure of your account's limits. |
| `DHAN_429_MAX_ATTEMPTS`, `DHAN_429_BACKOFF_SEC` | `3`, `6` | Tries and base backoff for reads that get HTTP 429. |
| `DHAN_BREAKER_THRESHOLD`, `DHAN_BREAKER_COOLDOWN_SEC`, `DHAN_BREAKER_MAX_COOLDOWN_SEC` | `3`, `30`, `120` | Consecutive 429s that pause reads, and the pause length (doubles up to the max). |
| `DHAN_AUTH_FAILURE_COOLDOWN_SEC` | `300` | How long to stop sending requests with a token Dhan rejected (401). |
| `DHAN_QUOTE_CACHE_TTL_SEC`, `DHAN_CHAIN_CACHE_TTL_SEC` | `2`, `15` | How long a live-price / option-chain response is shared between tabs and the auto-traders. |
| `OPTION_PREMIUM_PCT`, `OPTION_DELTA` | `1.5`, `0.5` | Assumed option premium (% of the underlying's price) and delta used to convert the underlying backtest into option P&L and costs. |

> The statutory rates are the published exchange/government charges (taken from Zerodha's public charges
> table, since Dhan's pricing page lists only brokerage, GST and the SEBI fee). Check them against your own
> Dhan contract note and override via the variables above if they differ.

Watchlists live in code: **`config/watchlist.py`** (15 stocks and 8 MCX commodities).

Stocks: RELIANCE, BHARTIARTL, HDFCBANK, ICICIBANK, SBIN, TCS, BAJFINANCE, LT, LICI, SUNPHARMA, HINDUNILVR,
TITAN, ADANIPORTS, INFY, KOTAKBANK. Commodities: CRUDEOIL, CRUDEOILM, NATURALGAS, GOLD, GOLDM, SILVER,
SILVERM, COPPER (lot multipliers come from MCX's published specs, not Dhan - verify before trusting P&L).
Use real NSE symbols (e.g. `HDFCBANK`, `LICI`).

## Deploying on Streamlit Community Cloud

1. Push the repo to GitHub; create the app with main file `quant_intelligence/app.py`.
2. App **Settings -> Secrets**:
   ```toml
   DHAN_CLIENT_ID = "your-client-id"
   DHAN_ACCESS_TOKEN = "your-token"
   ```
   Top-level secrets are exposed as environment variables, so no code change is needed.
3. **Refresh the token daily** (it expires) and reboot the app.
4. Keep the app **private** (restrict viewers) - the account, controls and kill switch are shared by
   everyone who can open it.

Cloud caveats: servers run in UTC (the app uses an IST clock internally); the disk is ephemeral, so the
SQLite database, paper account and cached candles reset on reboot; free apps **sleep after inactivity**,
which stops the background traders - "run until I stop it" cannot be guaranteed there; and Dhan generally
requires a whitelisted static IP for *order* APIs, which the cloud does not provide (data APIs are fine).
A `.devcontainer/` is included for Codespaces.

## Project layout

```
quant_intelligence/
  app.py                       Streamlit command center
  pages/                       01-14 dashboard pages
  config/        settings.py (env config) | watchlist.py (stocks, commodities) | credentials.py (token updates)
  data/          data_manager.py (source order, cache, quality) | quality.py (validation/cleaning)
  data_adapters/ dhan_adapter | dhan_instrument_master (scrip master) | csv_adapter | nse_heatmap | synthetic (tests only)
  features/      feature_engine.py
  market_state/  latest feature snapshot
  regimes/       regime_engine.py
  strategies/    base_strategy.py | registry.py | 17 strategy modules
  backtesting/   engine.py (backtest, costs, walk-forward)
  analogues/     analogue_engine.py (k-NN of similar past setups)
  ranking/       ranking_engine.py (score, eligibility, NO TRADE)
  research/      pipeline.py (orchestrates data -> ranking)
  options/       chain_analytics | option_selector | premium_model
  execution/     engine.py (background runners) | multi_cycle | auto_trader | setup_detector
                 position_monitor | square_off | capital
  risk/          risk_engine.py
  brokers/       base_broker | paper_broker | dhan_broker (live) | dhan_api_client (REST)
                 dhan_rate_limit (account-wide limiter, 429 breaker) | dhan_cache (shared price/chain caches)
  database/      db.py | models.py
  reports/       report_generator.py
  ui/            state | theme | components (status bar, KPI tiles, position cards) | charts (Plotly) | tables
                 format (Indian numbers) | position_views, chain_views (pure analytics: P&L, stats, ladder, ticket)
                 open_positions (shared live positions block) | actions (manual exit) | market_data (cached quotes)
                 multi_instrument_panel | credentials_panel
  utils/         timeutil (IST) | market_profile (NSE/MCX) | market_calendar | logging_utils
  tests/         pytest suite
scripts/         init_db.py | smoke_test.py | ui_smoke_test.py
data_cache/      (gitignored) SQLite DB, parquet candle cache, optional csv/
logs/            (gitignored) system.log (JSON lines)
```

## Database

SQLAlchemy models (`database/models.py`), SQLite by default. Tables: `instruments`,
`market_data_metadata` (provenance and quality report of every data fetch), `market_states`,
`regime_states`, `strategy_definitions`, `strategy_observations`, `backtest_runs`, `backtest_trades`,
`strategy_metrics`, `analogue_matches`, `signals`, **`orders`**, **`fills`**, **`positions`**,
**`risk_events`** (every approval and veto), `system_events`, `research_reports`. Columns added to models
later are migrated automatically on start (`ALTER TABLE ... ADD COLUMN`).

## Testing

```bash
python -m pytest -q          # 150+ tests, ~1 minute, no network
python scripts/ui_smoke_test.py   # runs every page headlessly
```

Tests cover the data-quality gate, timezone alignment, feature no-look-ahead, strategies, backtester,
analogues, ranking, the risk engine, the paper broker, option selection and the premium model, Dhan client
request shapes/retry/caching (HTTP is mocked), capital accounting, the background engine (start/stop,
failure isolation, market-hours and kill-switch behaviour), square-off and restart recovery. Tests use a
temporary database and never write to your real one.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Dhan API error (401) ... DH-901 Invalid_Authentication` | Access token expired (about 24 h) or wrong. Generate a new one and update the sidebar / Secrets. |
| `Dhan API error (429) ... 805 Too many requests` | Dhan's per-account rate limit. The app spaces, caches and retries calls and pauses briefly if it persists (see *Health* page / sidebar banner). Usually another app copy, tab or script is using the same account - run **one** instance. |
| "Dhan rejected your credentials (401)" / "Authentication Failed" | The access token expired or is wrong. Requests are paused for 5 min or until you enter a new token under **API Keys**. |
| `DataUnavailableError: No real market data ...` | No CSV and Dhan failed or isn't configured; the message lists the reason. |
| Data quality `FAIL` / `DEGRADED` | Read the issues in the log: stale data (market closed), gaps, bad bars, or misaligned timezone. The system stays in NO TRADE. |
| Everything is `NO TRADE` | Often correct: no strategy clears the edge/confidence bar, or data isn't `OK`. See Known limitations. |
| Chain error for one instrument | Check `logs/system.log`; usually rate limit or an expired token. |
| Trades missing after a restart | Paper state is in memory; today's book is restored from the database, older open rows are marked `STALE`. |
| Times look 5.5 h off | Server clock is UTC; the app uses IST internally - restart on the latest version. |

## Extending it

- **Add a strategy:** subclass `BaseStrategy` (implement `generate_historical_setups` and `check_setup`),
  then register it in `strategies/registry.py`. The backtester, ranking engine and UI pick it up with no
  other changes.
- **Change the watchlist:** edit `config/watchlist.py` (stocks must be NSE F&O symbols; commodities
  need their lot multiplier).
- **Add a data source:** implement `data_adapters/base.py::DataAdapter` and add it to `DataManager`.
- **Tune risk:** use the `MAX_*` environment variables - the risk engine reads them at start-up.

## Known limitations (read this before trusting a signal)

- **No proven edge.** The strategies are well-known indicator setups. Nothing here has been shown to be
  profitable after costs: on ~60 days of 5-minute data the strategies are roughly breakeven *before* costs
  and slightly negative after realistic option costs. An earlier walk-forward check of the original ranker
  (17 instruments, ~260 trades) found **no measurable advantage over picking a strategy at random**; that
  check used an older, inflated cost model, and re-measuring the current ranker with `evaluate_ranker` is the
  way to judge it. Treat output as research, run
  extended paper trading, and never size real money from it without your own validation.
- **Costs are modelled, not measured.** The backtest has no historical option prices, so premium is assumed
  to be a fixed % of the underlying and to move by a fixed delta (`OPTION_PREMIUM_PCT`, `OPTION_DELTA`);
  real premiums, spreads, theta and IV differ by instrument, strike and time to expiry. Statutory rates come
  from published tables - verify them against your contract note. With these costs, a net-of-cost ranker may
  say NO TRADE most of the time when no strategy shows an edge; that is intended.
- **Researched on the underlying, traded in premium.** Edge is measured on index/stock price moves;
  execution is option premium, which also moves with theta, IV and bid-ask spread. The premium model is a
  delta approximation.
- **Chain-aware strategies cannot be ranked.** PCR, OI-buildup, IV-crush and unusual-volume strategies have
  no historical setups, so the ranker never selects them.
- **Index volume is unavailable** from Dhan, so volume-based features and strategies are skipped for
  NIFTY/BANKNIFTY.
- **Ranking is statistically thin.** Analogues are the 30 nearest setups, mostly from the same sessions
  (autocorrelated), and the winner is picked from many strategies on the same data used to estimate its
  edge - so the optimism in "best of N" is not corrected.
- **Paper != live.** Paper fills use simple slippage on the last price; real fills depend on liquidity and
  spread. Sold-option margin is not modelled. Exchange holidays are not modelled.
- **Live trading is unproven** against a funded account.
- **Paper state is in memory** (restored from the database on restart for the current day only).

## Disclaimer

This software is provided for research and educational purposes. Trading derivatives involves substantial
risk of loss. Nothing here is investment advice, and the authors accept no liability for any losses.
