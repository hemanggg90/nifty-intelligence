"""Volatility modelling layer: estimators, forecasts and (later phases) GARCH / HAR-RV / implied vol.

Research code only. It is never imported by `risk/` (the risk engine stays deterministic and independent).
Every function here is causal: a value at time t uses only data up to and including t.
"""
