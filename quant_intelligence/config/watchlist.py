"""Fixed stock watchlist traded together with the indices (NIFTY/BANKNIFTY).

Symbols are Dhan/NSE trading symbols (verified against the Dhan scrip master):
note HDFC Bank is HDFCBANK and LIC is LICI on NSE.
"""
from __future__ import annotations

WATCHLIST_STOCKS: list[dict[str, str]] = [
    {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "Conglomerate (Energy/Telecom/Retail)"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "sector": "Telecommunications"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "sector": "Financial Services (Banking)"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "sector": "Financial Services (Banking)"},
    {"symbol": "SBIN", "name": "State Bank of India", "sector": "Financial Services (Banking)"},
    {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "sector": "Financial Services (NBFC)"},
    {"symbol": "LT", "name": "Larsen & Toubro", "sector": "Engineering & Infrastructure"},
    {"symbol": "LICI", "name": "Life Insurance Corporation", "sector": "Financial Services (Insurance)"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries", "sector": "Healthcare"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "sector": "Fast-Moving Consumer Goods (FMCG)"},
    {"symbol": "TITAN", "name": "Titan Company", "sector": "Consumer Cyclical"},
    {"symbol": "ADANIPORTS", "name": "Adani Ports & SEZ", "sector": "Infrastructure & Logistics"},
    {"symbol": "INFY", "name": "Infosys", "sector": "Information Technology"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "sector": "Financial Services (Banking)"},
]
