import os
import json
import time
import requests
import pandas as pd
from datetime import datetime

CACHE_FILE = "cache/token_prices.json"

# ---------------------------
# Cache Helpers
# ---------------------------
def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# In-memory cache
_price_cache = _load_cache()

def _cache_key(symbol: str, date: str | None = None):
    return f"{symbol.upper()}_{date}" if date else symbol.upper()

# ---------------------------
# Coingecko API Helpers
# ---------------------------
def _fetch_coingecko_current(symbol: str):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if symbol.lower() in data:
            return float(data[symbol.lower()]["usd"])
    return None

def _fetch_coingecko_historical(symbol: str, date: str):
    """Date format: DD-MM-YYYY (Coingecko requirement)."""
    url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/history?date={date}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        try:
            return float(data["market_data"]["current_price"]["usd"])
        except Exception:
            return None
    return None

# ---------------------------
# Moralis Fallback
# ---------------------------
def _fetch_moralis_price(address: str, chain: str):
    api_key = os.getenv("MORALIS_API_KEY")
    if not api_key:
        return None
    url = f"https://deep-index.moralis.io/api/v2/erc20/{address}/price?chain={chain}"
    headers = {"X-API-Key": api_key}
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        return float(data.get("usdPrice", 0)) or None
    return None

# ---------------------------
# Public Functions
# ---------------------------
def get_token_price(symbol: str, address: str = None, chain: str = None, block_time: pd.Timestamp = None):
    """
    Return USD price of a token.
    - Uses cache if available
    - Falls back: Coingecko -> Moralis
    - Supports historical (if block_time given)
    """

    if block_time is not None:
        date_str = block_time.strftime("%d-%m-%Y")
        cache_key = _cache_key(symbol, date_str)
    else:
        date_str = None
        cache_key = _cache_key(symbol)

    # 1. Check cache
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    price = None

    # 2. Try Coingecko (historical or current)
    try:
        if date_str:
            price = _fetch_coingecko_historical(symbol, date_str)
        else:
            price = _fetch_coingecko_current(symbol)
    except Exception:
        pass

    # 3. Fallback Moralis
    if price is None and address and chain:
        try:
            price = _fetch_moralis_price(address, chain)
        except Exception:
            pass

    # 4. Save to cache
    if price is not None:
        _price_cache[cache_key] = price
        _save_cache(_price_cache)

    return price
