import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Config
COINGECKO_API_KEY = os.getenv("GECKO_API_KEY", "")
COINGECKO_BASE_URL = "pro-api.coingecko.com" if COINGECKO_API_KEY else "api.coingecko.com"
RATE_LIMIT_DELAY = 2
PRICE_CACHE = {}

# Scam filtering
def is_likely_scam_token(token_symbol, token_address):
    if not token_symbol or pd.isna(token_symbol):
        return True
    scam_indicators = [
        'http://', 'https://', 'visit', 'claim', 'airdrop', 'free',
        'generator', '://', 'www.', '.com', '.io', '.vip', '.supply',
        'rarible', 'uniswap', 'reward', 'promo', 'bonus'
    ]
    token_lower = str(token_symbol).lower()
    for indicator in scam_indicators:
        if indicator in token_lower:
            return True
    if len(token_symbol) > 50:
        return True
    if any(ord(char) > 127 for char in token_symbol) and len(token_symbol) < 15:
        return True
    if token_address and len(token_address) != 42:
        return True
    return False

# Update Config section
COINGECKO_API_KEY = os.getenv("GECKO_API_KEY", "")
COINGECKO_BASE_URL = "pro-api.coingecko.com" if COINGECKO_API_KEY else "api.coingecko.com"
RATE_LIMIT_DELAY = 2
PRICE_CACHE = {}

# Update TOKEN_ID_OVERRIDES with more mappings
TOKEN_ID_OVERRIDES = {
    "USDC": "usd-coin",
    "USDT": "tether",
    "ARB": "arbitrum",
    "wBTC": "wrapped-bitcoin",
    "WETH": "weth",
    "LINK": "chainlink",
    "DAI": "dai"
}

def get_coin_id(blockchain: str, token_symbol: str, token_address: str = None) -> str:
    """
    Resolves CoinGecko coin ID from token details.
    Prioritizes contract address lookup over symbol matching.
    """
    if not token_symbol or not token_address:
        return None
        
    # 1. Try contract address lookup first
    chain_platforms = {
        'ethereum': 'ethereum',
        'bsc': 'binance-smart-chain',
        'polygon': 'polygon-pos',
        'arbitrum': 'arbitrum-one',
        'optimism': 'optimism',
        'base': 'base'
    }
    
    platform = chain_platforms.get(blockchain.lower())
    if platform and token_address:
        try:
            url = f"https://{COINGECKO_BASE_URL}/api/v3/coins/{platform}/contract/{token_address}"
            headers = {"X-Cg-Pro-Api-Key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                return r.json().get('id')
            elif r.status_code != 404:  # Log non-404 errors
                st.warning(f"API error looking up {token_symbol} ({token_address}): {r.status_code}")
        except Exception as e:
            st.warning(f"Error looking up contract: {e}")
    
    # 2. Fallback to override mappings only for well-known tokens
    if token_symbol.upper() in TOKEN_ID_OVERRIDES:
        return TOKEN_ID_OVERRIDES[token_symbol.upper()]
    
    return None

def get_market_chart_range(coin_id: str, start_date: datetime, end_date: datetime):
    """Fetch price data for a date range."""
    url = f"https://{COINGECKO_BASE_URL}/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": int(start_date.timestamp()),
        "to": int(end_date.timestamp())
    }
    headers = {"X-Cg-Pro-Api-Key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
    
    try:
        st.info(f"Fetching prices for {coin_id} from {start_date} to {end_date}")
        r = requests.get(url, params=params, headers=headers, timeout=30)
        st.info(f"Response status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            prices = {int(ts/1000): price for ts, price in data.get("prices", [])}
            st.info(f"Retrieved {len(prices)} price points for {coin_id}")
            if prices:
                return prices
            else:
                st.warning(f"No prices returned for {coin_id} despite 200 status")
                st.debug(f"Raw response: {data}")
        elif r.status_code == 429:
            st.error("Rate limit exceeded - please wait")
            time.sleep(RATE_LIMIT_DELAY * 2)
        else:
            st.error(f"API error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Error fetching market chart for {coin_id}: {e}")
    return {}

def assign_nearest_price(ts: datetime, price_dict: dict):
    """Find nearest available price to the given timestamp."""
    if not price_dict:
        return None
    target = int(ts.timestamp())
    available_ts = np.array(list(price_dict.keys()))
    nearest_idx = (np.abs(available_ts - target)).argmin()
    nearest_ts = available_ts[nearest_idx]
    return price_dict[nearest_ts]

def fill_missing_prices_batch(df):
    """
    Batch mode: fetches ranges instead of per-day history calls.
    """
    if df.empty:
        return df

    missing_rows = df[df['Price Status'] == '❌ Missing']
    if missing_rows.empty:
        st.success("All prices already available!")
        return df

    legit_missing_rows = missing_rows[
        ~missing_rows.apply(
            lambda row: is_likely_scam_token(row['token_symbol'], row['token_address']),
            axis=1
        )
    ]

    if legit_missing_rows.empty:
        st.info("No legitimate tokens with missing prices found.")
        return df

    st.info(f"Batch fetching prices for {len(legit_missing_rows)} transactions...")

    grouped = legit_missing_rows.groupby(["blockchain", "token_symbol", "token_address"])

    for (blockchain, symbol, address), group in grouped:
        coin_id = get_coin_id(blockchain, symbol, address)
        if not coin_id:
            st.warning(f"⚠️ No CoinGecko ID for {symbol}")
            continue

        start_date = group['block_time'].min() - pd.Timedelta(days=2)
        end_date = group['block_time'].max() + pd.Timedelta(days=2)

        cache_key = f"{coin_id}:{start_date}:{end_date}"
        if cache_key in PRICE_CACHE:
            price_dict = PRICE_CACHE[cache_key]
        else:
            price_dict = get_market_chart_range(coin_id, start_date, end_date)
            PRICE_CACHE[cache_key] = price_dict

        for idx, row in group.iterrows():
            ts = row['block_time']
            price = assign_nearest_price(ts, price_dict)
            if price and price > 0:
                df.at[idx, 'price_usd'] = price
                df.at[idx, 'usd_value'] = row['amount'] * price
                df.at[idx, 'Price Status'] = '✅ Available'
            else:
                st.warning(f"Could not assign price for {symbol} at {ts}")

    # 👇 ensure this is ALWAYS reached
    return df

# Wrapper for backward compatibility
def fill_missing_prices(df):
    return fill_missing_prices_batch(df)

def get_token_price(blockchain: str, token_address: str, token_symbol: str, ts: datetime = None):
    """
    Simple wrapper for backwards compatibility.
    Returns a single historical price (nearest available).
    """
    coin_id = get_coin_id(blockchain, token_symbol, token_address)
    if not coin_id:
        return None

    if ts is None:
        ts = datetime.utcnow()

    # Use a 2-day window around timestamp
    start_date = ts - pd.Timedelta(days=2)
    end_date = ts + pd.Timedelta(days=2)

    price_dict = get_market_chart_range(coin_id, start_date, end_date)
    return assign_nearest_price(ts, price_dict)


