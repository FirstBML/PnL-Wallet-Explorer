import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuration
COINGECKO_API_KEY = os.getenv("GECKO_API_KEY", "")
COINGECKO_BASE_URL = "pro-api.coingecko.com" if COINGECKO_API_KEY else "api.coingecko.com"
REQUEST_DELAY = 1.5  # Unified rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 5
PRICE_CACHE = {}

# Token ID mappings to avoid API calls
TOKEN_ID_OVERRIDES = {
    "BTC": "bitcoin",
    "ETH": "ethereum", 
    "USDT": "tether",
    "USDC": "usd-coin",
    "BNB": "binancecoin",
    "ADA": "cardano",
    "XRP": "ripple",
    "SOL": "solana",
    "DOT": "polkadot",
    "DOGE": "dogecoin",
    "MATIC": "matic-network",
    "AVAX": "avalanche-2",
    "SHIB": "shiba-inu",
    "UNI": "uniswap",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "ALGO": "algorand",
    "ICP": "internet-computer",
    "VET": "vechain",
    "ARB": "arbitrum",
    "WBTC": "wrapped-bitcoin",
    "WETH": "weth",
    "DAI": "dai"
}

def is_likely_scam_token(token_symbol, token_address):
    """Filter out obvious scam tokens"""
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

def rate_limited_request(url, params=None, headers=None):
    """Make a rate-limited request with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                time.sleep(RETRY_DELAY)
            else:
                time.sleep(REQUEST_DELAY)
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                st.warning(f"Rate limit hit, waiting {RETRY_DELAY} seconds before retry {attempt + 1}/{MAX_RETRIES}")
                continue
            else:
                st.warning(f"API request failed with status {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                continue
            return None
    
    st.error("Max retries exceeded")
    return None

def get_coin_id(blockchain: str, token_symbol: str, token_address: str = None) -> str:
    """Get CoinGecko coin ID with multiple strategies"""
    if not token_symbol:
        return None
    
    # 1. Check hardcoded mappings first
    if token_symbol.upper() in TOKEN_ID_OVERRIDES:
        return TOKEN_ID_OVERRIDES[token_symbol.upper()]
    
    # 2. Try contract address lookup
    if token_address and blockchain:
        chain_platforms = {
            'ethereum': 'ethereum',
            'bsc': 'binance-smart-chain',
            'polygon': 'polygon-pos',
            'arbitrum': 'arbitrum-one',
            'optimism': 'optimism',
            'base': 'base'
        }
        
        platform = chain_platforms.get(blockchain.lower())
        if platform:
            url = f"https://{COINGECKO_BASE_URL}/api/v3/coins/{platform}/contract/{token_address}"
            headers = {"X-Cg-Pro-Api-Key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
            
            data = rate_limited_request(url, headers=headers)
            if data and 'id' in data:
                return data['id']
    
    # 3. Fallback to search API
    url = f"https://{COINGECKO_BASE_URL}/api/v3/search"
    params = {"query": token_symbol}
    headers = {"X-Cg-Pro-Api-Key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
    
    data = rate_limited_request(url, params=params, headers=headers)
    if data and 'coins' in data and data['coins']:
        return data['coins'][0]['id']
    
    return None

def get_price_range(coin_id: str, from_date: datetime, to_date: datetime):
    """Get price range data with rate limiting"""
    url = f"https://{COINGECKO_BASE_URL}/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': int(from_date.timestamp()),
        'to': int(to_date.timestamp())
    }
    headers = {"X-Cg-Pro-Api-Key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
    
    st.info(f"Fetching prices for {coin_id} from {from_date.date()} to {to_date.date()}")
    
    data = rate_limited_request(url, params=params, headers=headers)
    if data and 'prices' in data:
        st.info(f"Retrieved {len(data['prices'])} price points for {coin_id}")
        return data['prices']
    
    return []

def assign_nearest_price(ts: datetime, price_data: list):
    """Find nearest available price to the given timestamp"""
    if not price_data:
        return None
    
    target_timestamp = int(ts.timestamp() * 1000)  # Convert to milliseconds
    
    # Find closest timestamp
    closest_price = min(price_data, key=lambda x: abs(x[0] - target_timestamp))
    return closest_price[1]

def assign_stablecoin_prices(df):
    """Assign $1.00 to stablecoins"""
    stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FRAX', 'LUSD']
    
    for coin in stablecoins:
        mask = (
            (df['token_symbol'].str.upper() == coin) & 
            (df['price_usd'].isna() | (df['price_usd'] <= 0))
        )
        
        if mask.any():
            df.loc[mask, 'price_usd'] = 1.0
            df.loc[mask, 'usd_value'] = df.loc[mask, 'amount'] * 1.0
            st.success(f"Assigned $1.00 price for {mask.sum()} {coin} transactions")
    
    return df

def fill_missing_prices_batch(df):
    """Fill missing prices with improved rate limiting and error handling"""
    df_copy = df.copy()
    
    # First, assign stablecoin prices
    df_copy = assign_stablecoin_prices(df_copy)
    
    # Filter out scam tokens
    scam_mask = df_copy.apply(lambda row: is_likely_scam_token(row['token_symbol'], row.get('token_address')), axis=1)
    if scam_mask.any():
        st.warning(f"Filtered out {scam_mask.sum()} likely scam tokens")
        df_copy = df_copy[~scam_mask]
    
    # Find remaining missing prices
    missing_prices = df_copy[df_copy['price_usd'].isna() | (df_copy['price_usd'] <= 0)]
    
    if missing_prices.empty:
        st.success("All prices are available!")
        return df_copy
    
    st.info(f"Fetching prices for {len(missing_prices)} transactions across {missing_prices['token_symbol'].nunique()} tokens")
    
    # Group by token to minimize API calls
    grouped = missing_prices.groupby(['token_symbol', 'blockchain'])
    progress_bar = st.progress(0)
    total_groups = len(grouped)
    processed_groups = 0
    
    for (token_symbol, blockchain), group in grouped:
        st.write(f"Processing {token_symbol} on {blockchain}...")
        
        # Get CoinGecko ID
        sample_address = group['token_address'].iloc[0] if 'token_address' in group.columns else None
        coin_id = get_coin_id(blockchain, token_symbol, sample_address)
        
        if not coin_id:
            st.warning(f"Could not find CoinGecko ID for {token_symbol}")
            processed_groups += 1
            progress_bar.progress(processed_groups / total_groups)
            continue
        
        # Get date range for this token group - FIXED BUG HERE
        dates = pd.to_datetime(group['block_time'])
        min_date = dates.min()
        max_date = dates.max()  # Fixed: was dates.min() 
        
        # Add buffer around dates
        min_date = min_date - pd.Timedelta(days=1)
        max_date = max_date + pd.Timedelta(days=1)
        
        try:
            # Get price data for the date range
            price_data = get_price_range(coin_id, min_date, max_date)
            
            if price_data:
                # Fill prices for this token group
                filled_count = 0
                for idx, row in group.iterrows():
                    tx_timestamp = pd.to_datetime(row['block_time'])
                    price = assign_nearest_price(tx_timestamp, price_data)
                    
                    if price and price > 0:
                        df_copy.at[idx, 'price_usd'] = price
                        df_copy.at[idx, 'usd_value'] = row['amount'] * price
                        filled_count += 1
                
                if filled_count > 0:
                    st.success(f"Filled {filled_count} prices for {token_symbol}")
                else:
                    st.warning(f"Could not assign any prices for {token_symbol}")
            else:
                st.warning(f"No price data returned for {token_symbol}")
                
        except Exception as e:
            st.error(f"Error processing {token_symbol}: {e}")
        
        processed_groups += 1
        progress_bar.progress(processed_groups / total_groups)
    
    progress_bar.empty()
    
    # Show final summary
    original_missing = len(df[df['price_usd'].isna() | (df['price_usd'] <= 0)])
    final_missing = len(df_copy[df_copy['price_usd'].isna() | (df_copy['price_usd'] <= 0)])
    filled_count = original_missing - final_missing
    
    st.success(f"Successfully filled {filled_count} missing prices. {final_missing} prices still missing.")
    
    return df_copy

# Backward compatibility wrapper
def fill_missing_prices(df):
    """Wrapper for backward compatibility"""
    return fill_missing_prices_batch(df)

def get_token_price(blockchain: str, token_address: str, token_symbol: str, ts: datetime = None):
    """Get single token price for backward compatibility"""
    coin_id = get_coin_id(blockchain, token_symbol, token_address)
    if not coin_id:
        return None

    if ts is None:
        ts = datetime.utcnow()

    # Use a 2-day window around timestamp
    start_date = ts - pd.Timedelta(days=1)
    end_date = ts + pd.Timedelta(days=1)

    price_data = get_price_range(coin_id, start_date, end_date)
    return assign_nearest_price(ts, price_data)