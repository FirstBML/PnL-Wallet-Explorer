# app.py - Full Streamlit Wallet PnL Explorer (Cached vs Fresh + CoinGecko FIFO PnL timeline)

import os
import time
import asyncio
import logging
import threading
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union


import numpy as np
import pandas as pd
import altair as alt
import aiohttp
import requests
import streamlit as st
import nest_asyncio

from dotenv import load_dotenv
from dune_client.client import DuneClient
# Load query output from Dune

load_dotenv()
DUNE_API_KEY = os.getenv('DUNE_CLIENT_API')
coingecko_api_key = os.getenv('GECKO_API_KEY')
dune = DuneClient(DUNE_API_KEY)

# Apply nest_asyncio to reduce async/Streamlit conflicts
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wallet-pnl")

# -----------------------
# Config / Secrets
# -----------------------
DUNE_QUERY_ID = 5687370
DUNE_API_BASE_URL = "https://api.dune.com/api/v1"

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

DUNE_API_KEY = get_secret("DUNE_CLIENT_API")
COINGECKO_API_KEY = get_secret("GECKO_API_KEY")


# -----------------------
# Dune helpers (fresh execution / poll) + cached fetch
# -----------------------
def execute_dune_query(query_id: int, parameters: Dict = None) -> str:
    """Start fresh execution on Dune and return execution_id"""
    if not DUNE_API_KEY:
        raise RuntimeError("Dune API key not configured")
    url = f"{DUNE_API_BASE_URL}/query/{query_id}/execute"
    headers = {"X-Dune-API-Key": DUNE_API_KEY}
    dune_parameters = []

    if parameters:
        # blockchains: format for VALUES SQL: ('ethereum'),('bnb')
        if "blockchains" in parameters and parameters["blockchains"]:
            bc_val = parameters["blockchains"]
            if isinstance(bc_val, list):
                bc_val = ",".join([f"('{c}')" for c in bc_val])
            dune_parameters.append({"key": "blockchains", "type": "text", "value": bc_val})
        if "wallet_address" in parameters:
            dune_parameters.append({"key": "wallet", "type": "text", "value": parameters["wallet_address"]})
        if "start_date" in parameters:
            sd = parameters["start_date"]
            if isinstance(sd, datetime):
                sd = sd.strftime("%Y-%m-%d")
            dune_parameters.append({"key": "start_date", "type": "date", "value": sd})
        if "end_date" in parameters:
            ed = parameters["end_date"]
            if isinstance(ed, datetime):
                ed = ed.strftime("%Y-%m-%d")
            dune_parameters.append({"key": "end_date", "type": "date", "value": ed})

    payload = {"parameters": dune_parameters} if dune_parameters else {}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["execution_id"]

def poll_dune_results(execution_id: str, max_wait: int = 180, interval: int = 5) -> pd.DataFrame:
    """Poll Dune execution until finished and return rows as DataFrame"""
    if not DUNE_API_KEY:
        raise RuntimeError("Dune API key not configured")
    url = f"{DUNE_API_BASE_URL}/execution/{execution_id}/results"
    headers = {"X-Dune-API-Key": DUNE_API_KEY}

    waited = 0
    while waited < max_wait:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state")
        if state == "QUERY_STATE_COMPLETED":
            rows = data.get("result", {}).get("rows", [])
            return pd.DataFrame(rows)
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Dune query {execution_id} failed or cancelled: {data.get('error')}")
        time.sleep(interval)
        waited += interval

    raise TimeoutError("Dune query timed out")

def get_latest_cached_result_df(query_id: int) -> (pd.DataFrame, Optional[str]):
    """Fetch latest cached results using dune_client if available; return (df, last_updated_iso)"""
    try:
        from dune_client.client import DuneClient
    except Exception:
        raise RuntimeError("dune_client not installed or available")

    client = DuneClient(DUNE_API_KEY)
    res = client.get_latest_result(query_id)
    # Some dune_client responses include execution timestamps; attempt to extract a last-updated field
    last_updated = getattr(res, "execution_ended_at", None)
    if hasattr(res, "result") and getattr(res.result, "rows", None) is not None:
        df = pd.DataFrame(res.result.rows)
    else:
        # Fallback if dune client format changed
        rows = getattr(res, "rows", None) or []
        df = pd.DataFrame(rows)
    return df, last_updated

# -----------------------
# Demo fallback
# -----------------------
def get_fallback_demo_data() -> pd.DataFrame:
    return pd.DataFrame({
        'action': ['deposit', 'withdrawal', 'deposit'],
        'block_time': [datetime.now() - timedelta(days=i) for i in range(3)],
        'blockchain': ['ethereum', 'bnb', 'arbitrum'],
        'gas_usd': [10.5, 5.2, 8.1],
        'token_in_address': ['0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', None, '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'],
        'token_in_amount': [1.5, None, 2.0],
        'token_in_symbol': ['ETH', None, 'ETH'],
        'token_out_address': [None, '0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c', None],
        'token_out_amount': [None, 3000.0, None],
        'token_out_symbol': [None, 'WBNB', None],
        'trade_value_usd': [3000.0, 3000.0, 4000.0],
        'tx_hash': ['0x123...', '0x456...', '0x789...']
    })

import pandas as pd
import asyncio
import aiohttp
import time
from datetime import datetime
import logging
import threading
from functools import lru_cache
import streamlit as st
import nest_asyncio
from dune_client.client import DuneClient
from dune_client.types import QueryParameter, ParameterType

# --- Apply nest_asyncio for Streamlit runtime ---
nest_asyncio.apply()

# --- Logging config ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# DUNE SQL QUERY
# ----------------------
DUNE_QUERY_ID = 5687370  # replace with your actual query id

# ----------------------
# CoinGecko PnL Calculator
# ----------------------
class ScalablePnLCalculator:
    def __init__(self, coingecko_api_key: str = None):
        self.coingecko_api_key = coingecko_api_key or COINGECKO_API_KEY
        self.coingecko_cache = {}
        self.token_mapping_cache = {}
        self.cache_lock = threading.Lock()
        self.session = None

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    def _get_coingecko_headers(self):
        if self.coingecko_api_key:
            return {'x-cg-pro-api-key': self.coingecko_api_key}
        return {}

    async def get_coingecko_id_from_address(self, token_address: str, blockchain: str):
        cache_key = f"{blockchain}_{token_address.lower()}"
        with self.cache_lock:
            if cache_key in self.token_mapping_cache:
                return self.token_mapping_cache[cache_key]

        platform_map = {
            'ethereum': 'ethereum',
            'bnb': 'binance-smart-chain',
            'arbitrum': 'arbitrum-one',
            'optimism': 'optimistic-ethereum',
            'base': 'base'
        }

        if blockchain not in platform_map:
            return None

        try:
            url = f"https://api.coingecko.com/api/v3/coins/{platform_map[blockchain]}/contract/{token_address}"
            async with self.session.get(url, headers=self._get_coingecko_headers()) as response:
                if response.status == 200:
                    coin_data = await response.json()
                    with self.cache_lock:
                        self.token_mapping_cache[cache_key] = coin_data['id']
                    return coin_data['id']
        except Exception as e:
            logger.warning(f"Direct lookup failed for {token_address}: {e}")

        return None

    @lru_cache(maxsize=1000)
    async def get_historical_price(self, coingecko_id: str, timestamp: datetime):
        cache_key = f"{coingecko_id}_{timestamp.date()}"
        with self.cache_lock:
            if cache_key in self.coingecko_cache:
                return self.coingecko_cache[cache_key]

        try:
            date_str = timestamp.strftime('%d-%m-%Y')
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/history"
            params = {'date': date_str, 'localization': 'false'}

            async with self.session.get(url, params=params, headers=self._get_coingecko_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    price = data['market_data']['current_price']['usd']
                    with self.cache_lock:
                        self.coingecko_cache[cache_key] = price
                    await asyncio.sleep(0.5)
                    return price
        except Exception as e:
            logger.warning(f"Price fetch failed for {coingecko_id}: {e}")

        return None

    async def calculate_wallet_pnl(self, df: pd.DataFrame, wallet_address: str):
        await self.init_session()
        try:
            df_clean = df.copy()
            df_clean['block_time'] = pd.to_datetime(df_clean['block_time'])
            df_clean = df_clean.sort_values('block_time')

            inventory = {}
            realized_pnl = 0
            processed_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for _, row in df_clean.iterrows():
                try:
                    action = str(row.get('action', '')).lower().strip()
                    blockchain = str(row.get('blockchain', '')).lower().strip()

                    if action == 'deposit' and pd.notna(row.get('token_in_address')):
                        token_address = str(row['token_in_address']).lower()
                        amount = float(row['token_in_amount'])
                        coingecko_id = await self.get_coingecko_id_from_address(token_address, blockchain)
                        if not coingecko_id:
                            continue
                        price = await self.get_historical_price(coingecko_id, row['block_time'])
                        if not price:
                            continue
                        gas_cost = float(row.get('gas_usd', 0)) or 0
                        effective_cost = price + (gas_cost / amount) if amount > 0 else price
                        if token_address not in inventory:
                            inventory[token_address] = []
                        inventory[token_address].append((amount, effective_cost, row['block_time']))

                    elif action == 'withdrawal' and pd.notna(row.get('token_out_address')):
                        token_address = str(row['token_out_address']).lower()
                        amount = float(row['token_out_amount'])
                        coingecko_id = await self.get_coingecko_id_from_address(token_address, blockchain)
                        if not coingecko_id:
                            continue
                        current_price = await self.get_historical_price(coingecko_id, row['block_time'])
                        if not current_price:
                            continue
                        if token_address in inventory and inventory[token_address]:
                            remaining = amount
                            while remaining > 0 and inventory[token_address]:
                                lot_amount, lot_cost, _ = inventory[token_address][0]
                                take = min(lot_amount, remaining)
                                proceeds = take * current_price
                                cost = take * lot_cost
                                realized_pnl += proceeds - cost
                                remaining -= take
                                if take == lot_amount:
                                    inventory[token_address].pop(0)
                                else:
                                    inventory[token_address][0] = (lot_amount - take, lot_cost, _)

                except Exception as e:
                    logger.error(f"Error processing transaction: {e}")
                finally:
                    processed_count += 1
                    progress = processed_count / len(df_clean)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {processed_count}/{len(df_clean)} transactions")

            # Unrealized
            unrealized_pnl = 0
            for token_address, lots in inventory.items():
                coingecko_id = await self.get_coingecko_id_from_address(token_address, 'ethereum')
                if coingecko_id:
                    current_price = await self.get_historical_price(coingecko_id, datetime.now())
                    if current_price:
                        for amount, cost, _ in lots:
                            unrealized_pnl += amount * (current_price - cost)

            return {
                'wallet_address': wallet_address,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl + unrealized_pnl,
                'remaining_tokens': sum(sum(amt for amt, _, _ in lots) for lots in inventory.values()),
                'processed_transactions': processed_count
            }
        finally:
            await self.close_session()


# ----------------------
# Streamlit App
# ----------------------
def main():
    st.set_page_config(page_title="Wallet PnL Explorer", page_icon="💰", layout="wide")

    st.title("💰 Wallet PnL Explorer")
    st.markdown("Fetch wallet transactions from Dune + calculate realized and unrealized PnL with FIFO")

    # Load API keys safely
    dune_api_key = get_secret("DUNE_CLIENT_API")
    coingecko_api_key = get_secret("COINGECKO_API_KEY")
    dune = None
    if dune_api_key:
        from dune_client.client import DuneClient
        dune = DuneClient(api_key=dune_api_key)

    # Inputs
    wallet_address = st.text_input("Wallet Address", value="0x...")
    blockchains = st.multiselect(
        "Select Blockchains",
        ["ethereum", "bnb", "arbitrum", "optimism", "base"],
        default=["ethereum"]
    )

    # Date inputs
    start_date_input = st.date_input("Start Date", datetime(2023, 1, 1))
    end_date_input = st.date_input("End Date", datetime.now().date())

    # Convert to datetime with full timestamp
    start_date = datetime.combine(start_date_input, datetime.min.time())  # 00:00:00
    end_date = datetime.combine(end_date_input, datetime.max.time())      # 23:59:59

    query_mode = st.radio("Query Mode", ["Use Cached Results (faster)", "Run Fresh Query (slower)"], index=0)

    if st.button("Run Analysis", type="primary"):
        if not dune:
            st.error("DUNE_API_KEY not configured. Check your .streamlit/secrets.toml or environment variables.")
            return

        with st.spinner("Fetching data from Dune..."):
            try:
                if query_mode == "Use Cached Results (faster)":
                    # -----------------------------
                    # Cached mode: fetch latest results
                    # -----------------------------
                    res = dune.get_latest_result(DUNE_QUERY_ID)

                    # Extract rows safely
                    if hasattr(res, "result") and getattr(res.result, "rows", None) is not None:
                        df = pd.DataFrame(res.result.rows)
                    else:
                        rows = getattr(res, "rows", None) or []
                        df = pd.DataFrame(rows)

                    if df.empty:
                        st.warning("No cached transactions found.")
                        return

                    # -----------------------------
                    # Column mapping / defaults
                    # -----------------------------
                    df.rename(columns={
                        'token_in_amount': 'token_in_amount',
                        'token_in_address': 'token_in_address',
                        'token_in_symbol': 'token_in_symbol',
                        'token_out_address': 'token_out_address',
                        'token_out_amount': 'token_out_amount',
                        'token_out_symbol': 'token_out_symbol',
                        'block_time': 'block_time',
                        'blockchain': 'blockchain',
                        'action': 'action',
                        'tx_hash': 'tx_hash'
                    }, inplace=True)

                    # Add missing gas_usd column
                    if 'gas_usd' not in df.columns:
                        df['gas_usd'] = 0.0

                    # Convert block_time to datetime
                    df['block_time'] = pd.to_datetime(df['block_time'], errors='coerce')
                    df = df.dropna(subset=['block_time'])

                else:
                    # -----------------------------
                    # Fresh query with parameters
                    # -----------------------------
                    params = [
                        QueryParameter.text_type("wallet", wallet_address),
                        QueryParameter.text_type("blockchains_csv", ",".join(blockchains)),
                        QueryParameter.date_type("start_date", start_date.strftime("%Y-%m-%d %H:%M:%S")),
                        QueryParameter.date_type("end_date", end_date.strftime("%Y-%m-%d %H:%M:%S"))
                    ]
                    res = dune.run_query(DUNE_QUERY_ID, parameters=params)
                    df = pd.DataFrame(res.result.rows)

                st.write("Preview of transactions:")
                st.dataframe(df.head())

                if df.empty:
                    st.warning("No transactions found for this wallet in the given date range")
                    return

                # -----------------------------
                # PnL Calculation
                # -----------------------------
                calculator = ScalablePnLCalculator(coingecko_api_key=coingecko_api_key)
                result = asyncio.run(calculator.calculate_wallet_pnl(df, wallet_address))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Realized PnL", f"${result['realized_pnl']:,.2f}")
                with col2:
                    st.metric("Unrealized PnL", f"${result['unrealized_pnl']:,.2f}")
                with col3:
                    st.metric("Total PnL", f"${result['total_pnl']:,.2f}")

                st.success(f"Processed {result['processed_transactions']} transactions successfully!")

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    main()
