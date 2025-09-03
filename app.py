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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wallet-pnl")

def get_secret(key: str, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

DUNE_API_KEY = get_secret("DUNE_CLIENT_API", DUNE_API_KEY)
COINGECKO_API_KEY = get_secret("GECKO_API_KEY", COINGECKO_API_KEY)


class CachedPnLCalculator:
    NATIVE_TOKEN_MAP = {
        'ethereum': {
            '0x0000000000000000000000000000000000000000': 'ethereum',
            '0xC02aaa39b223FE8D0A0e5c4f27ead9083c756cc2': 'weth',
            '0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'usd-coin',
            '0xdac17f958d2ee523a2206206994597c13d831ec7': 'tether'
        },
        'optimism': {
            '0x4200000000000000000000000000000000000006': 'weth',
            '0x7f5c764cbc14f9669b88837ca1490cca17c31607': 'usd-coin',
            '0x94b008aa00579c1307b0ef2c499ad98a8ce58e58': 'tether'
        },
        'arbitrum': {
            '0x82af49447d8a07e3bd95bd0d56f35241523fbab1': 'ethereum',
            '0xff970a61a04b1ca14834a43f5de4533ebddb5cc8': 'usd-coin',
            '0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9': 'tether'
        },
        'bnb': {
            '0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c': 'wbnb',
            '0x55d398326f99059ff775485246999027b3197955': 'tether',
            '0xe9e7cea3dedca5984780bafc599bd69add087d56': 'busd'
        }
    }
    def __init__(self):
        # your init code here
        pass
    async def calculate_cached_pnl(self, df: pd.DataFrame, wallet_address: str):
        # your async PnL calculation code here
        return {
            'wallet_address': wallet_address,
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'total_pnl': 0,
            'remaining_tokens': 0,
            'processed_transactions': len(df)
        }
    def __init__(self, coingecko_api_key=None):
        self.coingecko_api_key = coingecko_api_key or COINGECKO_API_KEY
        self.coingecko_cache = {}
        self.token_mapping_cache = {}
        self.session = None

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    def _headers(self):
        return {'x-cg-pro-api-key': self.coingecko_api_key} if self.coingecko_api_key else {}

    async def get_coingecko_id(self, token_address: str, blockchain: str, token_symbol: str = None):
        token_address = token_address.lower()
        blockchain = blockchain.lower()
        if blockchain in self.NATIVE_TOKEN_MAP and token_address in self.NATIVE_TOKEN_MAP[blockchain]:
            return self.NATIVE_TOKEN_MAP[blockchain][token_address]

        cache_key = f"{blockchain}_{token_address}"
        if cache_key in self.token_mapping_cache:
            return self.token_mapping_cache[cache_key]

        platform_map = {
            'ethereum': 'ethereum',
            'optimism': 'optimism',
            'arbitrum': 'arbitrum-one',
            'bnb': 'binance-smart-chain',
            'base': 'base'
        }
        if blockchain not in platform_map:
            return None
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{platform_map[blockchain]}/contract/{token_address}"
            async with self.session.get(url, headers=self._headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    cg_id = data.get('id')
                    if cg_id:
                        self.token_mapping_cache[cache_key] = cg_id
                        return cg_id
        except Exception as e:
            logger.warning(f"CG lookup failed for {token_address} on {blockchain}: {e}")
        if token_symbol:
            return token_symbol.lower()
        return None

    async def get_historical_price(self, coingecko_id: str, timestamp: datetime):
        cache_key = f"{coingecko_id}_{timestamp.date()}"
        if cache_key in self.coingecko_cache:
            return self.coingecko_cache[cache_key]
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/history"
            params = {'date': timestamp.strftime('%d-%m-%Y'), 'localization': 'false'}
            async with self.session.get(url, params=params, headers=self._headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = data.get('market_data', {}).get('current_price', {}).get('usd')
                    if price is not None:
                        self.coingecko_cache[cache_key] = price
                        await asyncio.sleep(0.2)  # rate limit friendly
                        return price
        except Exception as e:
            logger.warning(f"Failed historical price fetch {coingecko_id}: {e}")
        return None
async def calculate_cached_pnl(self, df: pd.DataFrame, wallet_address: str):
    await self.init_session()
    try:
        df = df.copy()
        df['block_time'] = pd.to_datetime(df['block_time'])
        df = df.sort_values('block_time')
        inventory = {}
        realized_pnl = 0
        deposits_processed = 0
        withdrawals_processed = 0
        skipped_due_to_missing_price = 0

        processed_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in df.iterrows():
            action = str(row.get('action', '')).lower()
            blockchain = str(row.get('blockchain', '')).lower()
            tx_time = row['block_time']

            token_in_address = row.get('token_in_address')
            token_out_address = row.get('token_out_address')
            token_in_symbol = row.get('token_in_symbol')
            token_out_symbol = row.get('token_out_symbol')

            # Debug: show current transaction info
            # st.write(f"Tx {idx}: action={action}, token_in={token_in_address}, token_out={token_out_address}")

            if action == 'deposit' and pd.notna(token_in_address):
                deposits_processed += 1
                amount = float(row['token_in_amount'])
                coingecko_id = await self.get_coingecko_id(token_in_address, blockchain, token_in_symbol)
                if coingecko_id is None:
                    skipped_due_to_missing_price += 1
                    processed_count += 1
                    progress_bar.progress(processed_count / len(df))
                    status_text.text(f"Processed {processed_count}/{len(df)} transactions")
                    continue
                price = await self.get_historical_price(coingecko_id, tx_time)
                if price is None:
                    skipped_due_to_missing_price += 1
                    processed_count += 1
                    progress_bar.progress(processed_count / len(df))
                    status_text.text(f"Processed {processed_count}/{len(df)} transactions")
                    continue
                gas_cost = float(row.get('gas_usd', 0)) or 0
                effective_cost = price + (gas_cost / amount if amount else 0)
                inventory.setdefault(token_in_address.lower(), []).append((amount, effective_cost, tx_time))

            elif action == 'withdrawal' and pd.notna(token_out_address):
                withdrawals_processed += 1
                amount = float(row['token_out_amount'])
                coingecko_id = await self.get_coingecko_id(token_out_address, blockchain, token_out_symbol)
                if coingecko_id is None:
                    skipped_due_to_missing_price += 1
                    processed_count += 1
                    progress_bar.progress(processed_count / len(df))
                    status_text.text(f"Processed {processed_count}/{len(df)} transactions")
                    continue
                current_price = await self.get_historical_price(coingecko_id, tx_time)
                if current_price is None:
                    skipped_due_to_missing_price += 1
                    processed_count += 1
                    progress_bar.progress(processed_count / len(df))
                    status_text.text(f"Processed {processed_count}/{len(df)} transactions")
                    continue
                token_addr_lower = token_out_address.lower()
                if token_addr_lower in inventory and inventory[token_addr_lower]:
                    remaining = amount
                    while remaining > 0 and inventory[token_addr_lower]:
                        lot_amount, lot_cost, _ = inventory[token_addr_lower][0]
                        take = min(lot_amount, remaining)
                        proceeds = take * current_price
                        cost = take * lot_cost
                        realized_pnl += proceeds - cost
                        remaining -= take
                        if take == lot_amount:
                            inventory[token_addr_lower].pop(0)
                        else:
                            inventory[token_addr_lower][0] = (lot_amount - take, lot_cost, _)

            processed_count += 1
            progress_bar.progress(processed_count / len(df))
            status_text.text(f"Processed {processed_count}/{len(df)} transactions")

        st.write(f"Deposits processed: {deposits_processed}")
        st.write(f"Withdrawals processed: {withdrawals_processed}")
        st.write(f"Transactions skipped due to missing price or token info: {skipped_due_to_missing_price}")

        # Calculate unrealized PnL
        unrealized_pnl = 0
        for token_addr, lots in inventory.items():
            coingecko_id = await self.get_coingecko_id(token_addr, 'ethereum')
            if coingecko_id:
                current_price = await self.get_historical_price(coingecko_id, datetime.now())
                if current_price:
                    for amt, cost, _ in lots:
                        unrealized_pnl += amt * (current_price - cost)

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

def run_async(coro):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


def main():
    st.set_page_config(page_title="Wallet PnL Explorer", layout="wide")
    st.title("💰 Wallet PnL Explorer")

    wallet_address = st.text_input("Wallet Address", value="0x...")
    blockchains = st.multiselect(
        "Select Blockchains",
        ["ethereum", "bnb", "arbitrum", "optimism", "base"],
        default=["ethereum"]
    )

    start_date_input = st.date_input("Start Date", datetime(2023, 1, 1))
    end_date_input = st.date_input("End Date", datetime.now().date())
    start_date = datetime.combine(start_date_input, datetime.min.time())
    end_date = datetime.combine(end_date_input, datetime.max.time())

    query_mode = st.radio("Query Mode", ["Cached", "Fresh"], index=0)

    if st.button("Run Analysis"):
        if not DUNE_API_KEY:
            st.error("DUNE_API_KEY not configured")
            return

        if not wallet_address or wallet_address == "0x...":
            st.error("Please enter a valid wallet address")
            return

        dune = DuneClient(DUNE_API_KEY)
        with st.spinner("Fetching transactions from Dune..."):
            try:
                if query_mode == "Cached":
                    result_obj = dune.get_latest_result(DUNE_QUERY_ID)
                    df = pd.DataFrame(result_obj.result.rows)
                else:
                    st.warning("Fresh mode temporarily disabled for this demo")
                    return

                if df.empty:
                    st.warning("No transactions found")
                    return

                # Filter by wallet address using the new 'wallet' column
                if 'wallet' in df.columns:
                    df = df[df['wallet'].str.lower() == wallet_address.lower()]
                else:
                    st.warning("Wallet column not found in data; skipping wallet filter")

                # Filter by blockchains if column exists
                if 'blockchain' in df.columns:
                    df = df[df['blockchain'].str.lower().isin([b.lower() for b in blockchains])]
                else:
                    st.warning("Blockchain column not found in data; skipping blockchain filter")

                if df.empty:
                    st.warning("No transactions found after filtering")
                    return

                st.write("Preview of filtered transactions:")
                st.dataframe(df.head())

                calculator = CachedPnLCalculator()
                pnl_result = run_async(calculator.calculate_cached_pnl(df, wallet_address))

                col1, col2, col3 = st.columns(3)
                col1.metric("Realized PnL", f"${pnl_result['realized_pnl']:,.2f}")
                col2.metric("Unrealized PnL", f"${pnl_result['unrealized_pnl']:,.2f}")
                col3.metric("Total PnL", f"${pnl_result['total_pnl']:,.2f}")

                st.success(f"Processed {pnl_result['processed_transactions']} transactions")
            except Exception as e:
                st.error(f"Error fetching data: {e}")


if __name__ == "__main__":
    main()


