# app.py
import os
import time
import glob
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from analyzer import ExtendedMoralisAnalyzer, calculate_pnl_improved, validate_pnl_calculation
from price_fetcher import get_token_price

import numpy as np
import pytz
import random
import logging
import traceback


# Sample Data
def generate_sample_data(n_days=7, txs_per_day=7, wallet_address="0xDEADBEEF1234567890ABCDEF1234567890ABCDEF"):
    np.random.seed(42)  # reproducible
    rows = []
    start_date = datetime.today() - timedelta(days=n_days)

    tokens = [
        ("USDC", 1.0),
        ("USDT", 1.0), 
        ("ETH", 2000.0),
        ("ARB", 3.0),
        ("OP", 3.5),
        ("MATIC", 0.7),
    ]
    chains = ["eth", "arbitrum", "optimism", "polygon"]

    tx_types = ["deposit", "withdrawal", "buy", "sell"]

    for d in range(n_days):
        for _ in range(txs_per_day):
            block_time = start_date + timedelta(days=d, hours=np.random.randint(0, 24))
            token, base_price = tokens[np.random.randint(len(tokens))]
            
            # Generate slightly different current vs historical prices for unrealized PnL demo
            historical_price = round(base_price * np.random.uniform(0.85, 1.15), 2)
            
            amount = round(np.random.uniform(10, 1000), 2) if token in ["USDC", "USDT"] else round(np.random.uniform(0.1, 20), 4)

            tx_type = np.random.choice(tx_types)

            # USD value logic
            if tx_type in ["deposit", "buy"]:
                usd_value = amount * historical_price
            elif tx_type in ["withdrawal", "sell"]:
                usd_value = amount * historical_price * np.random.uniform(0.95, 1.05)
            else:
                usd_value = amount * historical_price

            # fake addresses
            from_addr = f"0x{random.randint(10**15, 10**18):x}"
            to_addr = f"0x{random.randint(10**15, 10**18):x}"

            # Simulate some withdrawals going back to your own wallet
            if tx_type == "withdrawal" and random.random() < 0.3:  # 30% chance
                to_addr = wallet_address
                tx_type = "withdrawal_move"

            rows.append({
                "tx_hash": f"0x{random.randint(10**15, 10**18):x}",
                "block_time": block_time,
                "blockchain": np.random.choice(chains),
                "transaction_type": tx_type,
                "amount": amount,               
                "price_usd": historical_price,             
                "usd_value": usd_value,         
                "gas_cost_usd": round(np.random.uniform(1, 20), 2),
                "token_symbol": token,
                "token_address": f"0x{random.randint(10**15, 10**18):x}",
                "from_address": from_addr,
                "to_address": to_addr,
            })

    return pd.DataFrame(rows)

# Mock current prices for sample data (slightly different from historical for demo)
def get_sample_current_prices():
    """Generate mock current prices that differ from historical prices"""
    return {
        # These would be token addresses in real data, using symbols for demo
        "USDC": 1.00,
        "USDT": 0.999,
        "ETH": 2150.0,  # Higher than historical average
        "ARB": 2.85,    # Lower than historical average  
        "OP": 3.75,     # Higher than historical average
        "MATIC": 0.72,  # Slightly higher
    }

# Create ~50 transactions across a week
sample_df = generate_sample_data(n_days=7, txs_per_day=7)

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="Wallet PnL Explorer", page_icon="💰", layout="wide")
load_dotenv()
API_KEY = os.getenv("MORALIS_API_KEY")
if not API_KEY:
    st.error("⚠️ Please add MORALIS_API_KEY to your .env file!")
    st.stop()

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------
# Disk cache helpers (per-wallet + per-chain)
# -------------------------------
def _wallet_dir(wallet: str):
    return os.path.join(CACHE_DIR, wallet.lower())

def save_to_disk(wallet: str, chain: str, df: pd.DataFrame):
    wdir = _wallet_dir(wallet)
    os.makedirs(wdir, exist_ok=True)
    path = os.path.join(wdir, f"{chain}.parquet")
    df.to_parquet(path, index=False)

def load_from_disk(wallet: str, chain: str):
    path = os.path.join(_wallet_dir(wallet), f"{chain}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

@st.cache_data(show_spinner=False)
def get_wallet_data(_analyzer, wallet: str, chains: list, max_txs: int, force_refresh: bool = False) -> pd.DataFrame:
    """Hybrid memory+disk+API cache. Returns concatenated df for requested chains."""
    dfs = []
    for ch in chains:
        if not force_refresh:
            cached = load_from_disk(wallet, ch)
            if cached is not None:
                dfs.append(cached)
                continue

        # API call for that chain
        try:
            df = _analyzer.get_detailed_data_for_wallet(wallet, max_per_chain=max_txs, chains=[ch])
        except TypeError:
            # Fallback: fetch all and filter by chain
            df_all = _analyzer.get_detailed_data_for_wallet(wallet, max_per_chain=max_txs)
            df = df_all[df_all["blockchain"] == ch] if not df_all.empty else df_all

        if not df.empty:
            save_to_disk(wallet, ch, df)
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def main():
    st.title("Wallet PnL Explorer")
    st.sidebar.header("🔧 Controls")

    # Sidebar controls
    diagnostic_mode = st.sidebar.checkbox("Enable Diagnostic Mode", value=False)
    pnl_method = st.sidebar.selectbox("PnL Accounting Method", ["FIFO", "LIFO", "ACB"], index=0)
    wallet_address = st.sidebar.text_input("Wallet Address", value="", help="Leave empty to preview demo data.")
    selected_chains = st.sidebar.multiselect(
        "Blockchains",
        ["eth", "bsc", "polygon", "arbitrum", "optimism", "base"],
        default=["eth", "arbitrum", "optimism"]
    )
    start_date = st.sidebar.date_input("Start Date", value=(datetime.utcnow() - timedelta(days=30)).date())
    end_date = st.sidebar.date_input("End Date", value=datetime.utcnow().date())
    max_txs = st.sidebar.slider("Max transactions per chain", min_value=10, max_value=200, value=50, step=10)
    cache_mode = st.sidebar.radio("Cache Mode", ["Always Use Cache", "Force Refresh", "Disable Cache"], index=0)
    analyze_button = st.sidebar.button("🔍 Analyze Wallet")

    # Initialize analyzer
    if cache_mode == "Always Use Cache":
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=False)
        force_refresh = False
    elif cache_mode == "Force Refresh":
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=True)
        force_refresh = True
    else:
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=False)
        force_refresh = False

    # Determine wallet mode
    if analyze_button and wallet_address.strip():
        chosen_wallet = wallet_address.strip()
        using_default = False
        window_start = datetime.combine(start_date, datetime.min.time())
        window_end = datetime.combine(end_date, datetime.max.time())
    else:
        chosen_wallet = "sample_wallet"
        using_default = True
        window_start = sample_df["block_time"].min()
        window_end = sample_df["block_time"].max()
        st.info("💡 Sample wallet preview for the past 7 days: Enter your wallet on the left to analyze real data.")

    if not selected_chains:
        st.warning("Please select at least one blockchain in the sidebar.")
        st.stop()

    # Fetch/load wallet data
    if using_default:
        df = sample_df.copy()
    else:
        progress = st.progress(0, text="Preparing analysis...")
        progress.progress(20, text="Checking cache / fetching data...")
        df = get_wallet_data(analyzer, chosen_wallet, selected_chains, max_txs, force_refresh=force_refresh)
        progress.progress(50, text="Applying filters...")

        if df.empty:
            progress.empty()
            st.error("No transactions found for this wallet.")
            st.stop()

        # Ensure UTC datetime
        if df["block_time"].dt.tz is None:
            df["block_time"] = df["block_time"].dt.tz_localize("UTC")
        window_start = pd.Timestamp(window_start).tz_localize("UTC")
        window_end = pd.Timestamp(window_end).tz_localize("UTC")

        # Filter by date window
        df = df[(df["block_time"] >= window_start) & (df["block_time"] <= window_end)]

        # Keep only tokens with valid prices
        df = df[df["price_usd"].notna() & (df["price_usd"] > 0)]

        # Mark withdrawals to your own wallet as moves
        df['transaction_type'] = df.apply(
            lambda row: 'withdrawal_move'
            if row['transaction_type'] == 'withdrawal' and str(row.get('to_address', '')).lower() == chosen_wallet.lower()
            else row['transaction_type'],
            axis=1
        )

        progress.progress(70, text="Computing summaries and PnL...")

    if df.empty:
        st.warning("⚠️ No transactions available after filters.")
        st.stop()

    # -------------------------------
    # Summary metrics
    # -------------------------------
    total_in = float(df[df["transaction_type"] == "deposit"]["usd_value"].sum())
    total_out = float(df[df["transaction_type"] == "withdrawal"]["usd_value"].sum())
    gas_cost = float(df.get("gas_cost_usd", pd.Series()).fillna(0).sum()) if "gas_cost_usd" in df else 0.0
    pnl = total_in - total_out - gas_cost

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deposits (USD)", f"${total_in:,.2f}")
    col2.metric("Total Withdrawals (USD)", f"${total_out:,.2f}")
    col3.metric("Gas Costs (USD)", f"${gas_cost:,.2f}")
    col4.metric("Net Cash Flow (USD)", f"${pnl:,.2f}")

    # -------------------------------
    # PnL calculation with current prices
    # -------------------------------
    realized_total = 0.0
    unrealized_total = 0.0
    total_gas_costs = 0.0  # NEW: Track total gas costs from PnL calculation
    breakdown_list = []

    # Group by token_symbol for sample data, token_address for real data
    group_key = 'token_symbol' if using_default else 'token_address'
    grouped = df.groupby(group_key)

    tokens_with_valid_prices = set()
    tokens_with_missing_prices = set()

    for token_key, group in grouped:
        # Filter group to rows with valid prices
        group_valid = group[group['price_usd'].notna() & (group['price_usd'] > 0)]
        if group_valid.empty:
            tokens_with_missing_prices.add(token_key)
            continue

        tokens_with_valid_prices.add(token_key)

        # Calculate PnL for this token group - PASS ANALYZER for current prices
        if using_default:
            # For sample data, create a mock analyzer that returns sample current prices
            class MockAnalyzer:
                def get_current_prices(self, tokens):
                    sample_prices = get_sample_current_prices()
                    result = {}
                    for token in tokens:
                        symbol = token.get('symbol', '')
                        result[token.get('address', symbol)] = sample_prices.get(symbol, 0)
                    return result
            
            mock_analyzer = MockAnalyzer()
            # UPDATED: Unpack 4 values instead of 3
            realized, unrealized, gas_costs, breakdown = calculate_pnl_improved(group_valid, method=pnl_method, analyzer=mock_analyzer)
        else:
            # UPDATED: Unpack 4 values instead of 3
            realized, unrealized, gas_costs, breakdown = calculate_pnl_improved(group_valid, method=pnl_method, analyzer=analyzer)

        realized_total += realized
        unrealized_total += unrealized
        total_gas_costs += gas_costs  # NEW: Accumulate gas costs
        breakdown_list.append(breakdown)

    # Combine breakdowns into one DataFrame
    if breakdown_list:
        breakdown_df = pd.concat(breakdown_list, ignore_index=True)
    else:
        breakdown_df = pd.DataFrame()

    if tokens_with_missing_prices:
        st.warning(f"Tokens excluded from PnL due to missing prices: {tokens_with_missing_prices}")

    # -------------------------------
    # PnL Validation - UPDATED to include gas costs
    # -------------------------------
    validation_df = validate_pnl_calculation(
        df[df[group_key].isin(tokens_with_valid_prices)], 
        realized_total, 
        unrealized_total, 
        total_gas_costs,  # NEW: Pass gas costs
        breakdown_df
    )
    failed_validations = validation_df[validation_df['Pass'] == False]

    if not failed_validations.empty:
        st.warning("⚠️ PnL Validation Issues Detected")
        with st.expander("View Validation Details", expanded=True):
            st.dataframe(validation_df, use_container_width=True)
            st.write("**Issues found:**")
            for _, row in failed_validations.iterrows():
                st.write(f"- {row['Check']}: Failed")
    else:
        st.success("✅ PnL Calculations Validated Successfully")
        with st.expander("View Validation Details"):
            st.dataframe(validation_df, use_container_width=True)
  
    # UPDATED: Display gas costs and net PnL metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{pnl_method} Realized PnL (USD)", f"${realized_total:,.2f}")
    col2.metric(f"{pnl_method} Unrealized PnL (USD)", f"${unrealized_total:,.2f}")
    col3.metric("Total Gas Costs (PnL)", f"${total_gas_costs:,.2f}")
    col4.metric("Net PnL (After Gas)", f"${realized_total + unrealized_total - total_gas_costs:,.2f}")
    
    # Count open positions
    open_positions = len(breakdown_df[breakdown_df['Current Holdings'] > 0]) if not breakdown_df.empty else 0
    st.metric("Open Positions", f"{open_positions}")

    st.subheader("💹 PnL Breakdown by Token")
    if not breakdown_df.empty:
        st.dataframe(breakdown_df, use_container_width=True, height=320)
    else:
        st.info("No PnL data available.")
     
    # -------------------------------
    # Transactions table
    # -------------------------------
    st.subheader("📊 Enriched Transactions")
    st.dataframe(df, use_container_width=True, height=420)

    # Price validation and cleanup
    from price_fetcher import get_token_price

    # Replace invalid prices with historical fetch
    df["price_usd"] = df.apply(
        lambda row: row["price_usd"]
        if row["price_usd"] > 0
        else (
            get_token_price(
                row["token_symbol"],
                row.get("token_address"),
                row["blockchain"],
                block_time=row["block_time"]
            ) or 0
        ),
        axis=1
    )

    # Drop unsupported tokens (no price found)
    df = df[df["price_usd"] > 0]

    if diagnostic_mode:
        st.subheader("🔍 Diagnostic Report")
        # UPDATED: Pass gas costs to validation
        validation_df = validate_pnl_calculation(df, realized_total, unrealized_total, total_gas_costs, breakdown_df)
        st.dataframe(validation_df, use_container_width=True)

        st.subheader("📊 First 10 Transactions")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📊 PnL Breakdown")
        st.dataframe(breakdown_df, use_container_width=True)

    if not using_default:
        with st.expander("📂 View local cache files"):
            files = glob.glob(os.path.join(CACHE_DIR, chosen_wallet.lower(), "*.parquet"))
            st.write([os.path.basename(f) for f in files]) if files else st.write("No cache files yet.")

        progress.progress(100, text="Done!")
        time.sleep(0.1)
        progress.empty()
      
if __name__ == "__main__":
    main()