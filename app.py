# app.py
import os
import time
import glob
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from analyzer import ExtendedMoralisAnalyzer, calculate_pnl

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------------
# Synthetic extended demo data
# -------------------------------
def generate_sample_data(n_days=7, txs_per_day=7):
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

    for d in range(n_days):
        for t in range(txs_per_day):
            block_time = start_date + timedelta(days=d, hours=np.random.randint(0, 24))
            token, base_price = tokens[np.random.randint(len(tokens))]
            price = round(base_price * np.random.uniform(0.9, 1.1), 2)  # small price variation
            amount = round(np.random.uniform(10, 1000), 2) if token in ["USDC", "USDT"] else round(np.random.uniform(0.1, 20), 4)

            tx_type = np.random.choice(["deposit", "withdrawal"])
            usd_value = amount * price * (1 if tx_type == "deposit" else -1)

            rows.append({
                "block_time": block_time,
                "blockchain": np.random.choice(chains),
                "transaction_type": tx_type,
                "usd_value": abs(usd_value),
                "gas_cost_usd": round(np.random.uniform(1, 20), 2),
                "token_symbol": token,
                "token_amount": amount,
                "token_price_usd": price,
            })

    return pd.DataFrame(rows)

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
        # If your analyzer does not support 'chains' arg, fetch all then filter
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

# -------------------------------
# Styles (cards layout)
# -------------------------------
CARD_CSS = """
<style>
.card-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(180px, 1fr));
  gap: 14px;
  margin: 8px 0 22px 0;
}
.card {
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  background: white;
  border: 1px solid rgba(0,0,0,0.06);
}
.card h4 {
  margin: 0 0 6px 0;
  font-size: 0.9rem;
  color: #666;
  font-weight: 500;
}
.card .value {
  font-size: 1.2rem;
  font-weight: 700;
}
.pnl-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(180px, 1fr));
  gap: 14px;
  margin: 8px 0 12px 0;
}
</style>
"""

def card(label: str, value: str) -> str:
    return f"""
    <div class="card">
      <h4>{label}</h4>
      <div class="value">{value}</div>
    </div>
    """

def main():
    st.title("💰 Wallet PnL Explorer")
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # Sidebar: controls
    st.sidebar.header("🔧 Controls")
    cache_mode = st.sidebar.radio("Cache Mode", ["Always Use Cache", "Force Refresh", "Disable Cache"], index=0)
    pnl_method = st.sidebar.selectbox("PnL Accounting Method", ["FIFO", "LIFO","ACB"], index=0)  # default FIFO
    wallet_address = st.sidebar.text_input("Wallet Address", value="", help="Leave empty to preview demo data.")
    selected_chains = st.sidebar.multiselect(
        "Blockchains",
        ["eth", "bsc", "polygon", "arbitrum", "optimism", "base"],
        default=["eth", "arbitrum", "optimism"]
    )
    start_date = st.sidebar.date_input("Start Date", value=(datetime.utcnow() - timedelta(days=30)).date())
    end_date = st.sidebar.date_input("End Date", value=datetime.utcnow().date())
    max_txs = st.sidebar.slider("Max transactions per chain", min_value=10, max_value=200, value=50, step=10)
    analyze_button = st.sidebar.button("🔍 Analyze Wallet")

    # Initialize analyzer per cache mode
    if cache_mode == "Always Use Cache":
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=False)
        force_refresh = False
    elif cache_mode == "Force Refresh":
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=True)
        force_refresh = True
    else:
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=False)
        force_refresh = False

    # -------------------------------
    # Wallet selection
    # -------------------------------
    if analyze_button and wallet_address.strip():
        # Real wallet path
        chosen_wallet = wallet_address.strip()
        using_default = False
        window_start = datetime.combine(start_date, datetime.min.time())
        window_end = datetime.combine(end_date, datetime.max.time())
    else:
        # Demo mode
        chosen_wallet = "sample_wallet"
        using_default = True
        window_start = sample_df["block_time"].min()
        window_end = sample_df["block_time"].max()
        st.info("💡 Showing sample wallet preview. Enter your wallet on the left to analyze real data.")

    if not selected_chains:
        st.warning("Please select at least one blockchain in the sidebar.")
        st.stop()

    # -------------------------------
    # Fetch or load data
    # -------------------------------
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

        # Date filter
        df = df[(df["block_time"] >= pd.to_datetime(window_start)) & (df["block_time"] <= pd.to_datetime(window_end))]

        progress.progress(70, text="Computing summaries and PnL...")

    if df.empty:
        st.warning("⚠️ No transactions available after filters.")
        st.stop()

    # -------------------------------
    # Summary cards (NEW st.metric version)
    # -------------------------------
    total_in = float(df[df["transaction_type"] == "deposit"]["usd_value"].sum())
    total_out = float(df[df["transaction_type"] == "withdrawal"]["usd_value"].sum())
    gas_cost = float(df.get("gas_cost_usd", pd.Series()).fillna(0).sum()) if "gas_cost_usd" in df else 0.0
    pnl = total_in - total_out - gas_cost

    st.markdown("### 📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deposits (USD)", f"${total_in:,.2f}")
    col2.metric("Total Withdrawals (USD)", f"${total_out:,.2f}")
    col3.metric("Gas Costs (USD)", f"${gas_cost:,.2f}")
    col4.metric("Net PnL (USD)", f"${pnl:,.2f}")

    # -------------------------------
    # PnL Analysis (NEW st.metric version)
    # -------------------------------
    realized, unrealized, breakdown = calculate_pnl(df, method=pnl_method)

    st.markdown("### 💹 PnL Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{pnl_method} Realized PnL (USD)", f"${realized:,.2f}")
    col2.metric(f"{pnl_method} Unrealized PnL (USD)", f"${unrealized:,.2f}")
    open_positions = int((breakdown["remaining_amount"] > 0).sum()) if "remaining_amount" in breakdown.columns else 0
    col3.metric("Positions (Open Lots)", f"{open_positions}")

    # -------------------------------
    # Transactions table
    # -------------------------------
    st.subheader("📊 Enriched Transactions")
    st.dataframe(df, use_container_width=True, height=420)

    if not using_default:
        with st.expander("📂 View local cache files"):
            files = glob.glob(os.path.join(_wallet_dir(chosen_wallet), "*.parquet"))
            st.write([os.path.basename(f) for f in files]) if files else st.write("No cache files yet.")

    if not using_default:
        progress.progress(100, text="Done!")
        time.sleep(0.1)
        progress.empty()

if __name__ == "__main__":
    main()
