# app.py
import os
import time
import glob
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from analyzer import ExtendedMoralisAnalyzer, calculate_pnl

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

# -------------------------------
# App
# -------------------------------
def main():
    st.title("💰 Wallet PnL Explorer")
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # Sidebar: controls
    st.sidebar.header("🔧 Controls")
    cache_mode = st.sidebar.radio("Cache Mode", ["Always Use Cache", "Force Refresh", "Disable Cache"], index=0)
    pnl_method = st.sidebar.selectbox("PnL Accounting Method", ["FIFO", "LIFO"], index=0)  # default FIFO (your request)
    wallet_address = st.sidebar.text_input("Wallet Address", value="", help="Leave empty to preview Vitalik by default.")
    selected_chains = st.sidebar.multiselect(
        "Blockchains",
        ["eth", "bsc", "polygon", "arbitrum", "optimism", "base"],
        default=["eth", "arbitrum", "optimism"]
    )
    # Date selectors: for default wallet we’ll override to last 7 days
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

    # Decide wallet and date window
    default_wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    using_default = False
    if analyze_button and wallet_address.strip():
        chosen_wallet = wallet_address.strip()
        window_start = datetime.combine(start_date, datetime.min.time())
        window_end = datetime.combine(end_date, datetime.max.time())
    else:
        # Default Vitalik view → last 7 days & only tokens with prices
        chosen_wallet = default_wallet
        using_default = True
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=2)
        st.info(" Analysis for Vitalik’s:Last 3 days.")

    # Safety: at least one chain selected
    if not selected_chains:
        st.warning("Please select at least one blockchain in the sidebar.")
        st.stop()

    # -------------------------------
    # Fetch with progress feedback
    # -------------------------------
    progress = st.progress(0, text="Preparing analysis...")
    time.sleep(0.1)

    progress.progress(10, text="Checking cache / fetching data...")
    df = get_wallet_data(analyzer, chosen_wallet, selected_chains, max_txs, force_refresh=(cache_mode == "Force Refresh"))
    progress.progress(50, text="Applying filters...")

    if df.empty:
        progress.empty()
        st.error("No transactions found.")
        st.stop()

    # Date filter
    df = df[(df["block_time"] >= pd.to_datetime(window_start)) & (df["block_time"] <= pd.to_datetime(window_end))]

    # Default view: only tokens with prices
    if using_default:
        if "price_usd" in df.columns:
            df = df[df["price_usd"].fillna(0) > 0]

    if df.empty:
        progress.empty()
        st.warning("No transactions found after applying filters.")
        st.stop()

    # Ensure gas USD exists (if analyzer didn’t already)
    if "gas_cost_usd" not in df.columns:
        df["gas_cost_usd"] = df.get("gas_cost_native", pd.Series([0]*len(df))).fillna(0) * df.get("native_price_usd", 0)

    progress.progress(70, text="Computing summaries and PnL...")

    # -------------------------------
    # Summary cards (horizontal)
    # -------------------------------
    total_in = float(df[df["transaction_type"] == "deposit"]["usd_value"].sum())
    total_out = float(df[df["transaction_type"] == "withdrawal"]["usd_value"].sum())
    gas_cost = float(df.get("gas_cost_usd", pd.Series()).fillna(0).sum()) if "gas_cost_usd" in df else 0.0
    pnl = total_in - total_out - gas_cost

    cols_html = (
        card("Total Deposits (USD)", f"${total_in:,.2f}") +
        card("Total Withdrawals (USD)", f"${total_out:,.2f}") +
        card("Gas Costs (USD)", f"${gas_cost:,.2f}") +
        card("Net PnL (USD)", f"${pnl:,.2f}")
    )
    st.markdown("### 📈 Summary")
    st.markdown(f'<div class="card-row">{cols_html}</div>', unsafe_allow_html=True)

    # -------------------------------
    # PnL Analysis (horizontal) – default shows FIFO (your request)
    # -------------------------------
    realized, unrealized, breakdown = calculate_pnl(df, method=pnl_method)
    pnl_cards = (
        card(f"{pnl_method} Realized PnL (USD)", f"${realized:,.2f}") +
        card(f"{pnl_method} Unrealized PnL (USD)", f"${unrealized:,.2f}") +
        card("Positions (Open Lots)", f"{int((breakdown['remaining_amount']>0).sum()) if 'remaining_amount' in breakdown.columns else 0}")
    )
    st.markdown("### 💹 PnL Analysis")
    st.markdown(f'<div class="pnl-row">{pnl_cards}</div>', unsafe_allow_html=True)

    st.dataframe(
        breakdown,
        use_container_width=True,
        height=320
    )

    progress.progress(100, text="Done!")
    time.sleep(0.1)
    progress.empty()

    # -------------------------------
    # Transactions table
    # -------------------------------
    st.subheader("📊 Enriched Transactions")
    st.dataframe(df, use_container_width=True, height=420)

    # Optional: Cache file explorer
    with st.expander("📂 View local cache files"):
        files = glob.glob(os.path.join(_wallet_dir(chosen_wallet), "*.parquet"))
        if files:
            st.write("Cached chains for this wallet:")
            st.write([os.path.basename(f) for f in files])
        else:
            st.write("No cache files yet for this wallet.")

if __name__ == "__main__":
    main()
