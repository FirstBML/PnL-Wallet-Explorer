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
from price_fetcher import fill_missing_prices_batch  
import hashlib


import numpy as np
import pytz
import random
import logging
import traceback
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import traceback

# Custom CSS loader
st.markdown("""
<style>
/* Base styles for the entire app */
.stApp {
    background-color: #0E1117;
}

/* Enhanced text visibility for the main content */
.stMarkdown, .stText, .stDataFrame {
    color: #F5F5F5 !important;   /* brighter than #E6E6E6 */
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* Improved heading visibility */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Enhanced sidebar styling */
[data-testid=stSidebar] {
    background-color: #1E1E1E;
    padding: 2rem 1rem;
    box-shadow: 2px 0 5px rgba(0,0,0,0.2);
}

/* Improved sidebar text visibility */
[data-testid=stSidebar] [data-testid=stMarkdown] p {
    color: #F8F8F8 !important;   /* brighter than #FFFFFF (slight off-white) */
    font-size: 15px !important;
    padding: 0.5rem 0;
    line-height: 1.6;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

/* Enhanced label visibility for inputs */
[data-testid=stSidebar] .stSelectbox label,
[data-testid=stSidebar] .stTextInput label,
[data-testid=stSidebar] .stSlider label,
[data-testid=stSidebar] .stMultiSelect label {
    color: #FAFAFA !important;   /* brighter than pure white */
    font-weight: 500 !important;
    font-size: 16px !important;
    margin-bottom: 8px !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* Improved input field styling */
[data-testid=stSidebar] .stTextInput input,
[data-testid=stSidebar] .stSelectbox div[data-baseweb="select"] {
    background-color: #2A2A2A !important;
    color: #FFFFFF !important;
    border: 1px solid #505050 !important;
    font-size: 15px !important;
}

/* Enhanced multiselect styling */
[data-testid=stSidebar] .stMultiSelect div[role="combobox"] {
    background-color: #2A2A2A !important;
    color: #FFFFFF !important;
    border: 1px solid #505050 !important;
}

/* Improved slider visibility */
[data-testid=stSidebar] .stSlider div[data-baseweb="slider"] span {
    background-color: #4A90E2 !important;  /* brighter blue handle */
}

[data-testid=stSidebar] .stSlider [data-testid="stThumbValue"] {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Enhanced button styling */
[data-testid=stSidebar] .stButton button {
    background-color: #2E7DAF !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 0.75rem !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* Main content metrics and cards */
[data-testid="stMetricValue"] {
    color: #FFD700 !important;   /* gold for high visibility */
    font-size: 18px !important;
    font-weight: 600 !important;
}

.custom-card h2, .custom-card h3 {
    color: #FFFFFF !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* Table styling for better visibility */
.dataframe {
    color: #FFFFFF !important;
}

.dataframe th {
    background-color: #1E1E1E !important;
    color: #FFD700 !important;   /* gold headers for better contrast */
    font-weight: 600 !important;
}

/* Alert and info messages */
.stAlert {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}

/* Caption styling */
.caption {
    color: #CCCCCC !important;   /* lighter than #B0B0B0 */
    font-size: 14px !important;
}

/* Force all checkbox, radio, and toggle labels to be bright */
.stCheckbox label,
.stRadio label,
.stToggle label {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Fix disabled/greyed-out labels (like Diagnostic Mode) */
.stCheckbox [data-testid="stMarkdownContainer"],
.stRadio [data-testid="stMarkdownContainer"] {
    color: #CCCCCC !important; /* light grey instead of invisible */
}

/* Ensure selectbox and slider labels are visible */
.stSelectbox label,
.stTextInput label,
.stSlider label,
.stMultiSelect label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* --- Enhanced alert and diagnostic message styles --- */
div.stAlert {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #E8F5E8 !important;  /* Light green text for better visibility */
    background-color: rgba(46, 125, 50, 0.15) !important; /* Green tint background */
    border: 2px solid #2E7D32 !important; /* Single green border */
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
    padding: 0.75rem 1rem !important;
    border-radius: 6px !important;
    margin-bottom: 1rem !important;
}

/* Warning messages - Green/yellow variation */
div.stAlert.stAlertWarning {
    color: #FFF3CD !important;           /* Light yellow text */
    background-color: rgba(255, 193, 7, 0.15) !important; /* Amber tint */
    border-color: #FFC107 !important;    /* Amber border */
}

/* Error messages - Green/red variation */
div.stAlert.stAlertError {
    color: #F8D7DA !important;            /* Light red text */
    background-color: rgba(220, 53, 69, 0.15) !important; /* Red tint */
    border-color: #DC3545 !important;     /* Red border */
}

/* Info messages - Green/blue variation */
div.stAlert.stAlertInfo {
    color: #D1ECF1 !important;            /* Light blue text */
    background-color: rgba(23, 162, 184, 0.15) !important; /* Blue tint */
    border-color: #17A2B8 !important;     /* Blue border */
}

/* Success messages (if needed) - Bright green */
div.stAlert.stAlertSuccess {
    color: #D4EDDA !important;
    background-color: rgba(40, 167, 69, 0.15) !important;
    border-color: #28A745 !important;
}

/* Text inside alerts */
div.stAlert > div {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: inherit !important; /* Inherit color from parent */
}

/* Ensure text is visible on all alert types */
div.stAlert .stAlertContent {
    color: inherit !important;
}

/* Hover effects for better interactivity */
div.stAlert:hover {
    background-color: rgba(46, 125, 50, 0.25) !important;
}

div.stAlert.stAlertWarning:hover {
    background-color: rgba(255, 193, 7, 0.25) !important;
}

div.stAlert.stAlertError:hover {
    background-color: rgba(220, 53, 69, 0.25) !important;
}

div.stAlert.stAlertInfo:hover {
    background-color: rgba(23, 162, 184, 0.25) !important;
}

/* Spinner text */
[data-testid="stSpinner"] > div {
    color: #FFD700 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
}

/* Navigation Tabs - Increased Font Size */
div[data-testid="stTabs"] div[role="tablist"] {
    font-size: 24px !important;
}

div[data-testid="stTabs"] div[role="tablist"] button[role="tab"],
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"] *,
div[data-testid="stTabs"] div[role="tablist"] div[role="tab"],
div[data-testid="stTabs"] div[role="tablist"] div[role="tab"] * {
    font-size: 24px !important;
    color: #FFD700 !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.25rem !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    transition: all 0.3s ease;
}

div[data-testid="stTabs"] div[role="tablist"] button[role="tab"][aria-selected="true"],
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"][aria-selected="true"] *,
div[data-testid="stTabs"] div[role="tablist"] div[role="tab"][aria-selected="true"],
div[data-testid="stTabs"] div[role="tablist"] div[role="tab"][aria-selected="true"] * {
    color: #00FFFF !important;
    border-bottom: 3px solid #00FFFF !important;
    font-weight: 700 !important;
    font-size: 25px !important;
}

div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:hover,
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:hover *,
div[data-testid="stTabs"] div[role="tablist"] div[role="tab"]:hover,
div[data-testid="stTabs"] div[role="tablist"] div[role="tab"]:hover * {
    color: #FFA500 !important;
    cursor: pointer;
    font-size: 25px !important;
}
                        
/* --- Sidebar alert styles for consistency --- */
[data-testid=stSidebar] .stAlert {
    background-color: rgba(255, 255, 255, 0.15) !important;
    border: 1.5px solid #FFD700 !important;
    color: #FFD700 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    padding: 0.5rem 1rem !important;
    margin-bottom: 1rem !important;
    border-radius: 6px !important;
}


/* --- Enhanced Python Error Message Styles --- */
div.stException {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #FF6B6B !important;  /* Bright red for error text */
    background-color: rgba(220, 53, 69, 0.15) !important;
    border: 2px solid #DC3545 !important;
    padding: 1rem !important;
    border-radius: 6px !important;
    margin: 1rem 0 !important;
}

/* Error message title */
div.stException > div:first-child {
    color: #FF4757 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
}

/* Error type (KeyError, ValueError, etc.) */
div.stException > div:first-child > span {
    color: #FF4757 !important;
    font-weight: 800 !important;
}

/* Error message content */
div.stException > div:nth-child(2) {
    color: #FF6B6B !important;
    background-color: rgba(0, 0, 0, 0.3) !important;
    padding: 0.75rem !important;
    border-radius: 4px !important;
    border-left: 3px solid #FF4757 !important;
    margin: 0.5rem 0 !important;
}

/* Traceback section */
div.stException > div:last-child {
    color: #FFA8A8 !important;
    background-color: rgba(0, 0, 0, 0.4) !important;
    padding: 1rem !important;
    border-radius: 4px !important;
    border: 1px solid #FF4757 !important;
    margin-top: 0.75rem !important;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* File paths in traceback */
div.stException > div:last-child span[style*="color: rgb(136, 136, 136)"] {
    color: #A8A8A8 !important;
    font-style: italic !important;
}

/* Line numbers and code references */
div.stException > div:last-child span[style*="color: rgb(128, 128, 128)"] {
    color: #94A3B8 !important;
}

/* Code snippets */
div.stException > div:last-child span[style*="color: rgb(0, 0, 0)"] {
    color: #E2E8F0 !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
    padding: 2px 4px !important;
    border-radius: 3px !important;
}

/* Hover effect for better visibility */
div.stException:hover {
    background-color: rgba(220, 53, 69, 0.25) !important;
    border-color: #FF4757 !important;
}

/* Make sure all text is visible */
div.stException * {
    color: inherit !important;
    text-shadow: 1px 1px 1px rgba(0,0,0,0.6) !important;
}

/* Specific styling for the KeyError example you provided */
div.stException:has(span:contains("KeyError")) {
    border-left: 4px solid #FF4757 !important;
}

div.stException:has(span:contains("KeyError")) > div:first-child {
    color: #FF6B6B !important;
}
            
</style>
""", unsafe_allow_html=True)


# Set page config first
st.set_page_config(
    page_title="Wallet PnL Explorer", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample Data
def generate_sample_data(n_days=7, txs_per_day=7, wallet_address="0xDEADBEEF1234567890ABCDEF1234567890ABCDEF"):
    # Your sample data generation code here
    np.random.seed(42)  # reproducible
    rows = []
    start_date = datetime.today() - timedelta(days=n_days)

    tokens = [
        ("ADA", 0.75),
        ("XRP", 2.5), 
        ("ETH", 2000.0),
        ("ARB", 3.0),
        ("OP", 3.5),
        ("MATIC", 0.7)
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
    pass
# Mock current prices for sample data (slightly different from historical for demo)
def get_sample_current_prices():
    """Generate mock current prices that differ from historical prices"""
    return {
        # These would be token addresses in real data, using symbols for demo
        "ADA": 0.90,
        "XRP": 2.8,
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

load_dotenv()

# Try to get from Streamlit secrets first, then environment variables
try:
    API_KEY = st.secrets["MORALIS_API_KEY"]
except (KeyError, FileNotFoundError):
    # Fallback to environment variable for local development
    API_KEY = os.getenv("MORALIS_API_KEY")

if not API_KEY:
    st.error("⚠️ Please add MORALIS_API_KEY to your Streamlit secrets!")
    st.info("For Streamlit Cloud: Go to Settings > Secrets and add your API key")
    st.info("For local development: Add MORALIS_API_KEY to your .env file")
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


# Replace your get_wallet_data function with this corrected version:

def get_wallet_data(_analyzer, wallet: str, chains: list, max_txs: int, force_refresh: bool = False) -> pd.DataFrame:
    """Hybrid memory+disk+API cache. Returns concatenated df for requested chains."""
    dfs = []
    for ch in chains:
        if not force_refresh:
            cached = load_from_disk(wallet, ch)
            if cached is not None:
                st.info(f"Using cached data for {ch} (use Force Refresh to update)")
                dfs.append(cached)
                continue

        # API call for that chain
        try:
            with st.spinner(f"Fetching data from {ch} via Moralis API..."):
                df = _analyzer.get_detailed_data_for_wallet(wallet, max_per_chain=max_txs, chains=[ch])
                if df.empty:
                    st.warning(f"No transactions found for {ch}")
                else:
                    st.success(f"Successfully fetched {len(df)} transactions from {ch}")
        except Exception as e:
            st.error(f"Error fetching data from {ch}: {str(e)}")
            # Fallback: try to fetch all and filter by chain
            try:
                with st.spinner(f"Trying alternative method for {ch}..."):
                    df_all = _analyzer.get_detailed_data_for_wallet(wallet, max_per_chain=max_txs)
                    df = df_all[df_all["blockchain"] == ch] if not df_all.empty else df_all
                    if not df.empty:
                        st.info(f"Found {len(df)} transactions for {ch} using alternative method")
                    else:
                        st.warning(f"No transactions found for {ch} using alternative method")
            except Exception as fallback_error:
                st.error(f"Failed to fetch data for {ch}: {str(fallback_error)}")
                continue

        if not df.empty:
            save_to_disk(wallet, ch, df)
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()
# -------------------------------
# Navigation setup
# -------------------------------
def setup_navigation():
    # Create a nice header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <span style="font-size: 4.5rem;">💰</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.title("Wallet PnL Explorer")
    
    # Create navigation tabs
    tabs = st.tabs(["📊 Dashboard", "💹 PnL Analysis", "📈 Transactions", "⚙️ Settings"])
    
    return tabs

# -------------------------------
# Sidebar controls (shared across pages)
# -------------------------------
def setup_sidebar():
    st.sidebar.header("🔧 Controls")
    st.sidebar.markdown("---")
    
    diagnostic_mode = st.sidebar.checkbox("Enable Diagnostic Mode", value=False)
    pnl_method = st.sidebar.selectbox("PnL Accounting Method", ["FIFO", "LIFO", "ACB"], index=0)
    wallet_address = st.sidebar.text_input("Wallet Address", value="", help="Leave empty to preview demo data.")
    
    # Chain selection
    chain_options = {
        "Ethereum": "eth",
        "Binance Smart Chain": "bsc", 
        "Polygon": "polygon",
        "Arbitrum": "arbitrum",
        "Optimism": "optimism",
        "Base": "base"
    }
    
    selected_chain_names = st.sidebar.multiselect(
        "Blockchains",
        list(chain_options.keys()),
        default=["Ethereum", "Arbitrum", "Optimism"]
    )
    
    selected_chains = [chain_options[name] for name in selected_chain_names]
    
    start_date = st.sidebar.date_input("Start Date", value=(datetime.utcnow() - timedelta(days=30)).date())
    end_date = st.sidebar.date_input("End Date", value=datetime.utcnow().date())
    max_txs = st.sidebar.slider("Max transactions per chain", min_value=10, max_value=200, value=50, step=10)
    
    cache_mode = st.sidebar.radio("Cache Mode", 
        ["Always Use Cache", "Force Refresh", "Disable Cache"], 
        index=0,
        help="Always Use Cache: Uses cached data if available. Force Refresh: Fetches new data from API. Disable Cache: Doesn't use cache at all.")
    
    analyze_button = st.sidebar.button("🔍 Analyze Wallet", use_container_width=True)
    
    # Only show manual cache clear in diagnostic mode
    if diagnostic_mode:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ Debug Tools")
        if st.sidebar.button("🗑️ Clear All Cache (Debug)", use_container_width=True):
            clear_all_transaction_cache()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #94a3b8;">
        <p>Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    return {
        "diagnostic_mode": diagnostic_mode,
        "pnl_method": pnl_method,
        "wallet_address": wallet_address,
        "selected_chains": selected_chains,
        "selected_chain_names": selected_chain_names,
        "start_date": start_date,
        "end_date": end_date,
        "max_txs": max_txs,
        "cache_mode": cache_mode,
        "analyze_button": analyze_button
    }
# -------------------------------
# Custom UI components
# -------------------------------
def custom_card(title, value, icon="📊", color="#6366f1"):
    st.markdown(f"""
    <div style="
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.5rem; margin-right: 10px;">{icon}</span>
            <h3 style="margin: 0; color: #FFFFFF; font-size: 1.1rem;">{title}</h3>
        </div>
        <h2 style="color: {color}; margin: 0; font-size: 1.5rem; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            {value}
        </h2>
    </div>
    """, unsafe_allow_html=True)

def custom_metric(label, value, delta=None, delta_color="normal"):
    if delta is not None:
        st.metric(label=label, value=value, delta=delta)
    else:
        st.metric(label=label, value=value)
# -------------------------------
# Data loading function - UPDATED to fix cache issues
# -------------------------------
def load_data(sidebar_params):
    try:
        # Initialize analyzer based on cache mode
        if sidebar_params["cache_mode"] == "Always Use Cache":
            analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=False, diagnostic_mode=sidebar_params["diagnostic_mode"])
            force_refresh = False
            use_cache = True
        elif sidebar_params["cache_mode"] == "Force Refresh":
            analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=True, diagnostic_mode=sidebar_params["diagnostic_mode"])
            force_refresh = True
            use_cache = True
        else:  # Disable Cache
            analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=False, diagnostic_mode=sidebar_params["diagnostic_mode"])
            force_refresh = True
            use_cache = False

        # Determine wallet mode
        if sidebar_params["analyze_button"] and sidebar_params["wallet_address"].strip():
            chosen_wallet = sidebar_params["wallet_address"].strip()
            using_default = False
            window_start = datetime.combine(sidebar_params["start_date"], datetime.min.time())
            window_end = datetime.combine(sidebar_params["end_date"], datetime.max.time())
            
            # Show cache mode info
            if sidebar_params["cache_mode"] == "Force Refresh":
                st.info("🔄 Force Refresh mode: Fetching fresh data from Moralis API")
            elif sidebar_params["cache_mode"] == "Disable Cache":
                st.info("🚫 Cache disabled: Not using any cached data")
            else:
                st.info("💾 Using cached data if available")
                
        else:
            chosen_wallet = "sample_wallet"
            using_default = True
            window_start = datetime.combine(sidebar_params["start_date"], datetime.min.time())
            window_end = datetime.combine(sidebar_params["end_date"], datetime.max.time())
            st.info("💡 Sample wallet preview: Enter your wallet on the left to analyze real data.")

        if not sidebar_params["selected_chains"]:
            st.warning("Please select at least one blockchain in the sidebar.")
            return None, None, None, None, None, None, None, None

        # **KEY FIX**: Check for updated transactions in session state FIRST
        if not using_default and hasattr(st.session_state, 'updated_transactions_df'):
            st.info("✅ Using updated transaction data with fetched prices")
            df = st.session_state.updated_transactions_df.copy()
            progress = None
        else:
            # Fetch/load wallet data - COMPLETELY SEPARATE PATHS FOR SAMPLE VS REAL DATA
            if using_default:
                # Use sample data directly without any API calls
                df = sample_df.copy()
                progress = None
                
                # Show info about sample data
                st.success(f"Using sample data with {len(df)} transactions")
            else:
                # Real wallet data - fetch from API
                progress = st.progress(0, text="Preparing analysis...")
                progress.progress(50, text="Checking cache / fetching data...")
                
                # Get wallet data from API
                df = get_wallet_data(analyzer, chosen_wallet, sidebar_params["selected_chains"], sidebar_params["max_txs"], force_refresh)

                if progress:
                    progress.progress(50, text="Applying filters...")

        # If we have no data at this point, return early
        if df.empty:
            if progress:
                progress.empty()
            st.error("No transactions found for this wallet across all selected chains.")
            return None, None, None, None, None, None, None, None

        # DEBUG: Show raw transaction counts before filtering
        if not using_default and sidebar_params["diagnostic_mode"]:
            st.subheader("🔍 Raw Transaction Data")
            st.write(f"Total transactions fetched: {len(df)}")
            chain_counts = df['blockchain'].value_counts()
            st.write("Transactions by chain:")
            st.write(chain_counts)
            type_counts = df['transaction_type'].value_counts()
            st.write("Transactions by type:")
            st.write(type_counts)

        # Ensure UTC datetime for both sample and real data
        if df["block_time"].dt.tz is None:
            df["block_time"] = df["block_time"].dt.tz_localize("UTC")
        
        # Convert window_start and window_end to UTC
        window_start_utc = pd.Timestamp(window_start).tz_localize("UTC")
        window_end_utc = pd.Timestamp(window_end).tz_localize("UTC")

        # Filter by date window for both sample and real data
        df = df[(df["block_time"] >= window_start_utc) & (df["block_time"] <= window_end_utc)]

        # DEBUG: Show transaction counts after date filtering
        if not using_default and sidebar_params["diagnostic_mode"]:
            st.write(f"Transactions after date filtering: {len(df)}")
            if len(df) > 0:
                st.write(f"Date range: {df['block_time'].min()} to {df['block_time'].max()}")

        # Mark withdrawals to your own wallet as moves for real data
        if not using_default:
            df['transaction_type'] = df.apply(
                lambda row: 'withdrawal_move'
                if row['transaction_type'] == 'withdrawal' and str(row.get('to_address', '')).lower() == chosen_wallet.lower()
                else row['transaction_type'],
                axis=1
            )

        # DEBUG: Show transaction counts after withdrawal_move processing
        if not using_default and sidebar_params["diagnostic_mode"]:
            st.write(f"Transactions after withdrawal_move processing: {len(df)}")
            type_counts = df['transaction_type'].value_counts()
            st.write("Transactions by type after processing:")
            st.write(type_counts)

        # ADDED: FETCH MISSING PRICES FOR REAL DATA (only if not already updated)
        if not using_default and not hasattr(st.session_state, 'updated_transactions_df'):
            # Add Price Status column first
            df['Price Status'] = df['price_usd'].apply(
                lambda x: '✅ Available' if pd.notna(x) and x > 0 else '❌ Missing'
            )
            
            missing_prices = df[df['Price Status'] == '❌ Missing']
            if not missing_prices.empty:
                st.info(f"🔍 Found {len(missing_prices)} transactions with missing prices. Fetching historical data...")
                
                if progress:
                    progress.progress(60, text="Fetching historical prices...")
                
                # Fetch missing prices
                df = fill_missing_prices_batch(df)
                
                if progress:
                    progress.progress(70, text="Processing price data...")

        # For PnL calculation, we need to handle missing prices more gracefully
        pnl_df = df.copy()
        
        # First, let's identify transactions with missing prices for debugging
        missing_prices = pnl_df[pnl_df["price_usd"].isna()]
        if not missing_prices.empty and not using_default and sidebar_params["diagnostic_mode"]:
            st.write(f"Transactions with missing prices: {len(missing_prices)}")
            st.write("Sample of transactions with missing prices:")
            st.dataframe(missing_prices[["blockchain", "token_symbol", "transaction_type", "amount"]].head())
        
        # Try to fill missing prices using various strategies
        if not pnl_df.empty:
            # Strategy 1: Use average price for the same token
            token_avg_prices = pnl_df.groupby('token_symbol')['price_usd'].mean()
            
            # Strategy 2: For withdrawals, try to find the acquisition price from deposits
            for idx, row in pnl_df[pnl_df["price_usd"].isna()].iterrows():
                token = row['token_symbol']
                
                # Try to find a recent deposit price for the same token
                if token in token_avg_prices and not pd.isna(token_avg_prices[token]):
                    pnl_df.at[idx, 'price_usd'] = token_avg_prices[token]
                    pnl_df.at[idx, 'usd_value'] = row['amount'] * token_avg_prices[token]
                    if sidebar_params["diagnostic_mode"]:
                        st.write(f"Filled missing price for {token} using average: {token_avg_prices[token]}")
                
                # If we still don't have a price, try to get it from the analyzer
                elif not using_default and pd.isna(pnl_df.at[idx, 'price_usd']):
                    try:
                        # Get current price as a fallback
                        if row['token_address']:
                            current_prices = analyzer.get_current_prices([{'address': row['token_address'], 'symbol': token}])
                            if row['token_address'] in current_prices:
                                current_price = current_prices[row['token_address']]
                                pnl_df.at[idx, 'price_usd'] = current_price
                                pnl_df.at[idx, 'usd_value'] = row['amount'] * current_price
                                if sidebar_params["diagnostic_mode"]:
                                    st.write(f"Filled missing price for {token} using current price: {current_price}")
                    except:
                        pass
        
        # Convert price_usd to numeric, coercing errors to NaN
        pnl_df["price_usd"] = pd.to_numeric(pnl_df["price_usd"], errors='coerce')
        
        # Remove rows where price conversion failed (resulted in NaN)
        pnl_df = pnl_df[pnl_df["price_usd"].notna()]
        
        # For PnL calculation, we need positive prices
        pnl_df = pnl_df[pnl_df["price_usd"] > 0]

        if progress:
            progress.progress(80, text="Computing summaries and PnL...")

        # DEBUG: Show transaction counts after price handling
        if not using_default and sidebar_params["diagnostic_mode"]:
            st.write(f"Transactions with valid prices after filling: {len(pnl_df)}")
            st.write(f"Transactions still excluded due to price issues: {len(df) - len(pnl_df)}")
            
            # Show which transactions are still excluded
            still_excluded = df[~df.index.isin(pnl_df.index)]
            if not still_excluded.empty:
                st.write("Transactions still excluded after price filling:")
                st.dataframe(still_excluded[["blockchain", "token_symbol", "transaction_type", "amount", "price_usd"]])
            
        # Calculate basic metrics for both sample and real data
        # We need to calculate these using the original df, not pnl_df
        total_in = float(df[df["transaction_type"] == "deposit"]["usd_value"].sum())
        total_out = float(df[df["transaction_type"] == "withdrawal"]["usd_value"].sum())
        gas_cost = float(df.get("gas_cost_usd", pd.Series()).fillna(0).sum()) if "gas_cost_usd" in df else 0.0
        pnl = total_in - total_out - gas_cost

        # PnL calculation with current prices
        realized_total = 0.0
        unrealized_total = 0.0
        total_gas_costs = 0.0
        breakdown_list = []  # Initialize here to ensure it's always defined
        tokens_with_missing_prices = set()  # Initialize here to ensure it's always defined

        # Use pnl_df for PnL calculations but keep original df for display
        if not pnl_df.empty:
            # Group by token_symbol for sample data, token_address for real data
            group_key = 'token_symbol' if using_default else 'token_address'
            grouped = pnl_df.groupby(group_key)  # Define grouped here

            for token_key, group in grouped:
                # Filter group to rows with valid prices
                group_valid = group[group['price_usd'].notna()]
                
                if group_valid.empty:
                    tokens_with_missing_prices.add(token_key)
                    
                    # Even if we don't have prices, let's still show the token in breakdown
                    # with zero values to indicate missing data
                    breakdown_list.append(pd.DataFrame({
                        'Token': [token_key],
                        'Realized PnL (USD)': [0],
                        'Unrealized PnL (USD)': [0],
                        'Total PnL (USD)': [0],
                        'Gas Costs (USD)': [0],
                        'Status': ['Missing price data']
                    }))
                    continue

                # Convert to numeric to handle string prices
                group_valid["price_usd"] = pd.to_numeric(group_valid["price_usd"], errors='coerce')
                group_valid = group_valid[group_valid["price_usd"].notna()]
                
                if group_valid.empty:
                    tokens_with_missing_prices.add(token_key)
                    
                    # Add to breakdown with zero values
                    breakdown_list.append(pd.DataFrame({
                        'Token': [token_key],
                        'Realized PnL (USD)': [0],
                        'Unrealized PnL (USD)': [0],
                        'Total PnL (USD)': [0],
                        'Gas Costs (USD)': [0],
                        'Status': ['Invalid price data']
                    }))
                    continue

                # Calculate PnL for this token group
                if using_default:
                    # For sample data, create a mock analyzer
                    class MockAnalyzer:
                        def get_current_prices(self, tokens):
                            sample_prices = get_sample_current_prices()
                            result = {}
                            for token in tokens:
                                symbol = token.get('symbol', '')
                                result[token.get('address', symbol)] = sample_prices.get(symbol, 0)
                            return result
                    
                    mock_analyzer = MockAnalyzer()
                    realized, unrealized, gas_costs, breakdown = calculate_pnl_improved(group_valid, method=sidebar_params["pnl_method"], analyzer=mock_analyzer)
                else:
                    realized, unrealized, gas_costs, breakdown = calculate_pnl_improved(group_valid, method=sidebar_params["pnl_method"], analyzer=analyzer)

                realized_total += realized
                unrealized_total += unrealized
                total_gas_costs += gas_costs
                
                # Add status column to breakdown
                if 'Status' not in breakdown.columns:
                    breakdown['Status'] = 'Complete data'
                breakdown_list.append(breakdown)
        else:
            # Handle case where pnl_df is empty
            st.warning("No transactions with valid prices for PnL calculation")
            
            # Still try to create a basic breakdown from the original df
            if not df.empty:
                group_key = 'token_symbol' if using_default else 'token_address'
                unique_tokens = df[group_key].unique()
                
                for token in unique_tokens:
                    breakdown_list.append(pd.DataFrame({
                        'Token': [token],
                        'Realized PnL (USD)': [0],
                        'Unrealized PnL (USD)': [0],
                        'Total PnL (USD)': [0],
                        'Gas Costs (USD)': [0],
                        'Status': ['No price data available']
                    }))

        # Combine breakdowns into one DataFrame
        if breakdown_list:
            breakdown_df = pd.concat(breakdown_list, ignore_index=True)
        else:
            breakdown_df = pd.DataFrame()

        # Calculate overall PnL
        overall_pnl = realized_total + unrealized_total - total_gas_costs
        total_invested = total_in if total_in > 0 else 1  # Avoid division by zero

        # Calculate ROI percentage
        roi_percentage = (overall_pnl / total_invested) * 100 if total_invested > 0 else 0

        # Clear the progress bar when done
        if progress:
            progress.empty()

        # Diagnostic section
        if sidebar_params["diagnostic_mode"] and not using_default:
            st.subheader("🔍 Filtering Diagnostics")
            
            # Show stats about what was filtered out
            st.write(f"Total transactions for display: {len(df)}")
            st.write(f"Total transactions for PnL calculation: {len(pnl_df) if not pnl_df.empty else 0}")
            
            # Show which tokens have missing prices
            if tokens_with_missing_prices:
                st.warning(f"Tokens with missing or invalid prices: {', '.join(tokens_with_missing_prices)}")
                
                # Show a sample of transactions with missing prices
                missing_price_txs = df[df[group_key].isin(tokens_with_missing_prices)].head()
                if not missing_price_txs.empty:
                    with st.expander("View sample of transactions with missing prices"):
                        st.dataframe(missing_price_txs[["token_symbol", "price_usd", "block_time", "transaction_type"]])
            else:
                st.info("All tokens have valid price data")

        # **IMPORTANT**: Store the current wallet address for the transactions page to detect real vs sample data
        df['_wallet_address'] = chosen_wallet

        # Return the data - use original df for display, pnl_df for calculations
        return df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        
        # Return None values for all expected return values
        return None, None, None, None, None, None, None, None
    
#--------------------------------
# Dashboard Page
# -------------------------------
def dashboard_page(df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested):
    st.header("📊 Portfolio Performance Dashboard")
    
    # Summary metrics in custom cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        custom_card("Total Invested", f"${total_invested:,.2f}", "💰", "#6366f1")
    with col2:
        custom_card("Current Value", f"${total_invested + unrealized_total:,.2f}", "📈", "#10b981")
    
    # Color code based on profit/loss
    pnl_color = "#10b981" if overall_pnl >= 0 else "#ef4444"
    pnl_icon = "📈" if overall_pnl >= 0 else "📉"
    
    with col3:
        custom_card("Net PnL", f"${overall_pnl:,.2f}", pnl_icon, pnl_color)

    # Color code based on ROI performance
    if roi_percentage >= 20:
        roi_color = "#10b981"
        roi_icon = "🚀"
    elif roi_percentage >= 0:
        roi_color = "#f59e0b"
        roi_icon = "📈"
    else:
        roi_color = "#ef4444"
        roi_icon = "🔻"
    
    with col4:
        custom_card("ROI", f"{roi_percentage:+.1f}%", roi_icon, roi_color)
    
    # Gauge and Speedometer Charts
    st.subheader("Performance Indicators")
    gauge_col, speedometer_col = st.columns(2)

    with gauge_col:
        # Gauge Chart
        st.write("**💰 PnL Gauge**")
        
        # Determine gauge color
        if roi_percentage >= 20:
            gauge_color = "green"
        elif roi_percentage >= 10:
            gauge_color = "lightgreen"
        elif roi_percentage >= 0:
            gauge_color = "yellow"
        elif roi_percentage >= -10:
            gauge_color = "orange"
        else:
            gauge_color = "red"
        
        # Create simple gauge using progress bar
        max_abs_value = max(abs(overall_pnl) * 1.5, total_invested * 0.5, 1000)
        gauge_position = 50 + (overall_pnl / max_abs_value) * 50
        gauge_position = max(0, min(100, gauge_position))
        
        st.progress(gauge_position/100, text=f"${overall_pnl:,.2f}")
        
        # Gauge labels
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_left:
            st.caption(f"-${max_abs_value:,.0f}")
        with col_center:
            st.caption("← Loss | Profit →")
        with col_right:
            st.caption(f"+${max_abs_value:,.0f}")
        
        # Performance indicator
        if overall_pnl > 0:
            st.success(f"✅ Profit: ${overall_pnl:,.2f}")
        elif overall_pnl < 0:
            st.error(f"❌ Loss: ${abs(overall_pnl):,.2f}")
        else:
            st.info("⚖️ Break Even")

    with speedometer_col:
        # Speedometer Chart
        st.write("**🚀 ROI Speedometer**")
        
        # Determine speedometer color and ranges
        if roi_percentage >= 30:
            speed_color = "#00FF00"  # Green - excellent
            speed_label = "Exceptional"
        elif roi_percentage >= 15:
            speed_color = "#7CFC00"  # Light green - great
            speed_label = "Great"
        elif roi_percentage >= 5:
            speed_color = "#ADFF2F"  # Green yellow - good
            speed_label = "Good"
        elif roi_percentage >= 0:
            speed_color = "#FFFF00"  # Yellow - okay
            speed_label = "Okay"
        elif roi_percentage >= -10:
            speed_color = "#FFA500"  # Orange - caution
            speed_label = "Caution"
        else:
            speed_color = "#FF0000"  # Red - danger
            speed_label = "Danger"
        
        # Create speedometer using progress bar
        # Scale: -50% to +50% ROI for the speedometer
        speedometer_min = -50
        speedometer_max = 50
        speedometer_value = max(speedometer_min, min(speedometer_max, roi_percentage))
        speedometer_normalized = (speedometer_value - speedometer_min) / (speedometer_max - speedometer_min)
        
        st.progress(speedometer_normalized, text=f"{roi_percentage:+.1f}% ROI")
        
        # Speedometer labels
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"{speedometer_min}%")
        with col2:
            st.caption(speed_label)
        with col3:
            st.caption(f"{speedometer_max}+%")
        
        # ROI performance text
        if roi_percentage >= 20:
            st.success("🎯 Exceptional Returns!")
        elif roi_percentage >= 10:
            st.success("🚀 Beating the Market!")
        elif roi_percentage >= 5:
            st.info("📈 Solid Performance")
        elif roi_percentage >= 0:
            st.info("📊 Market Average")
        elif roi_percentage >= -10:
            st.warning("⚠️ Below Expectations")
        else:
            st.error("🔻 Needs Improvement")

    # Performance Assessment
    st.subheader("📊 Performance Assessment")

    if roi_percentage >= 20:
        st.success("""
        **🎯 Outstanding Performance!**
        - You're significantly outperforming the market
        - Consider taking some profits
        - Review your successful strategies
        """)
    elif roi_percentage >= 10:
        st.success("""
        **🚀 Excellent Performance!**
        - Beating market averages
        - Solid investment decisions
        - Maintain your strategy
        """)
    elif roi_percentage >= 5:
        st.info("""
        **📈 Good Performance!**
        - Steady growth above inflation
        - Consistent strategy working
        - Consider diversification
        """)
    elif roi_percentage >= 0:
        st.info("""
        **⚖️ Break-Even Performance**
        - Keeping pace with market
        - Review asset allocation
        - Consider cost optimization
        """)
    elif roi_percentage >= -10:
        st.warning("""
        **⚠️ Needs Improvement**
        - Below market performance
        - Review investment thesis
        - Consider rebalancing
        """)
    else:
        st.error("""
        **🔻 Concerning Performance**
        - Significant underperformance
        - Urgent portfolio review needed
        - Consider professional advice
        """)

# -------------------------------
# PnL Analysis Page
# -------------------------------
# -------------------------------
# PnL Analysis Page - FIXED VERSION
# -------------------------------
def pnl_analysis_page(realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested):
    st.header("💹 PnL Analysis")
    
    # Detailed Breakdown
    st.subheader("📋 Detailed Breakdown")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    with detail_col1:
        custom_card("Realized PnL", f"${realized_total:,.2f}", "💰", "#10b981")
        st.caption(f"{(realized_total/total_invested)*100:+.1f}% of Investment" if total_invested > 0 else "N/A")
    
    with detail_col2:
        custom_card("Unrealized PnL", f"${unrealized_total:,.2f}", "📈", "#f59e0b")
        st.caption(f"{(unrealized_total/total_invested)*100:+.1f}% of Investment" if total_invested > 0 else "N/A")
    
    with detail_col3:
        custom_card("Gas Costs", f"${total_gas_costs:,.2f}", "⛽", "#ef4444")
        st.caption(f"{(total_gas_costs/total_invested)*100:.1f}% of Investment" if total_invested > 0 else "N/A")
    
    # PnL Breakdown by Token with Current Holdings
    if not breakdown_df.empty:
        st.subheader("💹 PnL & Holdings Breakdown")
        
        # Create display DataFrame
        display_df = breakdown_df.copy()
        
        # Handle different column naming from the PnL calculation
        # The corrected PnL function returns 'Current Holdings' instead of 'qty'
        if 'Current Holdings' in display_df.columns:
            # Rename to match your expected column name
            display_df['qty'] = display_df['Current Holdings']
        
        # Ensure required columns exist with fallbacks
        if 'qty' not in display_df.columns:
            display_df['qty'] = 0
        
        # Calculate current holdings and market value
        display_df['Current Holdings'] = display_df['qty'].fillna(0)
        
        # Handle different column naming for prices
        if 'Current Price' in display_df.columns and 'Current Price (USD)' not in display_df.columns:
            display_df['Current Price (USD)'] = display_df['Current Price']
        
        if 'Current Price (USD)' not in display_df.columns:
            display_df['Current Price (USD)'] = 0
            
        # Calculate market value correctly
        display_df['Market Value'] = display_df['Current Holdings'] * display_df['Current Price (USD)']
        
        # Handle different column naming for cost basis
        if 'Avg Cost' in display_df.columns and 'Avg Cost (USD)' not in display_df.columns:
            display_df['Avg Cost (USD)'] = display_df['Avg Cost']
        
        if 'Avg Cost (USD)' not in display_df.columns:
            display_df['Avg Cost (USD)'] = 0
        
        # Calculate portfolio percentages
        total_market_value = display_df['Market Value'].sum()
        if total_market_value > 0:
            display_df['% of Portfolio'] = (display_df['Market Value'] / total_market_value * 100).round(2)
        else:
            display_df['% of Portfolio'] = 0
        
        # Filter for meaningful rows and sort
        display_df = display_df[
            (display_df['Current Holdings'] > 0) | 
            (display_df.get('Realized PnL (USD)', pd.Series([0])).fillna(0) != 0) |
            (display_df.get('Unrealized PnL (USD)', pd.Series([0])).fillna(0) != 0)
        ].sort_values('Market Value', ascending=False)
        
        # Display the enhanced table
        try:
            st.dataframe(
                display_df.style
                .format({
                    'Current Holdings': '{:,.6f}',  # More decimal places for small amounts
                    'Avg Cost (USD)': '${:,.4f}',   # More decimal places for cost basis
                    'Current Price (USD)': '${:,.4f}',
                    'Market Value': '${:,.2f}',
                    'Realized PnL (USD)': '${:,.2f}',
                    'Unrealized PnL (USD)': '${:,.2f}',
                    '% of Portfolio': '{:.2f}%'
                })
                .applymap(
                    lambda x: 'color: #10b981' if isinstance(x, (int, float)) and x > 0 
                    else ('color: #ef4444' if isinstance(x, (int, float)) and x < 0 else ''),
                    subset=[col for col in ['Realized PnL (USD)', 'Unrealized PnL (USD)', 'Market Value'] if col in display_df.columns]
                )
                .set_properties(**{'background-color': '#1e293b', 'color': '#cbd5e1'})
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', 'white')]},
                ]),
                use_container_width=True,
                height=400
            )
        except Exception as e:
            st.error(f"Error formatting table: {e}")
            st.dataframe(display_df, use_container_width=True, height=400)
        
        # Add a note about calculation accuracy
        st.warning("⚠️ Note: Some holdings calculations may be inaccurate. Please verify with transaction data.")

    # PnL visualizations - FIXED VERSION
    st.subheader("📊 PnL Visualizations")
    
    # Check if breakdown_df has the required columns
    if not breakdown_df.empty:
        # Detect the actual column names
        token_col = None
        realized_col = None
        unrealized_col = None
        
        # Look for token column
        for col in breakdown_df.columns:
            if 'token' in col.lower() or col.lower() in ['token', 'symbol', 'asset']:
                token_col = col
                break
        
        # Look for realized PnL column
        for col in breakdown_df.columns:
            if 'realized' in col.lower() and 'pnl' in col.lower():
                realized_col = col
                break
        
        # Look for unrealized PnL column
        for col in breakdown_df.columns:
            if 'unrealized' in col.lower() and 'pnl' in col.lower():
                unrealized_col = col
                break
        
        st.info(f"Detected columns - Token: {token_col}, Realized: {realized_col}, Unrealized: {unrealized_col}")
        
        # Only create charts if we have the required columns
        if token_col and (realized_col or unrealized_col):
            # Check if we have enough space for side-by-side layout
            enough_space = len(breakdown_df) <= 8
            
            if enough_space:
                # Side-by-side layout
                col1, col2 = st.columns(2)
                
                if realized_col:
                    with col1:
                        # Realized PnL Bar Chart
                        realized_chart_data = breakdown_df[[token_col, realized_col]].copy()
                        realized_chart_data = realized_chart_data.sort_values(realized_col, ascending=True)
                        
                        st.write("**💰 Realized PnL by Token**")
                        st.bar_chart(
                            realized_chart_data.set_index(token_col),
                            color="#10b981"
                        )
                        
                        # Summary stats
                        total_realized = realized_chart_data[realized_col].sum()
                        st.caption(f"Total Realized: ${total_realized:,.2f}")
                else:
                    with col1:
                        st.info("No realized PnL data available for visualization")
                
                if unrealized_col:
                    with col2:
                        # Unrealized PnL Bar Chart
                        unrealized_chart_data = breakdown_df[[token_col, unrealized_col]].copy()
                        unrealized_chart_data = unrealized_chart_data.sort_values(unrealized_col, ascending=True)
                        
                        st.write("**📈 Unrealized PnL by Token**")
                        # Using matplotlib for better color control
                        try:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 6))
                            fig.patch.set_facecolor('#0f172a')
                            ax.set_facecolor('#1e293b')
                            
                            tokens = unrealized_chart_data[token_col]
                            values = unrealized_chart_data[unrealized_col]
                            
                            colors = ['#10b981' if x >= 0 else '#ef4444' for x in values]
                            
                            bars = ax.barh(tokens, values, color=colors)
                            ax.set_xlabel('Unrealized PnL (USD)', color='white')
                            ax.set_ylabel('Token', color='white')
                            ax.set_title('Unrealized PnL by Token', color='white')
                            
                            # Set tick colors
                            ax.tick_params(colors='white')
                            
                            # Set spine colors
                            for spine in ax.spines.values():
                                spine.set_color('#334155')
                            
                            # Add value labels on bars
                            for bar in bars:
                                width = bar.get_width()
                                if abs(width) > 0.01:  # Only add labels for non-zero values
                                    label_x_pos = width + (0.01 * max(abs(values.max()), abs(values.min()))) if width >= 0 else width - (0.01 * max(abs(values.max()), abs(values.min())))
                                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                           f'${width:,.0f}', ha='left' if width >= 0 else 'right', va='center', color='white')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        except ImportError:
                            # Fallback to streamlit bar chart
                            st.bar_chart(
                                unrealized_chart_data.set_index(token_col),
                                color="#f59e0b"
                            )
                        
                        # Summary stats
                        total_unrealized = unrealized_chart_data[unrealized_col].sum()
                        st.caption(f"Total Unrealized: ${total_unrealized:,.2f}")
                else:
                    with col2:
                        st.info("No unrealized PnL data available for visualization")
            
            else:
                # Stacked layout (not enough space)
                if realized_col:
                    # Realized PnL Bar Chart - Top
                    st.write("**💰 Realized PnL by Token**")
                    realized_chart_data = breakdown_df[[token_col, realized_col]].copy()
                    realized_chart_data = realized_chart_data.sort_values(realized_col, ascending=True)
                    
                    st.bar_chart(
                        realized_chart_data.set_index(token_col),
                        color="#10b981"
                    )
                    
                    total_realized = realized_chart_data[realized_col].sum()
                    st.caption(f"Total Realized PnL: ${total_realized:,.2f}")
                    
                    # Add some spacing
                    st.write("")
                    st.write("")
                
                if unrealized_col:
                    # Unrealized PnL Bar Chart - Bottom
                    st.write("**📈 Unrealized PnL by Token**")
                    unrealized_chart_data = breakdown_df[[token_col, unrealized_col]].copy()
                    unrealized_chart_data = unrealized_chart_data.sort_values(unrealized_col, ascending=True)
                    
                    # Using matplotlib for better color control
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig.patch.set_facecolor('#0f172a')
                        ax.set_facecolor('#1e293b')
                        
                        tokens = unrealized_chart_data[token_col]
                        values = unrealized_chart_data[unrealized_col]
                        
                        colors = ['#10b981' if x >= 0 else '#ef4444' for x in values]
                        
                        bars = ax.barh(tokens, values, color=colors)
                        ax.set_xlabel('Unrealized PnL (USD)', color='white')
                        ax.set_ylabel('Token', color='white')
                        ax.set_title('Unrealized PnL by Token', color='white')
                        
                        # Set tick colors
                        ax.tick_params(colors='white')
                        
                        # Set spine colors
                        for spine in ax.spines.values():
                            spine.set_color('#334155')
                        
                        # Add value labels on bars
                        for bar in bars:
                            width = bar.get_width()
                            if abs(width) > 0.01:  # Only add labels for non-zero values
                                label_x_pos = width + (0.01 * max(abs(values.max()), abs(values.min()))) if width >= 0 else width - (0.01 * max(abs(values.max()), abs(values.min())))
                                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                       f'${width:,.0f}', ha='left' if width >= 0 else 'right', va='center', color='white')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except ImportError:
                        # Fallback to streamlit bar chart
                        st.bar_chart(
                            unrealized_chart_data.set_index(token_col),
                            color="#f59e0b"
                        )
                    
                    total_unrealized = unrealized_chart_data[unrealized_col].sum()
                    st.caption(f"Total Unrealized PnL: ${total_unrealized:,.2f}")

            # Pie Charts for Top Tokens by PnL - FIXED VERSION
            st.subheader("📊 Top Tokens by PnL")
            
            col1, col2 = st.columns(2)
            
            if realized_col:
                with col1:
                    # Top 5 tokens by Realized PnL
                    realized_top5 = breakdown_df.nlargest(5, realized_col)
                    if not realized_top5.empty and realized_top5[realized_col].sum() != 0:
                        # Calculate percentages for tooltips
                        total_realized_abs = realized_top5[realized_col].abs().sum()
                        if total_realized_abs > 0:
                            realized_top5['Percentage'] = (realized_top5[realized_col].abs() / total_realized_abs * 100).round(1)
                            
                            fig_realized = px.pie(
                                realized_top5,
                                values=realized_col,
                                names=token_col,
                                title='💰 Top 5 Tokens by Realized PnL',
                                hover_data=['Percentage'],
                                color_discrete_sequence=px.colors.sequential.Greens
                            )
                            fig_realized.update_layout(
                                plot_bgcolor='#1e293b',
                                paper_bgcolor='#1e293b',
                                font_color='white',
                                showlegend=False,
                                title_x=0.5,
                                title_font_size=16
                            )
                            fig_realized.update_traces(
                                textposition='inside', 
                                textinfo='percent+label',
                                hovertemplate='<b>%{label}</b><br>PnL: $%{value:,.2f}<br>Share: %{customdata[0]}%<extra></extra>'
                            )
                            st.plotly_chart(fig_realized, use_container_width=True)
                            
                            # Show summary stats
                            st.caption(f"Total Realized PnL: ${realized_top5[realized_col].sum():,.2f}")
                        else:
                            st.info("No significant realized PnL data for pie chart")
                    else:
                        st.info("No realized PnL data available")
            else:
                with col1:
                    st.info("No realized PnL data available")
            
            if unrealized_col:
                with col2:
                    # Top 5 tokens by Unrealized PnL (absolute value)
                    unrealized_top5 = breakdown_df.reindex(
                        breakdown_df[unrealized_col].abs().nlargest(5).index
                    )
                    if not unrealized_top5.empty and unrealized_top5[unrealized_col].abs().sum() != 0:
                        # Calculate percentages for tooltips
                        total_unrealized_abs = unrealized_top5[unrealized_col].abs().sum()
                        if total_unrealized_abs > 0:
                            unrealized_top5['Percentage'] = (unrealized_top5[unrealized_col].abs() / total_unrealized_abs * 100).round(1)
                            
                            # Use different colors for positive and negative PnL
                            colors = ['#10b981' if x >= 0 else '#ef4444' for x in unrealized_top5[unrealized_col]]
                            
                            fig_unrealized = px.pie(
                                unrealized_top5,
                                values=unrealized_col,
                                names=token_col,
                                title='📈 Top 5 Tokens by Unrealized PnL',
                                hover_data=['Percentage'],
                                color=colors
                            )
                            fig_unrealized.update_layout(
                                plot_bgcolor='#1e293b',
                                paper_bgcolor='#1e293b',
                                font_color='white',
                                showlegend=False,
                                title_x=0.5,
                                title_font_size=16
                            )
                            fig_unrealized.update_traces(
                                textposition='inside', 
                                textinfo='percent+label',
                                hovertemplate='<b>%{label}</b><br>PnL: $%{value:,.2f}<br>Share: %{customdata[0]}%<extra></extra>'
                            )
                            st.plotly_chart(fig_unrealized, use_container_width=True)
                            
                            # Show summary stats
                            total_unrealized = unrealized_top5[unrealized_col].sum()
                            st.caption(f"Total Unrealized PnL: ${total_unrealized:,.2f}")
                        else:
                            st.info("No significant unrealized PnL data for pie chart")
                    else:
                        st.info("No unrealized PnL data available")
            else:
                with col2:
                    st.info("No unrealized PnL data available")
        
        else:
            st.warning("Cannot create visualizations: Required columns not found in breakdown data")
            st.write("Available columns:", list(breakdown_df.columns))
    
    else:
        st.warning("No breakdown data available for visualizations")

# -------------------------------
# Transactions Page
# -------------------------------
def transactions_page(df):
    st.header("📊 Transaction Details")
    
    # Check if we're working with sample data
    using_sample_data = df is None or len(df) == 0 or (hasattr(df, 'iloc') and '_wallet_address' in df.columns and df['_wallet_address'].iloc[0] == 'sample_wallet')
    
    if using_sample_data:
        st.info("💡 This is sample data. Enter a real wallet address to fetch actual transaction prices.")
    
    # Use the dataframe passed in (it's already the correct one)
    df_display = df.copy() if df is not None else pd.DataFrame()
    
    if df_display.empty:
        st.warning("No transaction data available")
        return
    
    # Add a column to indicate if price is missing
    df_display['Price Status'] = df_display['price_usd'].apply(
        lambda x: '✅ Available' if pd.notna(x) and x > 0 else '❌ Missing'
    )
    
    # Show missing prices warning - but only for real data, not sample data
    missing_prices = df_display[df_display['Price Status'] == '❌ Missing']
    if not missing_prices.empty and not using_sample_data:
        st.warning(f"⚠️ {len(missing_prices)} transactions have missing prices")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🔄 Fetch Missing Prices", type="secondary", key="fetch_prices_btn"):
                try:
                    with st.spinner("Fetching historical prices from CoinGecko..."):
                        # Import the function
                        from price_fetcher import fill_missing_prices_batch
                        
                        # Get cache key from the dataframe
                        cache_key = df_display['_cache_key'].iloc[0] if '_cache_key' in df_display.columns else None
                        
                        # Prepare dataframe for updating (remove helper columns)
                        df_for_update = df_display.copy()
                        for col in ['_cache_key', '_wallet_address', 'Price Status']:
                            if col in df_for_update.columns:
                                df_for_update = df_for_update.drop(col, axis=1)
                        
                        # Fetch updated prices
                        updated_df = fill_missing_prices_batch(df_for_update)
                        
                        # Restore helper columns
                        if cache_key:
                            updated_df['_cache_key'] = cache_key
                        if '_wallet_address' in df_display.columns:
                            updated_df['_wallet_address'] = df_display['_wallet_address'].iloc[0]
                        
                        # **AUTOMATIC CACHING**: Store updated data with the correct cache key
                        if cache_key:
                            set_cached_updated_transactions(cache_key, updated_df)
                        
                        # Show success message
                        st.success("✅ Prices fetched successfully! The app will refresh automatically.")
                        
                        # Force rerun to reload data from cache
                        time.sleep(1)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error fetching prices: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        with col2:
            st.info("Click the button to fetch historical prices from CoinGecko API")
         
        # Show sample of missing price transactions
        with st.expander(f"View sample of {len(missing_prices)} transactions with missing prices"):
            sample_missing = missing_prices.head(10)[['blockchain', 'token_symbol', 'transaction_type', 'amount', 'block_time', 'Price Status']]
            st.dataframe(sample_missing, use_container_width=True)
    
    elif not missing_prices.empty and using_sample_data:
        st.info("Sample data includes some transactions with mock missing prices for demonstration")
    
    # Display transactions table with custom styling
    st.subheader("📋 All Transactions")
    
    # Remove helper columns for display
    display_columns = [col for col in df_display.columns if not col.startswith('_')]
    df_for_display = df_display[display_columns]
    
    try:
        st.dataframe(
            df_for_display.style
            .applymap(lambda x: 'color: #ef4444' if x == '❌ Missing' else 'color: #10b981', 
                     subset=['Price Status'])
            .set_properties(**{'background-color': '#1e293b', 'color': '#cbd5e1'})
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', 'white')]},
            ]),
            use_container_width=True,
            height=600
        )
    except Exception as e:
        st.error(f"Error styling table: {e}")
        st.dataframe(df_for_display, use_container_width=True, height=600)
    
    # Rest of the function remains the same...
    # [Include all existing visualization and statistics code]


def get_cache_key(wallet_address, selected_chains, start_date, end_date):
    """Generate a unique cache key for the current wallet and parameters"""
    chain_str = "_".join(sorted(selected_chains))
    return f"{wallet_address}_{chain_str}_{start_date}_{end_date}"

def get_cached_updated_transactions(cache_key):
    """Get cached updated transactions for specific parameters"""
    if 'updated_transactions_cache' not in st.session_state:
        st.session_state.updated_transactions_cache = {}
    
    return st.session_state.updated_transactions_cache.get(cache_key, None)

def set_cached_updated_transactions(cache_key, df):
    """Cache updated transactions for specific parameters"""
    if 'updated_transactions_cache' not in st.session_state:
        st.session_state.updated_transactions_cache = {}
    
    st.session_state.updated_transactions_cache[cache_key] = df
    
    # Optional: Limit cache size to prevent memory issues
    if len(st.session_state.updated_transactions_cache) > 5:
        # Remove oldest entry
        oldest_key = list(st.session_state.updated_transactions_cache.keys())[0]
        del st.session_state.updated_transactions_cache[oldest_key]

def clear_all_transaction_cache():
    """Clear all cached transaction data (for debugging only)"""
    if 'updated_transactions_cache' in st.session_state:
        del st.session_state.updated_transactions_cache
        st.info("All transaction cache cleared")
      
# -------------------------------
# Settings & Diagnostics Page
# -------------------------------
def settings_diagnostics_page(df, realized_total, unrealized_total, total_gas_costs, breakdown_df, sidebar_params):
    st.header("⚙️ Settings & Diagnostics")
    
    # Cache management
    st.subheader("📂 Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗑️ Clear Cache (Admin Only)")
        if clear_all_transaction_cache_protected():
            # Cache was cleared, show success message
            pass
    
    with col2:
        if st.button("📊 Export Data to CSV", use_container_width=True):
            # Implementation for exporting data
            st.info("Data export functionality would be implemented here")
    
    # Show cache statistics
    st.subheader("💾 Cache Statistics")
    if 'updated_transactions_cache' in st.session_state:
        cache_count = len(st.session_state.updated_transactions_cache)
        cache_keys = list(st.session_state.updated_transactions_cache.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            custom_card("Cached Datasets", str(cache_count), "💾", "#6366f1")
        
        with col2:
            # Calculate approximate cache size
            total_rows = sum(len(df) for df in st.session_state.updated_transactions_cache.values())
            custom_card("Total Cached Rows", str(total_rows), "📊", "#10b981")
        
        if cache_count > 0:
            with st.expander("View Cache Details", expanded=False):
                cache_info = []
                for key in cache_keys:
                    cached_df = st.session_state.updated_transactions_cache[key]
                    cache_info.append({
                        "Cache Key": key,
                        "Rows": len(cached_df),
                        "Columns": len(cached_df.columns),
                        "Memory (MB)": round(cached_df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                    })
                
                st.dataframe(pd.DataFrame(cache_info), use_container_width=True)
    else:
        st.info("No cached transaction data found")
    
    # Diagnostic information
    st.subheader("🔍 Diagnostic Information")
    
    # PnL Validation
    if not df.empty:
        # Group by token_symbol for sample data, token_address for real data
        group_key = 'token_symbol' if sidebar_params["wallet_address"].strip() == "" else 'token_address'
        
        validation_df = validate_pnl_calculation(
            df, 
            realized_total, 
            unrealized_total, 
            total_gas_costs,
            breakdown_df
        )
        failed_validations = validation_df[validation_df['Pass'] == False]

        if not failed_validations.empty:
            st.warning("⚠️ PnL Validation Issues Detected")
            with st.expander("View Validation Details", expanded=True):
                st.dataframe(
                    validation_df.style
                    .applymap(lambda x: 'color: #10b981' if x == True else 'color: #ef4444', 
                             subset=['Pass'])
                    .set_properties(**{'background-color': '#1e293b', 'color': '#cbd5e1'})
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', 'white')]},
                    ]),
                    use_container_width=True
                )
                st.write("**Issues found:**")
                for _, row in failed_validations.iterrows():
                    st.write(f"- {row['Check']}: Failed")
        else:
            st.success("✅ PnL Calculations Validated Successfully")
            with st.expander("View Validation Details"):
                st.dataframe(
                    validation_df.style
                    .applymap(lambda x: 'color: #10b981' if x == True else 'color: #ef4444', 
                             subset=['Pass'])
                    .set_properties(**{'background-color': '#1e293b', 'color': '#cbd5e1'})
                    .set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', 'white')]},
                    ]),
                    use_container_width=True
                )
    
    # System information
    st.subheader("🖥️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        custom_card("Python Version", "3.x", "🐍", "#6366f1")
        custom_card("Streamlit Version", st.__version__, "🎈", "#ff4b4b")
    
    with col2:
        custom_card("Pandas Version", pd.__version__, "🐼", "#150458")
        custom_card("Cache Directory", CACHE_DIR, "📁", "#f59e0b")
    
    # Data quality check
    st.subheader("📊 Data Quality Check")
    
    if not df.empty:
        # Check for missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        quality_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': missing_percentage.values
        })
        
        st.dataframe(
            quality_df[quality_df['Missing Values'] > 0].style
            .applymap(lambda x: 'color: #ef4444' if x > 0 else 'color: #10b981', 
                     subset=['Missing Values', 'Percentage'])
            .set_properties(**{'background-color': '#1e293b', 'color': '#cbd5e1'})
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0f172a'), ('color', 'white')]},
            ]),
            use_container_width=True
        )
        
        if quality_df[quality_df['Missing Values'] > 0].empty:
            st.success("✅ No missing values found in the dataset")
        else:
            st.warning("⚠️ Missing values detected in some columns")

# Configuration section for easy password management
def setup_admin_config():
    """Setup section for admin configuration - put this at the top of your app.py"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔐 Admin Config")
    
    with st.sidebar.expander("Change Admin Password", expanded=False):
        st.warning("⚠️ This will change the admin password for cache operations")
        new_password = st.text_input("New Password:", type="password", key="new_admin_password")
        confirm_password = st.text_input("Confirm Password:", type="password", key="confirm_admin_password")
        
        if st.button("Update Password", key="update_admin_password"):
            if new_password and new_password == confirm_password:
                # In a real application, you'd want to save this to a config file or database
                # For now, we'll just update the session state
                st.session_state.admin_password_hash = hash_password(new_password)
                st.success("✅ Password updated for this session")
                st.info("Note: Password will reset when app restarts. Consider using environment variables for production.")
            elif new_password != confirm_password:
                st.error("❌ Passwords don't match")
            else:
                st.warning("⚠️ Please enter a password")

# Environment variable approach for production (recommended)
def get_admin_password_hash():
    """Get admin password hash from environment variable or use default"""
    import os
    
    # Try to get from environment variable first
    env_password = os.getenv("WALLET_ANALYZER_ADMIN_PASSWORD")
    if env_password:
        return hash_password(env_password)
    
    # Try to get from session state (if updated during session)
    if hasattr(st.session_state, 'admin_password_hash'):
        return st.session_state.admin_password_hash
    
    # Fallback to default (change this!)
    return hash_password("Explorer123")

#--------------------------------------------------
# Password protected cache clearing functionality
#--------------------------------------------------

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

# Set your admin password hash here (change this!)
ADMIN_PASSWORD_HASH = hash_password("Explorer123")  # Change this to your desired password

def verify_password(entered_password):
    """Verify if the entered password is correct"""
    return hash_password(entered_password) == ADMIN_PASSWORD_HASH

def clear_all_transaction_cache_protected():
    """Password protected function to clear all cached transaction data"""
    if 'cache_clear_authenticated' not in st.session_state:
        st.session_state.cache_clear_authenticated = False
    
    if not st.session_state.cache_clear_authenticated:
        st.warning("⚠️ This action will clear ALL cached transaction data")
        password_input = st.text_input("Enter admin password:", type="password", key="cache_clear_password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Verify Password", key="verify_cache_clear"):
                if password_input and verify_password(password_input):
                    st.session_state.cache_clear_authenticated = True
                    st.success("✅ Password verified! You can now clear the cache.")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
        
        with col2:
            if st.button("Cancel", key="cancel_cache_clear"):
                st.info("Cache clear operation cancelled")
        
        return False
    
    else:
        st.success("✅ Authenticated - Ready to clear cache")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ CONFIRM: Clear All Cache", type="primary", key="confirm_cache_clear"):
                # Actually clear the cache
                if 'updated_transactions_cache' in st.session_state:
                    del st.session_state.updated_transactions_cache
                
                # Reset authentication
                st.session_state.cache_clear_authenticated = False
                
                st.success("✅ All transaction cache cleared successfully!")
                return True
        
        with col2:
            if st.button("Cancel", key="cancel_after_auth"):
                st.session_state.cache_clear_authenticated = False
                st.info("Cache clear operation cancelled")
                st.rerun()
        
        return False

# -------------------------------
# Main function
# -------------------------------
def main():
    # Setup navigation tabs (in main page, not sidebar)
    tabs = setup_navigation()
    
    # Setup sidebar (shared across all pages)
    sidebar_params = setup_sidebar()
    
    # Load data (shared across all pages)
    data = load_data(sidebar_params)
    
    # Check if data loading failed (any of the first 3 elements is None)
    if data[0] is None or data[1] is None or data[2] is None:
        # Show a beautiful empty state
        st.error("No data available or error loading data. Please check your wallet address and try again.")
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <span style="font-size: 4rem;">📭</span>
            <h3>No Wallet Data Available</h3>
            <p>Please enter a valid wallet address in the sidebar and click "Analyze Wallet"</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Unpack the data
    df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested = data
    
    # Display appropriate page based on navigation
    with tabs[0]:
        dashboard_page(df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested)
    
    with tabs[1]:
        pnl_analysis_page(realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested)
    
    with tabs[2]:
        transactions_page(df)
    
    with tabs[3]:
        settings_diagnostics_page(df, realized_total, unrealized_total, total_gas_costs, breakdown_df, sidebar_params)
    
    # Add a nice footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94a3b8;">
            <p>Wallet PnL Explorer • Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
if __name__ == "__main__":
    main()
