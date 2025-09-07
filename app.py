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
import plotly.express as px
import matplotlib.pyplot as plt

# Set page config first
st.set_page_config(
    page_title="Wallet PnL Explorer", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample Data
def generate_sample_data(n_days=7, txs_per_day=7, wallet_address="0xDEADBEEF1234567890ABCDEF1234567890ABCDEF"):
    np.random.seed(42)  # reproducible
    rows = []
    start_date = datetime.today() - timedelta(days=n_days)

    tokens = [
        ("ADA", 0.75),
        ("XRP", 2.5), 
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

# -------------------------------
# Navigation setup
# -------------------------------
def setup_navigation():
    st.sidebar.title("💰 Wallet PnL Explorer")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.radio(
        "Navigate to:",
        ["Dashboard", "PnL Analysis", "Transactions", "Settings & Diagnostics"],
        index=0
    )
    
    return page

# -------------------------------
# Sidebar controls (shared across pages)
# -------------------------------
def setup_sidebar():
    st.sidebar.header("🔧 Controls")
    
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
    
    return {
        "diagnostic_mode": diagnostic_mode,
        "pnl_method": pnl_method,
        "wallet_address": wallet_address,
        "selected_chains": selected_chains,
        "start_date": start_date,
        "end_date": end_date,
        "max_txs": max_txs,
        "cache_mode": cache_mode,
        "analyze_button": analyze_button
    }

# -------------------------------
# Data loading function
# -------------------------------
def load_data(sidebar_params):
    # Initialize analyzer
    if sidebar_params["cache_mode"] == "Always Use Cache":
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=False)
        force_refresh = False
    elif sidebar_params["cache_mode"] == "Force Refresh":
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=True)
        force_refresh = True
    else:
        analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=False)
        force_refresh = False

    # Determine wallet mode
    if sidebar_params["analyze_button"] and sidebar_params["wallet_address"].strip():
        chosen_wallet = sidebar_params["wallet_address"].strip()
        using_default = False
        window_start = datetime.combine(sidebar_params["start_date"], datetime.min.time())
        window_end = datetime.combine(sidebar_params["end_date"], datetime.max.time())
    else:
        chosen_wallet = "sample_wallet"
        using_default = True
        window_start = sample_df["block_time"].min()
        window_end = sample_df["block_time"].max()
        st.info("💡 Sample wallet preview for the past 7 days: Enter your wallet on the left to analyze real data.")

    if not sidebar_params["selected_chains"]:
        st.warning("Please select at least one blockchain in the sidebar.")
        return None, None, None, None, None, None, None, None

    # Fetch/load wallet data
    if using_default:
        df = sample_df.copy()
    else:
        progress = st.progress(0, text="Preparing analysis...")
        progress.progress(20, text="Checking cache / fetching data...")
        df = get_wallet_data(analyzer, chosen_wallet, sidebar_params["selected_chains"], sidebar_params["max_txs"], force_refresh=force_refresh)
        progress.progress(50, text="Applying filters...")

        if df.empty:
            progress.empty()
            st.error("No transactions found for this wallet.")
            return None, None, None, None, None, None, None, None

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
        return None, None, None, None, None, None, None, None

    # Calculate PnL
    total_in = float(df[df["transaction_type"] == "deposit"]["usd_value"].sum())
    total_out = float(df[df["transaction_type"] == "withdrawal"]["usd_value"].sum())
    gas_cost = float(df.get("gas_cost_usd", pd.Series()).fillna(0).sum()) if "gas_cost_usd" in df else 0.0
    pnl = total_in - total_out - gas_cost

    # PnL calculation with current prices
    realized_total = 0.0
    unrealized_total = 0.0
    total_gas_costs = 0.0
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
            realized, unrealized, gas_costs, breakdown = calculate_pnl_improved(group_valid, method=sidebar_params["pnl_method"], analyzer=mock_analyzer)
        else:
            realized, unrealized, gas_costs, breakdown = calculate_pnl_improved(group_valid, method=sidebar_params["pnl_method"], analyzer=analyzer)

        realized_total += realized
        unrealized_total += unrealized
        total_gas_costs += gas_costs
        breakdown_list.append(breakdown)

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

    return df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested

# -------------------------------
# Dashboard Page
# -------------------------------
def dashboard_page(df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested):
    st.header("📊 Portfolio Performance Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invested (USD)", f"${total_invested:,.2f}")
    col2.metric("Current Value (USD)", f"${total_invested + unrealized_total:,.2f}")
    col3.metric("Net PnL (USD)", f"${overall_pnl:,.2f}")
    col4.metric("ROI", f"{roi_percentage:+.1f}%")
    
    # Gauge and Speedometer Charts
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
    st.write("---")
    st.write("**📊 Performance Assessment**")

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
def pnl_analysis_page(realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested):
    st.header("💹 PnL Analysis")
    
    # Detailed Breakdown
    st.write("**📋 Detailed Breakdown**")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    with detail_col1:
        st.write("**Realized PnL**")
        st.metric("Amount", f"${realized_total:,.2f}")
        st.metric("% of Investment", f"{(realized_total/total_invested)*100:+.1f}%" if total_invested > 0 else "N/A")
    
    with detail_col2:
        st.write("**Unrealized PnL**")
        st.metric("Amount", f"${unrealized_total:,.2f}")
        st.metric("% of Investment", f"{(unrealized_total/total_invested)*100:+.1f}%" if total_invested > 0 else "N/A")
    
    with detail_col3:
        st.write("**Costs & Efficiency**")
        st.metric("Gas Costs", f"${total_gas_costs:,.2f}")
        st.metric("Cost Ratio", f"{(total_gas_costs/total_invested)*100:.1f}%" if total_invested > 0 else "N/A")

    # PnL Breakdown by Token
    if not breakdown_df.empty:
        st.subheader("💹 PnL Breakdown by Token")
        st.dataframe(breakdown_df, use_container_width=True, height=320)
        
        # Visualizations
        st.subheader("📊 PnL Visualizations")
        
        # Check if we have enough space for side-by-side layout
        enough_space = len(breakdown_df) <= 8
        
        if enough_space:
            # Side-by-side layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Realized PnL Bar Chart
                realized_chart_data = breakdown_df[['Token', 'Realized PnL (USD)']].copy()
                realized_chart_data = realized_chart_data.sort_values('Realized PnL (USD)', ascending=True)
                
                st.write("**💰 Realized PnL by Token**")
                st.bar_chart(
                    realized_chart_data.set_index('Token'),
                    color="#4CAF50"
                )
                
                # Summary stats
                total_realized = realized_chart_data['Realized PnL (USD)'].sum()
                st.caption(f"Total Realized: ${total_realized:,.2f}")
            
            with col2:
                # Unrealized PnL Bar Chart
                unrealized_chart_data = breakdown_df[['Token', 'Unrealized PnL (USD)']].copy()
                unrealized_chart_data = unrealized_chart_data.sort_values('Unrealized PnL (USD)', ascending=True)
                
                st.write("**📈 Unrealized PnL by Token**")
                # Create custom colors for positive/negative
                colors = [
                    "#4CAF50" if x >= 0 else "#F44336" 
                    for x in unrealized_chart_data['Unrealized PnL (USD)']
                ]
                
                # Using matplotlib for better color control
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    tokens = unrealized_chart_data['Token']
                    values = unrealized_chart_data['Unrealized PnL (USD)']
                    
                    colors = ['#4CAF50' if x >= 0 else '#F44336' for x in values]
                    
                    bars = ax.barh(tokens, values, color=colors)
                    ax.set_xlabel('Unrealized PnL (USD)')
                    ax.set_title('Unrealized PnL by Token')
                    
                    # Add value labels on bars
                    for bar in bars:
                        width = bar.get_width()
                        label_x_pos = width + (0.01 * max(values)) if width >= 0 else width - (0.01 * abs(min(values)))
                        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                               f'${width:,.0f}', ha='left' if width >= 0 else 'right', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except ImportError:
                    # Fallback to streamlit bar chart
                    st.bar_chart(
                        unrealized_chart_data.set_index('Token'),
                        color=colors
                    )
                
                # Summary stats
                total_unrealized = unrealized_chart_data['Unrealized PnL (USD)'].sum()
                st.caption(f"Total Unrealized: ${total_unrealized:,.2f}")
        
        else:
            # Stacked layout (not enough space)
            # Realized PnL Bar Chart - Top
            st.write("**💰 Realized PnL by Token**")
            realized_chart_data = breakdown_df[['Token', 'Realized PnL (USD)']].copy()
            realized_chart_data = realized_chart_data.sort_values('Realized PnL (USD)', ascending=True)
            
            st.bar_chart(
                realized_chart_data.set_index('Token'),
                color="#4CAF50"
            )
            
            total_realized = realized_chart_data['Realized PnL (USD)'].sum()
            st.caption(f"Total Realized PnL: ${total_realized:,.2f}")
            
            # Add some spacing
            st.write("")
            st.write("")
            
            # Unrealized PnL Bar Chart - Bottom
            st.write("**📈 Unrealized PnL by Token**")
            unrealized_chart_data = breakdown_df[['Token', 'Unrealized PnL (USD)']].copy()
            unrealized_chart_data = unrealized_chart_data.sort_values('Unrealized PnL (USD)', ascending=True)
            
            # Using matplotlib for better color control
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                tokens = unrealized_chart_data['Token']
                values = unrealized_chart_data['Unrealized PnL (USD)']
                
                colors = ['#4CAF50' if x >= 0 else '#F44336' for x in values]
                
                bars = ax.barh(tokens, values, color=colors)
                ax.set_xlabel('Unrealized PnL (USD)')
                ax.set_title('Unrealized PnL by Token')
                
                # Add value labels on bars
                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width + (0.01 * max(values)) if width >= 0 else width - (0.01 * abs(min(values)))
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                           f'${width:,.0f}', ha='left' if width >= 0 else 'right', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except ImportError:
                # Fallback to streamlit bar chart
                colors = [
                    "#4CAF50" if x >= 0 else "#F44336" 
                    for x in unrealized_chart_data['Unrealized PnL (USD)']
                ]
                st.bar_chart(
                    unrealized_chart_data.set_index('Token'),
                    color=colors
                )
            
            total_unrealized = unrealized_chart_data['Unrealized PnL (USD)'].sum()
            st.caption(f"Total Unrealized PnL: ${total_unrealized:,.2f}")

        # Pie Charts for Top Tokens by PnL
        st.subheader("📊 Top Tokens by PnL")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 5 tokens by Realized PnL
            realized_top5 = breakdown_df.nlargest(5, 'Realized PnL (USD)')
            if not realized_top5.empty:
                # Calculate percentages for tooltips
                total_realized = realized_top5['Realized PnL (USD)'].sum()
                realized_top5['Percentage'] = (realized_top5['Realized PnL (USD)'] / total_realized * 100).round(1)
                
                fig_realized = px.pie(
                    realized_top5,
                    values='Realized PnL (USD)',
                    names='Token',
                    title='💰 Top 5 Tokens by Realized PnL',
                    hover_data=['Percentage'],
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                fig_realized.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>PnL: $%{value:,.2f}<br>Share: %{customdata[0]}%<extra></extra>'
                )
                fig_realized.update_layout(
                    showlegend=False,
                    title_x=0.5,
                    title_font_size=16
                )
                st.plotly_chart(fig_realized, use_container_width=True)
                
                # Show summary stats
                st.caption(f"Total Realized PnL: ${realized_top5['Realized PnL (USD)'].sum():,.2f}")
            else:
                st.info("No realized PnL data available")
        
        with col2:
            # Top 5 tokens by Unrealized PnL (absolute value)
            unrealized_top5 = breakdown_df.reindex(
                breakdown_df['Unrealized PnL (USD)'].abs().nlargest(5).index
            )
            if not unrealized_top5.empty:
                # Calculate percentages for tooltips
                total_unrealized_abs = unrealized_top5['Unrealized PnL (USD)'].abs().sum()
                unrealized_top5['Percentage'] = (unrealized_top5['Unrealized PnL (USD)'].abs() / total_unrealized_abs * 100).round(1)
                
                # Use different colors for positive and negative PnL
                colors = ['green' if x >= 0 else 'red' for x in unrealized_top5['Unrealized PnL (USD)']]
                
                fig_unrealized = px.pie(
                    unrealized_top5,
                    values='Unrealized PnL (USD)',
                    names='Token',
                    title='📈 Top 5 Tokens by Unrealized PnL',
                    hover_data=['Percentage'],
                    color=colors
                )
                fig_unrealized.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>PnL: $%{value:,.2f}<br>Share: %{customdata[0]}%<extra></extra>'
                )
                fig_unrealized.update_layout(
                    showlegend=False,
                    title_x=0.5,
                    title_font_size=16
                )
                st.plotly_chart(fig_unrealized, use_container_width=True)
                
                # Show summary stats
                total_unrealized = unrealized_top5['Unrealized PnL (USD)'].sum()
                st.caption(f"Total Unrealized PnL: ${total_unrealized:,.2f}")
            else:
                st.info("No unrealized PnL data available")

# -------------------------------
# Transactions Page
# -------------------------------
def transactions_page(df):
    st.header("📊 Transaction Details")
    
    # Display transactions table
    st.dataframe(df, use_container_width=True, height=600)
    
    # Transaction statistics
    st.subheader("📈 Transaction Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Total transactions
    total_txs = len(df)
    col1.metric("Total Transactions", f"{total_txs:,}")
    
    # Transactions by type
    tx_types = df['transaction_type'].value_counts()
    col2.metric("Transaction Types", f"{len(tx_types)}")
    
    # Date range
    if not df.empty:
        min_date = df['block_time'].min().strftime('%Y-%m-%d')
        max_date = df['block_time'].max().strftime('%Y-%m-%d')
        col3.metric("Date Range", f"{min_date} to {max_date}")
    
    # Unique tokens
    unique_tokens = df['token_symbol'].nunique()
    col4.metric("Unique Tokens", f"{unique_tokens}")
    
    # Transactions over time chart
    st.subheader("📅 Transactions Over Time")
    
    if not df.empty:
        # Group by date
        df_date = df.copy()
        df_date['date'] = df_date['block_time'].dt.date
        daily_counts = df_date.groupby('date').size().reset_index(name='count')
        
        # Create chart
        fig = px.line(
            daily_counts, 
            x='date', 
            y='count', 
            title='Daily Transaction Count',
            labels={'date': 'Date', 'count': 'Number of Transactions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Token distribution
    st.subheader("🪙 Token Distribution")
    
    if not df.empty:
        token_counts = df['token_symbol'].value_counts().reset_index()
        token_counts.columns = ['Token', 'Count']
        
        fig = px.pie(
            token_counts.head(10),  # Top 10 tokens
            values='Count', 
            names='Token', 
            title='Top 10 Tokens by Transaction Count'
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Settings & Diagnostics Page
# -------------------------------
def settings_diagnostics_page(df, realized_total, unrealized_total, total_gas_costs, breakdown_df, sidebar_params):
    st.header("⚙️ Settings & Diagnostics")
    
    # Cache management
    st.subheader("📂 Cache Management")
    
    if st.button("🔄 Clear All Cache"):
        # Implementation for clearing cache
        st.warning("Cache clearing functionality would be implemented here")
    
    if st.button("📊 Export Data to CSV"):
        # Implementation for exporting data
        st.info("Data export functionality would be implemented here")
    
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
                st.dataframe(validation_df, use_container_width=True)
                st.write("**Issues found:**")
                for _, row in failed_validations.iterrows():
                    st.write(f"- {row['Check']}: Failed")
        else:
            st.success("✅ PnL Calculations Validated Successfully")
            with st.expander("View Validation Details"):
                st.dataframe(validation_df, use_container_width=True)
    
    # System information
    st.subheader("🖥️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Python Version**")
        st.code("3.x")
        
        st.write("**Streamlit Version**")
        st.code(st.__version__)
    
    with col2:
        st.write("**Pandas Version**")
        st.code(pd.__version__)
        
        st.write("**Cache Directory**")
        st.code(CACHE_DIR)
    
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
        
        st.dataframe(quality_df[quality_df['Missing Values'] > 0], use_container_width=True)
        
        if quality_df[quality_df['Missing Values'] > 0].empty:
            st.success("✅ No missing values found in the dataset")
        else:
            st.warning("⚠️ Missing values detected in some columns")

# -------------------------------
# Main function
# -------------------------------
def main():
    # Setup navigation
    page = setup_navigation()
    
    # Setup sidebar (shared across all pages)
    sidebar_params = setup_sidebar()
    
    # Load data (shared across all pages)
    data = load_data(sidebar_params)
    
    if data[0] is None:
        # No data available
        return
    
    df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested = data
    
    # Display appropriate page based on navigation
    if page == "Dashboard":
        dashboard_page(df, realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested)
    elif page == "PnL Analysis":
        pnl_analysis_page(realized_total, unrealized_total, total_gas_costs, overall_pnl, roi_percentage, breakdown_df, total_invested)
    elif page == "Transactions":
        transactions_page(df)
    elif page == "Settings & Diagnostics":
        settings_diagnostics_page(df, realized_total, unrealized_total, total_gas_costs, breakdown_df, sidebar_params)

if __name__ == "__main__":
    main()