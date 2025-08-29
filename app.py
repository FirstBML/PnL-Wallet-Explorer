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

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalablePnLCalculator:
    def __init__(self, coingecko_api_key: str = None):
        self.coingecko_api_key = coingecko_api_key
        self.coingecko_cache = {}
        self.token_mapping_cache = {}
        self.cache_lock = threading.Lock()
        self.session = None
        
    async def init_session(self):
        """Initialize aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _get_coingecko_headers(self):
        """Get headers for CoinGecko API"""
        if self.coingecko_api_key:
            return {'x-cg-pro-api-key': self.coingecko_api_key}
        return {}
    
    async def get_coingecko_id_from_address(self, token_address: str, blockchain: str):
        """Map token address to CoinGecko ID"""
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
            # Try direct contract lookup first (more efficient)
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
        """Get cached historical price"""
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
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    return price
                    
        except Exception as e:
            logger.warning(f"Price fetch failed for {coingecko_id}: {e}")
        
        return None
    
    async def calculate_wallet_pnl(self, df: pd.DataFrame, wallet_address: str):
        """Main PnL calculation method"""
        await self.init_session()
        
        try:
            # Clean data
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
                    continue
                
                finally:
                    processed_count += 1
                    progress = processed_count / len(df_clean)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {processed_count}/{len(df_clean)} transactions")
            
            # Calculate unrealized PnL
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

# Streamlit UI
def main():
    st.set_page_config(page_title="Wallet PnL Calculator", page_icon="💰", layout="wide")
    
    st.title("💰 Wallet PnL Calculator")
    st.markdown("Calculate realized and unrealized PnL using FIFO accounting")
    
    # Sample data upload (replace with your Dune data loading)
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        wallet_address = st.text_input("Wallet Address", value="0x...")
        
        if st.button("Calculate PnL", type="primary"):
            if wallet_address:
                calculator = ScalablePnLCalculator(
                    coingecko_api_key=st.secrets.get("COINGECKO_API_KEY", None)
                )
                
                with st.spinner("Calculating PnL... This may take a few minutes"):
                    try:
                        # Run async calculation
                        result = asyncio.run(calculator.calculate_wallet_pnl(df, wallet_address))
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Realized PnL", f"${result['realized_pnl']:,.2f}")
                        with col2:
                            st.metric("Unrealized PnL", f"${result['unrealized_pnl']:,.2f}")
                        with col3:
                            st.metric("Total PnL", f"${result['total_pnl']:,.2f}")
                        
                        st.success(f"Processed {result['processed_transactions']} transactions successfully!")
                        
                    except Exception as e:
                        st.error(f"Error calculating PnL: {str(e)}")
            else:
                st.warning("Please enter a wallet address")

if __name__ == "__main__":
    main()