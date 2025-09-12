import os
import json
import time
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from price_fetcher import get_token_price as fetch_token_price
from price_fetcher import fill_missing_prices_batch
import streamlit as st  

# -------------------------------
# Cache for prices
# -------------------------------
class PriceCache:
    def __init__(self, filename="price_cache.json"):
        self.filename = filename
        self.cache = {}
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    self.cache = json.load(f)
                except:
                    self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value):
        self.cache[key] = value
        with open(self.filename, "w") as f:
            json.dump(self.cache, f)

# -------------------------------
# Cache for contract-to-CGID mapping
# -------------------------------
class AddressCache:
    def __init__(self, filename="address_to_cgid.json"):
        self.filename = filename
        self.cache = {}
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    self.cache = json.load(f)
                except:
                    self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value):
        self.cache[key] = value
        with open(self.filename, "w") as f:
            json.dump(self.cache, f)

# -------------------------------
# Extended Analyzer
# -------------------------------
class ExtendedMoralisAnalyzer:
    def __init__(self, api_key: str, use_cache: bool = True, force_refresh: bool = False, diagnostic_mode: bool = False):
        self.api_key = api_key
        self.use_cache = use_cache
        self.force_refresh = force_refresh
        self.diagnostic_mode = diagnostic_mode

        self.base_url = "https://deep-index.moralis.io/api/v2"
        self.headers = {"Accept": "application/json", "X-API-Key": api_key}

        self.chains = {
            'eth': '0x1',
            'bsc': '0x38',
            'polygon': '0x89',
            'arbitrum': '0xa4b1',
            'optimism': '0xa',
            'base': '0x2105'
        }

        self.price_cache = PriceCache()
        self.address_cache = AddressCache()
        self.price_cache = PriceCache()
        self.address_cache = AddressCache()

    # -------------------------------
    # Fetch ERC20 Transfers
    # -------------------------------
    def get_erc20_transfers(self, wallet: str, chain: str, limit: int = 50) -> List[Dict]:
        try:
            url = f"{self.base_url}/{wallet}/erc20/transfers"
            params = {"chain": chain, "limit": limit}
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json().get('result', [])
            else:
                print(f"Moralis API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"ERC20 transfer error: {e}")
            return []

    # -------------------------------
    # Fetch Tx Gas Cost (in native coin)
    # -------------------------------
    def get_tx_gas_cost(self, tx_hash: str, chain: str) -> Optional[float]:
        try:
            url = f"{self.base_url}/transaction/{tx_hash}"
            params = {"chain": chain}
            r = requests.get(url, headers=self.headers, params=params)
            if r.status_code == 200:
                data = r.json()
                gas_used = int(data.get("receipt_gas_used") or 0)
                gas_price = int(data.get("gas_price") or 0)
                native_spent = gas_used * gas_price / 1e18
                return native_spent
        except Exception as e:
            print(f"Gas fetch failed for {tx_hash}: {e}")
        return None

    # -------------------------------
    # PRICE FETCHER (Coingecko)
    # -------------------------------
    def get_price_usd(self, symbol: str, timestamp: str, token_address: str = None, blockchain: str = "ethereum") -> Optional[float]:
    
        if not symbol:
            return None

        date_str = timestamp.split("T")[0]
        cache_key = f"{symbol}_{date_str}"
        cached = self.price_cache.get(cache_key)
        if cached:
            if self.diagnostic_mode:
                st.info(f"Cache hit for {symbol} on {date_str}: ${cached}")
            return cached

        cg_id = None

        try:
            native_tokens = {
                'eth': 'ethereum',
                'bnb': 'binancecoin',
                'matic': 'polygon',
                'polygon': 'polygon',
                'arbitrum': 'ethereum',
                'optimism': 'ethereum',
                'base': 'ethereum'
            }

            if symbol.lower() in native_tokens:
                cg_id = native_tokens[symbol.lower()]
            elif blockchain.lower() in native_tokens and symbol.lower() in ['eth', 'bnb', 'matic']:
                cg_id = native_tokens[blockchain.lower()]

            if not cg_id and token_address:
                cached_cgid = self.address_cache.get(token_address.lower())
                if cached_cgid:
                    cg_id = cached_cgid
                    if self.diagnostic_mode:
                        st.info(f"Address cache hit for {token_address}: {cg_id}")
                else:
                    url = f"https://api.coingecko.com/api/v3/coins/{blockchain}/contract/{token_address}"
                    r = requests.get(url)
                    if self.diagnostic_mode:
                        st.info(f"Response status: {r.status_code} for contract {token_address}")
                    if r.status_code == 200:
                        data = r.json()
                        cg_id = data.get("id")
                        if cg_id:
                            self.address_cache.set(token_address.lower(), cg_id)
                            if self.diagnostic_mode:
                                st.info(f"Resolved CoinGecko ID for {token_address}: {cg_id}")
                    else:
                        if self.diagnostic_mode:
                            st.warning(f"Failed to resolve CoinGecko ID for {token_address}: {r.status_code}")

            if not cg_id:
                mapping = {
                    "weth": "weth",
                    "usdc": "usd-coin",
                    "usdt": "tether",
                    "dai": "dai",
                    "wbtc": "wrapped-bitcoin",
                    "link": "chainlink",
                    "arb": "arbitrum"
                }
                cg_id = mapping.get(symbol.lower())
                if not cg_id and self.diagnostic_mode:
                    st.warning(f"⚠️ No CoinGecko ID for {symbol}")

            if not cg_id:
                return None

            url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/history"
            params = {"date": datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m-%Y")}
            r = requests.get(url, params=params)

            if self.diagnostic_mode:
                st.info(f"Response status: {r.status_code} for {symbol} history on {date_str}")

            if r.status_code == 200:
                data = r.json()
                price = data.get("market_data", {}).get("current_price", {}).get("usd")
                if price:
                    self.price_cache.set(cache_key, price)
                    if self.diagnostic_mode:
                        st.info(f"Retrieved {symbol} price for {date_str}: ${price}")
                    return price
                else:
                    if self.diagnostic_mode:
                        st.warning(f"No price data found for {symbol} on {date_str}")
            else:
                if self.diagnostic_mode:
                    st.warning(f"CoinGecko API error for {symbol}: {r.status_code}")

        except Exception as e:
            if self.diagnostic_mode:
                st.error(f"Price fetch error for {symbol}: {e}")

        return None

    # -------------------------------
    # Get current prices for unrealized PnL
    # -------------------------------
    def get_current_prices(self, tokens: List[Dict]) -> Dict[str, float]:
    
        prices = {}
        coingecko_ids = []
        token_map = {}

        for token in tokens:
            symbol = token.get("symbol", "")
            address = token.get("address", "")
            blockchain = token.get("blockchain", "ethereum")

            cache_key = f"current_{address.lower()}"
            cached = self.price_cache.get(cache_key)
            if cached:
                prices[address] = cached
                if self.diagnostic_mode:
                    st.info(f"Cache hit for current price of {symbol} ({address}): ${cached}")
                continue

            cg_id = None

            if address:
                cached_cgid = self.address_cache.get(address.lower())
                if cached_cgid:
                    cg_id = cached_cgid
                    if self.diagnostic_mode:
                        st.info(f"Address cache hit for {address}: {cg_id}")
                else:
                    try:
                        url = f"https://api.coingecko.com/api/v3/coins/{blockchain}/contract/{address}"
                        r = requests.get(url)
                        if self.diagnostic_mode:
                            st.info(f"Response status: {r.status_code} for contract {address}")
                        if r.status_code == 200:
                            data = r.json()
                            cg_id = data.get("id")
                            if cg_id:
                                self.address_cache.set(address.lower(), cg_id)
                                if self.diagnostic_mode:
                                    st.info(f"Resolved CoinGecko ID for {address}: {cg_id}")
                    except Exception as e:
                        if self.diagnostic_mode:
                            st.error(f"Error resolving CG ID for {address}: {e}")

            if not cg_id and symbol:
                mapping = {
                    "eth": "ethereum",
                    "weth": "weth",
                    "usdc": "usd-coin",
                    "usdt": "tether",
                    "bnb": "binancecoin",
                    "matic": "polygon",
                    "arb": "arbitrum",
                    "op": "optimism"
                }
                cg_id = mapping.get(symbol.lower())
                if not cg_id and self.diagnostic_mode:
                    st.warning(f"⚠️ No CoinGecko ID for {symbol}")

            if cg_id:
                coingecko_ids.append(cg_id)
                token_map[cg_id] = address

        if coingecko_ids:
            try:
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {
                    "ids": ",".join(coingecko_ids),
                    "vs_currencies": "usd"
                }
                r = requests.get(url, params=params)
                if self.diagnostic_mode:
                    st.info(f"Response status: {r.status_code} for current prices batch")
                if r.status_code == 200:
                    data = r.json()
                    for cg_id, price_data in data.items():
                        if "usd" in price_data:
                            token_address = token_map[cg_id]
                            price = price_data["usd"]
                            prices[token_address] = price
                            cache_key = f"current_{token_address.lower()}"
                            self.price_cache.set(cache_key, price)
                            if self.diagnostic_mode:
                                st.info(f"Retrieved current price for {cg_id} ({token_address}): ${price}")
            except Exception as e:
                if self.diagnostic_mode:
                    st.error(f"Error fetching current prices: {e}")

        return prices
    
    def get_native_token_price(self, chain_name: str, timestamp: str) -> Optional[float]:
        
        native_token_map = {
            'eth': 'ethereum',
            'bsc': 'binancecoin',
            'polygon': 'polygon',
            'arbitrum': 'ethereum',
            'optimism': 'ethereum',
            'base': 'ethereum'
        }

        token_symbol = native_token_map.get(chain_name.lower(), 'ethereum')
        price = self.get_price_usd(token_symbol, timestamp, None, chain_name)

        if price is None or price <= 0:
            fallback_prices = {
                'ethereum': 3000.0,
                'polygon': 0.7,
                'binancecoin': 500.0,
            }
            price = fallback_prices.get(token_symbol, 0)
            if price > 0 and self.diagnostic_mode:
                st.info(f"Using fallback price for {chain_name}: ${price}")

        return price

    # -------------------------------
    # Enrich transfers with USD price + Gas cost - THIS WAS MISSING!
    # -------------------------------
    def get_detailed_data_for_wallet(self, wallet: str, max_per_chain: int = 50, chains: List[str] = None) -> pd.DataFrame:
        all_tx = []
        chains_to_fetch = chains if chains else list(self.chains.keys())
        
        for chain_name in chains_to_fetch:
            if chain_name not in self.chains:
                continue
                
            chain_id = self.chains[chain_name]
            print(f"Fetching ERC20 transfers on {chain_name}...")
            erc20_txs = self.get_erc20_transfers(wallet, chain=chain_id, limit=max_per_chain)
            
            if not erc20_txs:
                print(f"No transactions found on {chain_name}")
                continue
            
            for tx in erc20_txs:
                try:
                    decimals = int(tx.get("token_decimals") or 18)
                    raw_value = float(tx.get("value") or 0)
                    amount = raw_value / (10 ** decimals)
                    timestamp = tx.get("block_timestamp")

                    symbol = tx.get("token_symbol", "")
                    token_address = tx.get("address", "")
                    price_usd = self.get_price_usd(symbol, timestamp, token_address, chain_name) or 0
                    usd_value = amount * price_usd

                    # Gas cost (native coin)
                    tx_hash = tx.get("transaction_hash")
                    gas_native = self.get_tx_gas_cost(tx_hash, chain_id) or 0
                    
                    # Calculate gas cost in USD
                    native_token_price = self.get_native_token_price(chain_name, timestamp)
                    
                    # Debug logging
                    if gas_native > 0:
                        print(f"Chain: {chain_name}, Gas Native: {gas_native}, Native Token Price: {native_token_price}")
                    
                    gas_usd = gas_native * native_token_price if native_token_price else 0
                    
                    if gas_native > 0 and gas_usd == 0:
                        print(f"WARNING: Gas cost calculation failed for tx {tx_hash}")
                        print(f"  Chain: {chain_name}")
                        print(f"  Gas Native: {gas_native}")
                        print(f"  Native Token Price: {native_token_price}")

                    # Determine transaction type
                    from_addr = tx.get("from_address", "").lower()
                    to_addr = tx.get("to_address", "").lower()
                    wallet_lower = wallet.lower()
                    
                    if to_addr == wallet_lower:
                        tx_type = "deposit"
                    elif from_addr == wallet_lower:
                        tx_type = "withdrawal"
                    else:
                        tx_type = "unknown"

                    enriched = {
                        "wallet": wallet,
                        "blockchain": chain_name,
                        "tx_hash": tx_hash,
                        "block_time": timestamp,
                        "token_symbol": symbol,
                        "token_address": token_address,
                        "amount": amount,
                        "price_usd": price_usd,
                        "usd_value": usd_value,
                        "gas_cost_native": gas_native,
                        "gas_cost_usd": gas_usd,
                        "transaction_type": tx_type,
                        "from_address": from_addr,
                        "to_address": to_addr
                    }
                    all_tx.append(enriched)
                except Exception as e:
                    print(f"Error processing tx: {e}")
                    continue

            time.sleep(0.5)  # rate limit

        if not all_tx:
            print("No transactions found across all chains")
            return pd.DataFrame()

        df = pd.DataFrame(all_tx)
        df['block_time'] = pd.to_datetime(df['block_time'])
        df = df.sort_values("block_time").reset_index(drop=True)
        
        print(f"Successfully processed {len(df)} transactions")
        return df

# -------------------------------
# PnL CALCULATION - OUTSIDE THE CLASS
# -------------------------------
def calculate_pnl_improved(df, method="FIFO", analyzer=None):
    """
    FIXED: Improved PnL calculation that includes gas costs
    """
    # Validate inputs
    if df.empty:
        return 0, 0, 0, pd.DataFrame(columns=["Token", "Realized PnL (USD)", "Unrealized PnL (USD)", "Current Holdings", "Avg Cost", "Current Price"])
    
    # Ensure proper sorting by time
    df = df.sort_values("block_time").reset_index(drop=True)
    
    positions = {}   # token -> list of lots (FIFO/LIFO)
    avg_costs = {}   # token -> (total_qty, total_cost_basis) for ACB
    realized_pnl = 0
    total_gas_cost = 0  # Track total gas costs
    token_realized = {}
    token_unrealized = {}
    
    print(f"Processing {len(df)} transactions using {method} method...")
    
    for idx, row in df.iterrows():
        token = row.get("token_symbol", "")
        if not token:
            continue
            
        # Handle different quantity column names
        qty = row.get("amount", 0)
        price = row.get("price_usd", 0)
        tx_type = row.get("transaction_type", "")
        gas_cost = row.get("gas_cost_usd", 0)
        
        # Add gas cost to total (for all transactions)
        total_gas_cost += gas_cost
        
        # Skip invalid transactions
        if pd.isna(price) or price <= 0:
            print(f"Missing/invalid price for {token}, skipping transaction")
            continue

        # Skip invalid transactions
        if qty <= 0:
            print(f"Skipping invalid transaction: qty={qty}, type={tx_type}")
            continue
        
        # Classify transactions into buys/sells
        is_buy = tx_type in ["deposit", "buy", "swap_in", "mint", "receive", "incoming"]
        is_sell = tx_type in ["withdrawal", "sell", "swap_out", "burn", "send", "outgoing"]
        
        # Skip non-trading transactions
        if not (is_buy or is_sell):
            print(f"Skipping non-trading transaction type: {tx_type}")
            continue
            
        # --- FIFO / LIFO Logic ---
        if method in ["FIFO", "LIFO"]:
            if is_buy:
                # Add to position (include gas cost in cost basis)
                effective_cost = price + (gas_cost / qty) if qty > 0 else price
                lot = {"qty": float(qty), "cost": float(effective_cost)}
                positions.setdefault(token, []).append(lot)
                print(f"Added lot: {qty} {token} @ ${effective_cost:.4f} (includes gas)")
                
            elif is_sell:
                # Sell from position
                if token not in positions or not positions[token]:
                    print(f"WARNING: Selling {qty} {token} with no position!")
                    # Still record as realized loss (assuming cost basis = 0)
                    pnl_piece = qty * price - gas_cost
                    realized_pnl += pnl_piece
                    token_realized[token] = token_realized.get(token, 0) + pnl_piece
                    continue
                
                remaining_to_sell = float(qty)
                sell_price = float(price)
                
                while remaining_to_sell > 0 and positions[token]:
                    # Get lot based on method
                    lot_idx = 0 if method == "FIFO" else -1
                    lot = positions[token][lot_idx]
                    
                    lot_qty = lot["qty"]
                    lot_cost = lot["cost"]
                    
                    # Determine how much to sell from this lot
                    qty_to_sell = min(remaining_to_sell, lot_qty)
                    
                    # Calculate PnL for this portion (subtract gas cost)
                    proceeds = qty_to_sell * sell_price
                    cost_basis = qty_to_sell * lot_cost
                    pnl_piece = proceeds - cost_basis - (gas_cost * (qty_to_sell / qty)) if qty > 0 else proceeds - cost_basis
                    
                    realized_pnl += pnl_piece
                    token_realized[token] = token_realized.get(token, 0) + pnl_piece
                    
                    print(f"Sold {qty_to_sell} {token}: ${proceeds:.2f} - ${cost_basis:.2f} - ${gas_cost*(qty_to_sell/qty):.2f} gas = ${pnl_piece:.2f} PnL")
                    
                    # Update lot and remaining
                    lot["qty"] -= qty_to_sell
                    remaining_to_sell -= qty_to_sell
                    
                    # Remove empty lots
                    if lot["qty"] <= 0:
                        positions[token].pop(lot_idx)
        
        # --- ACB (Average Cost Basis) Logic ---
        elif method == "ACB":
            if is_buy:
                # Update average cost basis (include gas cost)
                current_qty, current_total_cost = avg_costs.get(token, (0, 0))
                additional_cost = (qty * price) + gas_cost
                new_qty = current_qty + qty
                new_total_cost = current_total_cost + additional_cost
                avg_costs[token] = (new_qty, new_total_cost)
                print(f"ACB updated for {token}: {new_qty} units, total cost ${new_total_cost:.2f} (includes gas)")
                
            elif is_sell:
                current_qty, current_total_cost = avg_costs.get(token, (0, 0))
                
                if current_qty <= 0:
                    print(f"WARNING: Selling {qty} {token} with no ACB position!")
                    pnl_piece = qty * price - gas_cost
                    realized_pnl += pnl_piece
                    token_realized[token] = token_realized.get(token, 0) + pnl_piece
                    continue
                
                # Calculate average cost
                avg_cost = current_total_cost / current_qty if current_qty > 0 else 0
                
                # Calculate PnL
                qty_to_sell = min(qty, current_qty)
                proceeds = qty_to_sell * price
                cost_basis = qty_to_sell * avg_cost
                pnl_piece = proceeds - cost_basis - gas_cost
                
                realized_pnl += pnl_piece
                token_realized[token] = token_realized.get(token, 0) + pnl_piece
                
                print(f"ACB sale: {qty_to_sell} {token} @ ${price} - ${cost_basis:.2f} cost - ${gas_cost:.2f} gas = ${pnl_piece:.2f} PnL")
                
                # Update position
                new_qty = max(0, current_qty - qty_to_sell)
                new_total_cost = max(0, current_total_cost - (qty_to_sell * avg_cost))
                avg_costs[token] = (new_qty, new_total_cost)
    
    # --- Calculate Unrealized PnL with CURRENT PRICES ---
    print("\nCalculating unrealized PnL with current market prices...")
    unrealized_pnl = 0.0
    token_holdings = {}
    current_prices = {}

    # Collect tokens for current price lookup
    tokens_for_current_prices = []

    if method in ["FIFO", "LIFO"]:
        for token, lots in positions.items():
            if lots:
                token_df = df[df["token_symbol"] == token]
                if not token_df.empty:
                    token_row = token_df.iloc[0]
                    tokens_for_current_prices.append({
                        "symbol": token,
                        "address": token_row.get("token_address", ""),
                        "blockchain": token_row.get("blockchain", "ethereum")
                    })

    elif method == "ACB":
        for token, (total_qty, _) in avg_costs.items():
            if total_qty > 0:
                token_df = df[df["token_symbol"] == token]
                if not token_df.empty:
                    token_row = token_df.iloc[0]
                    tokens_for_current_prices.append({
                        "symbol": token,
                        "address": token_row.get("token_address", ""),
                        "blockchain": token_row.get("blockchain", "ethereum")
                    })

    # Fetch current prices if analyzer is available
    if analyzer and tokens_for_current_prices:
        try:
            current_prices = analyzer.get_current_prices(tokens_for_current_prices)
            print(f"Fetched current prices for {len(current_prices)} tokens")
        except Exception as e:
            print(f"Error fetching current prices: {e}")
            # Final fallback: Use last transaction prices
            for token_info in tokens_for_current_prices:
                token_df = df[df["token_symbol"] == token_info["symbol"]]
                if not token_df.empty:
                    current_prices[token_info["address"]] = token_df.iloc[-1]["price_usd"]
    else:
        print("Using last transaction prices as current prices")
        for token in set(df["token_symbol"]):
            token_df = df[df["token_symbol"] == token]
            if not token_df.empty:
                token_addr = token_df.iloc[-1]["token_address"]
                current_prices[token_addr] = token_df.iloc[-1]["price_usd"]
    
    # Calculate unrealized PnL for FIFO/LIFO
    if method in ["FIFO", "LIFO"]:
        for token, lots in positions.items():
            if not lots:
                continue
                
            token_df = df[df["token_symbol"] == token]
            if token_df.empty:
                continue
                
            token_addr = token_df.iloc[0]["token_address"]
            current_price = current_prices.get(token_addr, 0)
            
            if current_price <= 0:
                print(f"No current price available for {token}, skipping unrealized PnL")
                continue
            
            total_qty = sum(lot["qty"] for lot in lots)
            total_cost_basis = sum(lot["qty"] * lot["cost"] for lot in lots)
            
            current_value = total_qty * current_price
            unrealized_pnl_token = current_value - total_cost_basis
            
            unrealized_pnl += unrealized_pnl_token
            token_unrealized[token] = unrealized_pnl_token
            token_holdings[token] = {
                "qty": total_qty,
                "avg_cost": total_cost_basis / total_qty if total_qty > 0 else 0,
                "current_price": current_price,
                "current_value": current_value
            }
            
            print(f"{token}: {total_qty:.4f} units @ avg ${total_cost_basis/total_qty:.4f}, current ${current_price:.4f} = ${unrealized_pnl_token:.2f} unrealized")
    
    # Calculate unrealized PnL for ACB
    elif method == "ACB":
        for token, (total_qty, total_cost_basis) in avg_costs.items():
            if total_qty <= 0:
                continue
                
            token_df = df[df["token_symbol"] == token]
            if token_df.empty:
                continue
            
            token_addr = token_df.iloc[0]["token_address"]
            current_price = current_prices.get(token_addr, 0)
            
            if current_price <= 0:
                print(f"No current price available for {token}, skipping unrealized PnL")
                continue
            
            avg_cost = total_cost_basis / total_qty if total_qty > 0 else 0
            
            current_value = total_qty * current_price
            unrealized_pnl_token = current_value - total_cost_basis
            
            unrealized_pnl += unrealized_pnl_token
            token_unrealized[token] = unrealized_pnl_token
            token_holdings[token] = {
                "qty": total_qty,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "current_value": current_value
            }
            
            print(f"{token}: {total_qty:.4f} units @ avg ${avg_cost:.4f}, current ${current_price:.4f} = ${unrealized_pnl_token:.2f} unrealized")
    
    # --- Build detailed breakdown ---
    all_tokens = set(token_realized.keys()).union(token_unrealized.keys())
    token_data = []
    
    for token in all_tokens:
        holdings = token_holdings.get(token, {})
        token_data.append({
            "Token": token,
            "Realized PnL (USD)": round(token_realized.get(token, 0.0), 2),
            "Unrealized PnL (USD)": round(token_unrealized.get(token, 0.0), 2),
            "Current Holdings": round(holdings.get("qty", 0.0), 6),
            "Avg Cost": round(holdings.get("avg_cost", 0.0), 4),
            "Current Price": round(holdings.get("current_price", 0.0), 4),
            "Current Value": round(holdings.get("current_value", 0.0), 2)
        })
    
    breakdown_df = pd.DataFrame(token_data)
    if not breakdown_df.empty:
        breakdown_df = breakdown_df.sort_values("Realized PnL (USD)", ascending=False)
    
    # Update final results to include gas costs
    print(f"\nFinal Results:")
    print(f"Realized PnL: ${realized_pnl:.2f}")
    print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
    print(f"Total Gas Costs: ${total_gas_cost:.2f}")

    return realized_pnl, unrealized_pnl, total_gas_cost, breakdown_df

# -------------------------------
# PnL VALIDATION FUNCTION
# -------------------------------
def validate_pnl_calculation(df, realized_pnl, unrealized_pnl, total_gas_cost, breakdown_df):
    """
    Validate PnL calculations including gas costs
    """
    validation_results = []
    
    # Check 1: Total PnL components should sum correctly
    breakdown_realized_sum = breakdown_df["Realized PnL (USD)"].sum() if not breakdown_df.empty else 0
    breakdown_unrealized_sum = breakdown_df["Unrealized PnL (USD)"].sum() if not breakdown_df.empty else 0
    
    validation_results.append({
        "Check": "PnL Components Sum",
        "Expected Realized": realized_pnl,
        "Breakdown Realized": breakdown_realized_sum,
        "Expected Unrealized": unrealized_pnl,
        "Breakdown Unrealized": breakdown_unrealized_sum,
        "Pass": abs(realized_pnl - breakdown_realized_sum) < 0.01 and abs(unrealized_pnl - breakdown_unrealized_sum) < 0.01
    })
    
    # Check 2: Gas costs should match sum of all transaction gas costs
    total_gas_from_data = df["gas_cost_usd"].sum() if "gas_cost_usd" in df.columns else 0
    validation_results.append({
        "Check": "Gas Costs Validation",
        "Expected Gas Costs": total_gas_cost,
        "Actual Gas Costs": total_gas_from_data,
        "Pass": abs(total_gas_cost - total_gas_from_data) < 0.01
    })
    
    return pd.DataFrame(validation_results)

# -------------------------------
# Simple token price fetcher (fallback)
# -------------------------------
# In analyzer.py, you should have something like this:

def get_token_price(token_symbol: str, token_address: str = None, blockchain: str = None, block_time: pd.Timestamp = None) -> Optional[float]:
    """
    Enhanced function to get token prices with multiple fallback strategies.
    """
    # First try the advanced price fetcher
    if token_address and blockchain:
        price = fetch_token_price(
            symbol=token_symbol,
            address=token_address,
            chain=blockchain,
            block_time=block_time
        )
        if price is not None and price > 0:
            return price
    
    # Fallback to simple mapping if advanced fetcher fails
    price_mapping = {
        "ETH": 3000.0,
        "BTC": 60000.0,
        "USDC": 1.0,
        "USDT": 1.0,
        "DAI": 1.0,
        "WBTC": 60000.0,
        "WETH": 3000.0,
        "ARB": 1.5,  # Add more tokens as needed
        "MATIC": 0.7,
        "BNB": 500.0,
        "FDUSD": 1.0,
    }
    
    return price_mapping.get(token_symbol.upper())

