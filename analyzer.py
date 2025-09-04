import os
import json
import time
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

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
    def __init__(self, api_key: str, use_cache: bool = True, force_refresh: bool = False):
        # ✅ FIXED: use api_key param, not undefined API_KEY
        self.api_key = api_key
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
        # -------------------------------
    # PRICE FETCHER (Coingecko)
    # -------------------------------
    def get_price_usd(self, symbol: str, timestamp: str, token_address: str = None, blockchain: str = "ethereum") -> Optional[float]:
        if not token_address and not symbol:
            return None

        date_str = timestamp.split("T")[0]
        cache_key = f"{token_address or symbol}_{date_str}"
        cached = self.price_cache.get(cache_key)
        if cached:
            return cached

        cg_id = None

        try:
            # Step 1: prefer contract lookup first
            if token_address:
                # Check local address cache
                cached_cgid = self.address_cache.get(token_address.lower())
                if cached_cgid:
                    cg_id = cached_cgid
                else:
                    # Query Coingecko contract API
                    url = f"https://api.coingecko.com/api/v3/coins/{blockchain}/contract/{token_address}"
                    r = requests.get(url)
                    if r.status_code == 200:
                        data = r.json()
                        cg_id = data.get("id")
                        if cg_id:
                            self.address_cache.set(token_address.lower(), cg_id)

            # Step 2: fallback to hardcoded mapping if contract failed
            if not cg_id and symbol:
                mapping = {
                    "eth": "ethereum",
                    "weth": "weth",
                    "usdc": "usd-coin",
                    "usdt": "tether",
                    "bnb": "binancecoin",
                    "matic": "polygon"
                }
                cg_id = mapping.get(symbol.lower())

            if not cg_id:
                return None

            # Step 3: fetch historical price
            url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/history"
            params = {"date": datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m-%Y")}
            r = requests.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                price = data.get("market_data", {}).get("current_price", {}).get("usd")
                if price:
                    self.price_cache.set(cache_key, price)
                    return price
        except Exception as e:
            print(f"Price fetch error for {symbol} / {token_address}: {e}")
            return None

    # -------------------------------
    # Enrich transfers with USD price + Gas cost
    # -------------------------------
    def get_detailed_data_for_wallet(self, wallet: str, max_per_chain: int = 50) -> pd.DataFrame:
        all_tx = []
        for chain_name, chain_id in self.chains.items():
            print(f"Fetching ERC20 transfers on {chain_name}...")
            erc20_txs = self.get_erc20_transfers(wallet, chain=chain_id, limit=max_per_chain)
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

                    # ✅ gas cost (native coin)
                    tx_hash = tx.get("transaction_hash")
                    gas_native = self.get_tx_gas_cost(tx_hash, chain_id) or 0

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
                        "transaction_type": "deposit" if tx.get("to_address", "").lower() == wallet.lower() else "withdrawal"
                    }
                    all_tx.append(enriched)
                except Exception as e:
                    print(f"Error processing tx: {e}")
                    continue

            time.sleep(0.5)  # rate limit

        if not all_tx:
            return pd.DataFrame()

        df = pd.DataFrame(all_tx)
        df['block_time'] = pd.to_datetime(df['block_time'])
        return df.sort_values("block_time").reset_index(drop=True)

# -------------------------------
# PnL Calculation Helper (FIFO, LIFO, ACB)
# -------------------------------
def calculate_pnl(df, method="FIFO"):
    positions = {}  # token -> list of lots (for FIFO/LIFO)
    avg_costs = {}  # token -> (total_qty, avg_cost_per_unit) for ACB
    realized_pnl = 0
    token_realized = {}
    token_unrealized = {}

    for _, row in df.sort_values("block_time").iterrows():
        token = row["token_symbol"]
        qty = row.get("amount", row.get("token_amount", 0))
        usd_value = row["usd_value"]

        if qty == 0:
            continue

        if method in ["FIFO", "LIFO"]:
            # Lot-based methods
            if row["transaction_type"] in ["deposit", "buy", "swap_in"]:
                cost_per_unit = usd_value / qty
                lot = {"qty": qty, "cost": cost_per_unit}
                positions.setdefault(token, []).append(lot)

            elif row["transaction_type"] in ["withdrawal", "sell", "swap_out"]:
                remaining = qty
                sell_per_unit = usd_value / qty
                while remaining > 0 and token in positions and positions[token]:
                    lot = positions[token][0] if method == "FIFO" else positions[token][-1]
                    lot_qty = lot["qty"]
                    consumed = min(remaining, lot_qty)

                    pnl_piece = consumed * (sell_per_unit - lot["cost"])
                    realized_pnl += pnl_piece
                    token_realized[token] = token_realized.get(token, 0) + pnl_piece

                    lot["qty"] -= consumed
                    remaining -= consumed
                    if lot["qty"] == 0:
                        if method == "FIFO":
                            positions[token].pop(0)
                        else:
                            positions[token].pop(-1)

        elif method == "ACB":
            # Average Cost Basis method
            total_qty, avg_cost = avg_costs.get(token, (0, 0))

            if row["transaction_type"] in ["deposit", "buy", "swap_in"]:
                # New weighted average
                new_total_qty = total_qty + qty
                new_total_cost = (total_qty * avg_cost) + usd_value
                new_avg_cost = new_total_cost / new_total_qty if new_total_qty > 0 else 0
                avg_costs[token] = (new_total_qty, new_avg_cost)

            elif row["transaction_type"] in ["withdrawal", "sell", "swap_out"]:
                if total_qty > 0:
                    sell_per_unit = usd_value / qty
                    pnl_piece = qty * (sell_per_unit - avg_cost)
                    realized_pnl += pnl_piece
                    token_realized[token] = token_realized.get(token, 0) + pnl_piece

                    # Reduce holdings
                    new_total_qty = total_qty - qty
                    avg_costs[token] = (max(new_total_qty, 0), avg_cost)

    # -------------------------------
    # Unrealized PnL
    # -------------------------------
    unrealized_pnl = 0
    token_data = []

    if method in ["FIFO", "LIFO"]:
        for token, lots in positions.items():
            current_price = df[df["token_symbol"] == token]["price_usd"].iloc[-1] if "price_usd" in df.columns else 0
            for lot in lots:
                pnl_piece = lot["qty"] * (current_price - lot["cost"])
                unrealized_pnl += pnl_piece
                token_unrealized[token] = token_unrealized.get(token, 0) + pnl_piece

    elif method == "ACB":
        for token, (total_qty, avg_cost) in avg_costs.items():
            current_price = df[df["token_symbol"] == token]["price_usd"].iloc[-1] if "price_usd" in df.columns else 0
            pnl_piece = total_qty * (current_price - avg_cost)
            unrealized_pnl += pnl_piece
            token_unrealized[token] = token_unrealized.get(token, 0) + pnl_piece

    # -------------------------------
    # Build breakdown DataFrame
    # -------------------------------
    all_tokens = set(token_realized.keys()).union(set(token_unrealized.keys()))
    for token in all_tokens:
        token_data.append({
            "Token": token,
            "Realized PnL (USD)": token_realized.get(token, 0),
            "Unrealized PnL (USD)": token_unrealized.get(token, 0),
        })

    breakdown_df = pd.DataFrame(token_data).sort_values("Realized PnL (USD)", ascending=False)

    return realized_pnl, unrealized_pnl, breakdown_df
    

# Load env
load_dotenv()
API_KEY = os.getenv("MORALIS_API_KEY")

# Option 1: Use cache always (default)
analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=False)

# Option 2: Force fresh API calls (ignore cache)
analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=True, force_refresh=True)

# Option 3: Disable cache entirely
analyzer = ExtendedMoralisAnalyzer(API_KEY, use_cache=False)
