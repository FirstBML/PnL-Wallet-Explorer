import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

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
    def __init__(self, api_key: str):
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
    # PRICE FETCHER (Coingecko)
    # -------------------------------
    def get_price_usd(self, symbol: str, timestamp: str, token_address: str = None, blockchain: str = "ethereum") -> Optional[float]:
        if not symbol and not token_address:
            return None

        symbol = (symbol or "").lower()
        date_str = timestamp.split("T")[0]
        cache_key = f"{symbol}_{token_address}_{date_str}"
        cached = self.price_cache.get(cache_key)
        if cached:
            return cached

        try:
            # Step 1: symbol mapping
            mapping = {
                "eth": "ethereum",
                "weth": "weth",
                "usdc": "usd-coin",
                "usdt": "tether",
                "bnb": "binancecoin",
                "matic": "polygon"
            }
            cg_id = mapping.get(symbol)

            # Step 2: check address cache
            if not cg_id and token_address:
                cached_cgid = self.address_cache.get(token_address.lower())
                if cached_cgid:
                    cg_id = cached_cgid

            # Step 3: contract lookup
            if not cg_id and token_address:
                try:
                    url = f"https://api.coingecko.com/api/v3/coins/{blockchain}/contract/{token_address}"
                    r = requests.get(url)
                    if r.status_code == 200:
                        data = r.json()
                        cg_id = data.get("id")
                        if cg_id:
                            self.address_cache.set(token_address.lower(), cg_id)
                except Exception as e:
                    print(f"Contract lookup failed for {token_address}: {e}")

            if not cg_id:
                return None

            # Step 4: fetch historical price
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
    # Enrich transfers with USD price
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

                    enriched = {
                        "wallet": wallet,
                        "blockchain": chain_name,
                        "tx_hash": tx.get("transaction_hash"),
                        "block_time": timestamp,
                        "token_symbol": symbol,
                        "token_address": token_address,
                        "amount": amount,
                        "price_usd": price_usd,
                        "usd_value": usd_value,
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
# Runner
# -------------------------------
if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("MORALIS_API_KEY")

    if not API_KEY:
        raise ValueError("⚠️ Please add MORALIS_API_KEY to your .env file!")

    wallet_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # replace with user input
    analyzer = ExtendedMoralisAnalyzer(API_KEY)

    df = analyzer.get_detailed_data_for_wallet(wallet_address, max_per_chain=30)

    if df.empty:
        print("No transactions found.")
    else:
        print(df.head(20))
        print("\nSummary USD values:")
        print(df.groupby("transaction_type")["usd_value"].sum())
