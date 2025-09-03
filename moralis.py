import asyncio
import aiohttp
from moralis import evm_api
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import time

# Initialize Moralis
api_key = "MORALIS_API_KEY"

class WalletAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chains = ['eth', 'bsc', 'arbitrum', 'optimism', 'base']
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def find_candidate_wallets(self) -> List[str]:
        """
        Step 1: Get candidate wallets from recent DEX activity
        Much faster than scanning all historical data
        """
        candidates = set()
        
        for chain in self.chains:
            try:
                # Get recent transactions from popular DEX contracts
                # You can get these from your existing database more efficiently
                result = evm_api.transaction.get_wallet_transactions(
                    api_key=self.api_key,
                    params={
                        "chain": chain,
                        "from_date": (datetime.now() - timedelta(days=7)).isoformat(),
                        "limit": 100,
                        # Add known DEX contract addresses to filter
                        "contract_addresses": [
                            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2
                            "0xE592427A0AEce92De3Edee1F18E0157C05861564",  # Uniswap V3
                            # Add more DEX contracts
                        ]
                    }
                )
                
                for tx in result.get('result', []):
                    candidates.add(tx['from_address'])
                    if tx['to_address']:
                        candidates.add(tx['to_address'])
                        
            except Exception as e:
                print(f"Error fetching candidates from {chain}: {e}")
                
        return list(candidates)[:100]  # Limit candidates

    async def validate_wallet_activity(self, wallet: str) -> Optional[Dict]:
        """
        Step 2: Check if wallet meets our criteria
        """
        try:
            total_swaps = 0
            total_transfers = 0
            active_chains = 0
            
            for chain in self.chains:
                try:
                    # Get wallet transactions for this chain
                    result = evm_api.transaction.get_wallet_transactions(
                        api_key=self.api_key,
                        params={
                            "chain": chain,
                            "address": wallet,
                            "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
                            "limit": 200
                        }
                    )
                    
                    transactions = result.get('result', [])
                    if not transactions:
                        continue
                        
                    active_chains += 1
                    
                    # Count swaps and transfers
                    chain_swaps = 0
                    chain_transfers = 0
                    
                    for tx in transactions:
                        if self.is_dex_swap(tx):
                            chain_swaps += 1
                        elif self.is_token_transfer(tx):
                            chain_transfers += 1
                    
                    total_swaps += chain_swaps
                    total_transfers += chain_transfers
                    
                    # Rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    print(f"Error checking {wallet} on {chain}: {e}")
                    continue
            
            # Check if wallet meets criteria
            if total_swaps >= 4 and total_transfers >= 4 and active_chains >= 3:
                return {
                    'wallet': wallet,
                    'total_swaps': total_swaps,
                    'total_transfers': total_transfers,
                    'active_chains': active_chains
                }
                
        except Exception as e:
            print(f"Error validating wallet {wallet}: {e}")
            
        return None

    async def get_detailed_wallet_data(self, qualified_wallets: List[Dict]) -> pd.DataFrame:
        """
        Step 3: Get detailed transaction data for qualified wallets
        """
        all_transactions = []
        
        for wallet_info in qualified_wallets[:20]:  # Limit to top 20
            wallet = wallet_info['wallet']
            
            for chain in self.chains:
                try:
                    # Get comprehensive transaction history
                    result = evm_api.transaction.get_wallet_transactions(
                        api_key=self.api_key,
                        params={
                            "chain": chain,
                            "address": wallet,
                            "from_date": (datetime.now() - timedelta(days=90)).isoformat(),
                            "limit": 500
                        }
                    )
                    
                    transactions = result.get('result', [])
                    
                    for tx in transactions:
                        processed_tx = self.process_transaction(tx, wallet, chain)
                        if processed_tx:
                            all_transactions.append(processed_tx)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error getting detailed data for {wallet} on {chain}: {e}")
        
        # Convert to DataFrame and sort by time
        df = pd.DataFrame(all_transactions)
        if not df.empty:
            df['block_time'] = pd.to_datetime(df['block_time'])
            df = df.sort_values('block_time')
            
        return df

    def is_dex_swap(self, transaction: Dict) -> bool:
        """Check if transaction is a DEX swap"""
        # Look for common DEX method signatures
        swap_signatures = [
            '0x38ed1739',  # swapExactTokensForTokens
            '0x7ff36ab5',  # swapExactETHForTokens
            '0x18cbafe5',  # swapExactTokensForETH
            '0x414bf389',  # swapExactTokensForTokensSupportingFeeOnTransferTokens
        ]
        
        input_data = transaction.get('input', '')
        return any(input_data.startswith(sig) for sig in swap_signatures)

    def is_token_transfer(self, transaction: Dict) -> bool:
        """Check if transaction is a token transfer"""
        # Look for ERC20 transfer signature
        return transaction.get('input', '').startswith('0xa9059cbb')

    def process_transaction(self, tx: Dict, wallet: str, chain: str) -> Optional[Dict]:
        """Process transaction into standardized format"""
        try:
            # Determine action type
            if self.is_dex_swap(tx):
                action = 'swap'
            elif tx['to_address'].lower() == wallet.lower():
                action = 'deposit'
            elif tx['from_address'].lower() == wallet.lower():
                action = 'withdrawal'
            else:
                return None
                
            return {
                'tx_hash': tx['hash'],
                'block_time': tx['block_timestamp'],
                'wallet': wallet,
                'blockchain': chain,
                'action': action,
                'value_usd': self.calculate_usd_value(tx),
                'gas_usd': self.calculate_gas_cost(tx),
                # Add more fields as needed
            }
        except Exception as e:
            print(f"Error processing transaction {tx.get('hash', 'unknown')}: {e}")
            return None

    def calculate_usd_value(self, tx: Dict) -> Optional[float]:
        """Calculate USD value of transaction"""
        # This would need price data - you could use Moralis price API
        # or integrate with your existing price data
        return None

    def calculate_gas_cost(self, tx: Dict) -> Optional[float]:
        """Calculate gas cost in USD"""
        try:
            gas_used = int(tx.get('gas_used', 0))
            gas_price = int(tx.get('gas_price', 0))
            gas_cost_wei = gas_used * gas_price
            gas_cost_eth = gas_cost_wei / 1e18
            # You'd need ETH price to convert to USD
            return gas_cost_eth  # Return ETH for now
        except:
            return None

async def main():
    """Main execution function"""
    analyzer = WalletAnalyzer(api_key)
    
    try:
        print("Step 1: Finding candidate wallets...")
        candidates = analyzer.find_candidate_wallets()
        print(f"Found {len(candidates)} candidate wallets")
        
        print("Step 2: Validating wallet activity...")
        qualified_wallets = []
        
        # Process candidates in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            tasks = [analyzer.validate_wallet_activity(wallet) for wallet in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    qualified_wallets.append(result)
            
            print(f"Processed batch {i//batch_size + 1}, qualified: {len(qualified_wallets)}")
            
            # Rate limiting between batches
            if i + batch_size < len(candidates):
                await asyncio.sleep(2)
        
        print(f"Step 3: Getting detailed data for {len(qualified_wallets)} wallets...")
        detailed_data = await analyzer.get_detailed_wallet_data(qualified_wallets)
        
        print(f"Final dataset: {len(detailed_data)} transactions")
        return detailed_data
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return pd.DataFrame()

# Alternative: Quick validation using existing database + Moralis for details
def hybrid_approach():
    """
    Use your existing database to find candidates quickly,
    then use Moralis for detailed validation and enrichment
    """
    
    # Step 1: Use your fast database query to get candidate wallets
    candidates_query = """
    SELECT tx_from AS wallet, COUNT(*) as activity_count
    FROM dex.trades 
    WHERE blockchain IN ('ethereum','bnb','arbitrum','optimism','base')
      AND block_time >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY tx_from
    HAVING COUNT(*) >= 10  -- Pre-filter for very active wallets
    ORDER BY COUNT(*) DESC
    LIMIT 50;
    """
    
    # Step 2: Use Moralis API to validate and get detailed data
    # This combines the speed of your database with Moralis enrichment

# Run the analysis
if __name__ == "__main__":
    result_df = asyncio.run(main())
    
    if not result_df.empty:
        # Save results
        result_df.to_csv('wallet_analysis_results.csv', index=False)
        print("Results saved to wallet_analysis_results.csv")
    else:
        print("No results obtained")