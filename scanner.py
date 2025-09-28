import datetime
import time
import csv
from typing import List, Tuple, Dict, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

# New imports for market data
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- Configuration Constants ---
# File path where API keys are stored in KEY=VALUE format
KEY_FILE_PATH = "/var/workspace/keys/alpaca_keys.txt"

# Define the exchanges we want to scan
TARGET_EXCHANGES = ["NASDAQ", "NYSE"]

# Momentum Filter Constants
MOMENTUM_LOOKBACK_DAYS = 7  # Look back 7 daily bars for momentum calculation
MIN_MOMENTUM_GAIN_PERCENT = 3.0  # Minimum required absolute gain (e.g., 5.0 means 5% gain)

# Volume and Price Filter Constants (Applied to the latest bar)
MIN_DAILY_VOLUME = 10000000.0  # Minimum required daily trading volume
MIN_CLOSE_PRICE = 30.0       # Minimum required closing price

# API Batching
MAX_SYMBOLS_PER_REQUEST = 1000  # Keep batch size reasonable to avoid long URLs/timeouts

# Output Configuration
OUTPUT_FILENAME = "strong_momentum_stocks.csv"


# -------------------------------

def load_api_keys(file_path: str) -> Tuple[str, str]:
    """
    Reads API key and secret from a specified file path.
    It expects keys in the format: PAPER_API_KEY=... and PAPER_API_SECRET=...
    """
    keys = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines or comments
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    keys[key.strip()] = value.strip()

        # Assuming we are using PAPER_TRADING credentials for this scanner
        api_key = keys.get('PAPER_API_KEY')
        secret_key = keys.get('PAPER_API_SECRET')

        if not api_key or not secret_key:
            raise ValueError(
                "Could not find 'PAPER_API_KEY' or 'PAPER_API_SECRET' in the key file. "
                "Please ensure the file is correctly formatted."
            )

        return api_key, secret_key

    except FileNotFoundError:
        print(f"ERROR: Key file not found at: {file_path}")
        return "", ""
    except Exception as e:
        print(f"ERROR: Failed to load API keys from file: {e}")
        return "", ""


def get_active_assets_for_exchanges(api_key: str, secret_key: str) -> List[str]:
    """
    Fetches all active US equity assets from Alpaca, filters by exchange, and returns a list of symbols.

    Returns:
        list[str]: A list of ticker symbols (strings).
    """
    if not api_key or not secret_key:
        print("API keys are missing. Cannot connect to Alpaca.")
        return []


    try:
        # We need the TradingClient to access the asset list endpoint
        client = TradingClient(api_key, secret_key)

        search_params = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE
        )

        print("Fetching all active US equities...")
        all_assets = client.get_all_assets(search_params)

        # Filter by exchange and extract only the symbol
        filtered_symbols = [
            asset.symbol for asset in all_assets
            if asset.exchange in TARGET_EXCHANGES
        ]

        print(f"Found {len(filtered_symbols)} active symbols on {', '.join(TARGET_EXCHANGES)}.")
        return filtered_symbols

    except Exception as e:
        print(f"An error occurred during asset retrieval: {e}")
        return []


def apply_stock_filters(symbols: List[str], api_key: str, secret_key: str) -> List[Dict[str, Any]]:
    """
    Fetches historical daily bar data and applies filters for price, volume, and momentum.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the symbol,
        momentum value, latest close price, and latest volume for qualifying stocks.
    """
    if not symbols:
        print("No symbols provided for filtering.")
        return []

    # Initialize the Data Client
    data_client = StockHistoricalDataClient(api_key, secret_key)

    # Set the start date to ensure we get enough bars for the momentum calculation
    # We request data going back 2x the lookback period for safety.
    start_date = datetime.date.today() - datetime.timedelta(days=MOMENTUM_LOOKBACK_DAYS * 2)

    print(f"\nApplying multi-stage filters:")
    print(f"  - Latest Close Price >= ${MIN_CLOSE_PRICE:.2f}")
    print(f"  - Latest Daily Volume >= {MIN_DAILY_VOLUME:,}")
    print(f"  - Absolute Momentum ({MOMENTUM_LOOKBACK_DAYS}D) >= {MIN_MOMENTUM_GAIN_PERCENT}%")

    # Change from List[str] to List[Dict[str, Any]]
    filtered_stocks_data = []
    total_symbols = len(symbols)

    # Process symbols in batches to handle API limits and timeouts
    for i in range(0, total_symbols, MAX_SYMBOLS_PER_REQUEST):
        batch_symbols = symbols[i:i + MAX_SYMBOLS_PER_REQUEST]
        # Calculate the batch number for cleaner printing
        batch_num = int(i / MAX_SYMBOLS_PER_REQUEST) + 1
        total_batches = int(total_symbols / MAX_SYMBOLS_PER_REQUEST) + 1
        print(
            f"  Processing batch {batch_num} of {total_batches} ({len(batch_symbols)} symbols)...")

        try:
            # 1. Prepare the historical data request
            request_params = StockBarsRequest(
                symbol_or_symbols=batch_symbols,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=datetime.date.today()
            )

            # 2. Get the bar data
            bar_set = data_client.get_stock_bars(request_params)
            # Convert to DataFrame and unstack to access data easily by symbol
            bar_set_df = bar_set.df.unstack(level=0)

            # 3. Iterate through results and apply filters
            for symbol in batch_symbols:

                try:
                    # Access the series of close prices and volume for the current symbol
                    close_prices = bar_set_df['close'][symbol]
                    volume_data = bar_set_df['volume'][symbol]

                    # Drop any NaN values (e.g., for non-trading days or missing data)
                    valid_prices = close_prices.dropna()
                    valid_volumes = volume_data.dropna()
                    # Ensure we have at least one data point to check price/volume
                    if not (len(valid_prices) > 0 and len(valid_volumes) > 0):
                        continue  # Skip, no recent data

                    latest_close = valid_prices.iloc[-1]
                    latest_volume = valid_volumes.iloc[-1]

                    # --- Filter 1: Price Check ---
                    if latest_close < MIN_CLOSE_PRICE:
                        continue

                    # --- Filter 2: Volume Check ---
                    if latest_volume < MIN_DAILY_VOLUME:
                        continue

                    # --- Filter 3: Momentum Check (Requires sufficient history) ---
                    if len(valid_prices) >= MOMENTUM_LOOKBACK_DAYS:
                        oldest_close = valid_prices.iloc[0]

                        # Calculate percentage change: (Latest - Oldest) / Oldest * 100
                        if oldest_close > 0:
                            momentum = ((latest_close - oldest_close) / oldest_close) * 100

                            # Check if the absolute momentum is greater than or equal to the threshold
                            if abs(momentum) >= MIN_MOMENTUM_GAIN_PERCENT:
                                # Add structured data to the list
                                filtered_stocks_data.append({
                                    'symbol': symbol,
                                    'momentum': round(momentum, 2),
                                    'yesterday_price': round(latest_close, 2),
                                    'yesterday_volume': int(latest_volume)
                                })

                except KeyError:
                    # This occurs if the symbol is in the asset list but has no bar data
                    continue
                except Exception as e:
                    # Catch any other specific errors during processing
                    print(f"    WARNING: Failed to process filters for symbol {symbol}: {e}")
                    continue

        except Exception as e:
            # Continue scanning even if one batch fails (e.g., network error, API throttling)
            print(f"    WARNING: Failed to process batch {batch_num} due to error: {e}")

        # Simple rate-limit delay to avoid hitting the 200 req/min limit
        time.sleep(1)

    return filtered_stocks_data


def output_to_csv(data: List[Dict[str, Any]], filename: str):
    """Writes the list of structured stock data to a CSV file."""
    if not data:
        print("No results to write to CSV.")
        return

    # Define the headers based on the requested output
    fieldnames = ['symbol', 'momentum', 'yesterday_price', 'yesterday_volume']

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerows(data)

        print(f"\nSuccessfully wrote {len(data)} results to {filename}")
    except Exception as e:
        print(f"ERROR: Failed to write to CSV file: {e}")


def print_asset_summary(assets: List[Dict[str, Any]], initial_count: int):
    """
    Prints a summary of the filtered symbols.
    """
    final_count = len(assets)

    print("\n--- Scanning Results ---")
    print(f"Initial symbols checked (NASDAQ/NYSE): {initial_count}")
    print(f"Final symbols after momentum filter: {final_count}")

    if final_count > 0:
        print(f"\nStrong Momentum Stocks (Top {min(final_count, 20)}):")
        # Print a sample of the results
        for symbol in assets[:20]:
            print(f"  - {symbol}")
    else:
        print("\nNo stocks met the strong momentum criteria.")

    print("\n--- Scan Complete ---")


def main():
    """
    Main execution function for the stock scanner.
    """
    # 1. Load API Keys
    api_key, secret_key = load_api_keys(KEY_FILE_PATH)

    # Check if keys were successfully loaded
    if not api_key:
        return

    # 2. Get the initial list of active stock symbols
    initial_symbols = get_active_assets_for_exchanges(api_key, secret_key)

    if not initial_symbols:
        return

    # 3. Apply all filters using historical data
    # The result is now a list of dictionaries, not just symbols
    strong_momentum_data = apply_stock_filters(initial_symbols, api_key, secret_key)

    # 4. Print the final summary
    print_asset_summary(strong_momentum_data, len(initial_symbols))

    # 5. Output to CSV file
    output_to_csv(strong_momentum_data, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
