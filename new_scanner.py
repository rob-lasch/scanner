import os
import sys
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo  # Standard library in Python 3.9+

# Import Alpaca API libraries
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass, AssetExchange


# --- Configuration (Matching Go Constants) ---
NY_TIMEZONE = ZoneInfo("America/New_York")

# Scanner Criteria
PRICE_THRESHOLD = 30.0
DAILY_VOLUME_THRESHOLD = 10 * 1000 * 1000
PREMARKET_VOLUME_THRESHOLD = 1* 1000* 1000
RELATIVE_VOLUME_THRESHOLD = 1.0
GAP_THRESHOLD = 1.0
HISTORICAL_DAYS = 14
MAX_WORKERS = 6

# Default file path for API keys
ALPACA_KEY_FILE = "/var/workspace/keys/alpaca_keys.txt"

# --- MOCKING CONFIGURATION ---
# Set to True to override the current time for testing pre-market logic when the market is not open
MOCKING_ENABLED = False
# Setting the mock date to Friday, September 26, 2025, 9:00 AM ET (Pre-market)
MOCK_DATETIME_STR = "2025-09-26 09:20:00"


@dataclass
class StockData:
    """Holds all the information for a potential trade."""
    symbol: str
    price: float
    gap_pct: float
    rel_vol: float
    premarket_volume: float


def get_current_datetime() -> datetime:
    """
    Returns a mocked datetime (9:00 AM ET on 2025-09-26) if MOCKING_ENABLED is True,
    otherwise returns the actual current datetime, localized to NY_TIMEZONE.
    """
    if MOCKING_ENABLED:
        try:
            # Parse the mock string and localize it to NY_TIMEZONE
            dt = datetime.strptime(MOCK_DATETIME_STR, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=NY_TIMEZONE)
        except ValueError:
            print(f"Error parsing MOCK_DATETIME_STR: {MOCK_DATETIME_STR}. Falling back to live time.")
            return datetime.now(NY_TIMEZONE)
    else:
        return datetime.now(NY_TIMEZONE)


def read_keys_from_file(file_path: str) -> Dict[str, str]:
    """
    Reads key-value pairs from a file, mirroring the Go implementation's logic.
    Lines should be in 'KEY = VALUE' format.
    """
    keys = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    # Trim quotes/spaces from the values if they exist in the file.
                    keys[key] = value.strip('"').strip("'").strip()
        return keys
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error reading key file {file_path}: {e}")
        return {}


def load_api_keys() -> Dict[str, str]:
    """
    1. Loads keys from the default file path, falling back to
    environment variables if the file read fails or keys are missing.
    """
    print(f"Attempting to read keys from file: {ALPACA_KEY_FILE}")
    file_keys = read_keys_from_file(ALPACA_KEY_FILE)

    paper_key = file_keys.get("PAPER_API_KEY")
    paper_secret = file_keys.get("PAPER_API_SECRET")

    if paper_key and paper_secret:
        print("Successfully loaded keys from file.")
        return {"PAPER_API_KEY": paper_key, "PAPER_API_SECRET": paper_secret}

    print("File reading failed or keys missing. Falling back to environment variables...")

    # Fall back to environment variables
    paper_api_key = os.environ.get("PAPER_API_KEY")
    paper_api_secret = os.environ.get("PAPER_API_SECRET")

    if not paper_api_key or not paper_api_secret:
        print("ERROR: PAPER_API_KEY and PAPER_API_SECRET must be set in either the key file or environment variables.")
        sys.exit(1)

    return {
        "PAPER_API_KEY": paper_api_key,
        "PAPER_API_SECRET": paper_api_secret
    }


def get_tradable_symbols(trading_client: TradingClient) -> List[str]:
    """
    3. Use client to get list of all active, tradable US equity stocks
       listed on NYSE or NASDAQ.
    """
    # Define the exchanges to scan
    exchanges_to_scan = [AssetExchange.NYSE, AssetExchange.NASDAQ]
    all_symbols = set()

    print(f"Fetching all active tradable US equity assets from: {[e.value for e in exchanges_to_scan]}...")

    try:
        for exchange in exchanges_to_scan:
            assets = trading_client.get_all_assets(GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
                exchange=exchange
            ))
            # Use a set to collect unique symbols across both exchanges
            exchange_symbols = {asset.symbol for asset in assets}
            all_symbols.update(exchange_symbols)
            print(f"  -> Found {len(exchange_symbols)} active symbols on {exchange.value}.")

        final_list = sorted(list(all_symbols))
        print(f"Total unique tradable US equities found: {len(final_list)}")
        return final_list

    except Exception as e:
        print(f"ERROR: Failed to get assets from Trading API. Check keys/network: {e}")
        return []


def pre_filter_stocks(data_client: StockHistoricalDataClient, all_symbols: List[str]) -> Dict[str, float]:
    """
    4. Get yesterday's one-day candle and filter on volume and price.

    Returns a dictionary mapping symbol to its previous day's closing price
    for stocks that meet the price and volume thresholds.
    """
    # Use the mocked or real current time to determine the date range
    current_dt = get_current_datetime()

    # Yesterday's date for bar start (e.g., if current is Friday, start is Wednesday) 5 because what is its friday and monday is a holiday?
    yesterday = current_dt.date() - timedelta(days=5)
    # Current date for bar end (e.g., if current is Friday, end is Friday)
    today = current_dt.date()

    print(f"Fetching daily bars for range {yesterday} to {today}...")

    yesterday_request = StockBarsRequest(
        symbol_or_symbols=all_symbols,
        timeframe=TimeFrame.Day,
        start=yesterday,
        end=today
    )

    pre_filtered_stocks: Dict[str, float] = {}
    print(
        f"Starting pre-filter based on Yesterday's Close > ${PRICE_THRESHOLD} and Daily Volume > {DAILY_VOLUME_THRESHOLD:,.0f}...")

    try:
        yesterday_bars_response = data_client.get_stock_bars(yesterday_request)
        if yesterday_bars_response.df is not None:
            # Group by symbol and apply filter
            for symbol, df in yesterday_bars_response.df.groupby(level=0):
                if not df.empty:
                    yesterday_bar = df.iloc[-1]
                    # Filter logic: Price and Volume check
                    if (
                            yesterday_bar['close'] > PRICE_THRESHOLD and
                            yesterday_bar['volume'] > DAILY_VOLUME_THRESHOLD
                    ):
                        pre_filtered_stocks[symbol] = yesterday_bar['close']
    except Exception as e:
        print(f"Error during bulk daily bar volume check: {e}")
        return {}  # Return empty on fatal error

    print(f"Pre-filtered list contains {len(pre_filtered_stocks)} stocks.")
    return pre_filtered_stocks


def get_historical_premarket_volume(
        data_client: StockHistoricalDataClient,
        symbol: str
) -> Optional[float]:
    """
    Fetches and averages pre-market volume (4:00 AM ET to 9:00 AM ET)
    for a symbol over HISTORICAL_DAYS (14 trading days).
    """
    total_volumes: List[float] = []
    current_dt = get_current_datetime()

    # Iterate backwards through days from the current mock/real date
    for i in range(1, HISTORICAL_DAYS + 1):
        date = current_dt.date() - timedelta(days=i)

        # Define pre-market window for that day
        pre_market_start = datetime.combine(date, datetime.min.time(), tzinfo=NY_TIMEZONE).replace(hour=4, minute=0)
        pre_market_end = datetime.combine(date, datetime.min.time(), tzinfo=NY_TIMEZONE).replace(hour=9, minute=20)

        # Skip weekends (simplified check, real life needs market holidays too)
        if pre_market_start.weekday() >= 5:  # Saturday or Sunday
            continue

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=pre_market_start,
            end=pre_market_end
        )

        try:
            bars_df = data_client.get_stock_bars(request).df
            if not bars_df.empty:
                # Sum the volume column
                daily_vol = bars_df['volume'].sum()
                if daily_vol > 0:
                    total_volumes.append(daily_vol)
        except Exception:
            # Silently skip if data is unavailable for a day
            continue

    if not total_volumes:
        return None

    # Return the average volume
    return sum(total_volumes) / len(total_volumes)


def get_premarket_data(
        data_client: StockHistoricalDataClient,
        symbol: str
) -> Optional[Tuple[float, float]]:
    """
    Fetches today's pre-market volume (4:00 AM ET to now) and the latest price.

    Returns: (pre_market_volume, latest_price)
    """
    now = get_current_datetime()
    pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)

    # If it's before 4 AM ET, no pre-market data is available
    if now < pre_market_start:
        return None

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=pre_market_start,
        end=now
    )

    try:
        bars_df = data_client.get_stock_bars(request).df
    except Exception as e:
        print(f"Failed to get today's pre-market bars for {symbol}: {e}")
        return None

    if bars_df is None or bars_df.empty:
        return None

    pre_market_vol = bars_df['volume'].sum()
    latest_price = bars_df['close'].iloc[-1]

    return pre_market_vol, latest_price

def process_stock(
        symbol: str,
        yesterday_close: float,
        data_client: StockHistoricalDataClient
    ) -> Optional[StockData]:
    """
    Worker function to run the full pre-market analysis for a single stock.
    Returns StockData if all criteria are met, otherwise None.
    """
    try:
        # Get today's pre-market volume and latest price
        pre_market_result = get_premarket_data(data_client, symbol)
        if pre_market_result is None:
            return None
        pre_market_vol, latest_price = pre_market_result  # <--- pre_market_vol is already here

        if pre_market_vol < PREMARKET_VOLUME_THRESHOLD:
            return None
        # Get historical pre-market average volume
        avg_historical_vol = get_historical_premarket_volume(data_client, symbol)
        if avg_historical_vol is None or avg_historical_vol == 0:
            return None

        # Calculate metrics
        # Gap Percentage: (Current Price - Yesterday's Close) / Yesterday's Close * 100
        gap_pct = ((latest_price - yesterday_close) / yesterday_close) * 100

        # Relative Volume (RVOL): Today's Pre-Market Vol / Historical Avg Pre-Market Vol
        rel_vol = pre_market_vol / avg_historical_vol

        # Apply final filter criteria (GAP and RVOL)
        is_gapping = abs(gap_pct) > GAP_THRESHOLD
        is_high_rvol = rel_vol > RELATIVE_VOLUME_THRESHOLD

        if is_gapping and is_high_rvol:
            return StockData(
                symbol=symbol,
                price=latest_price,
                gap_pct=gap_pct,
                rel_vol=rel_vol,
                premarket_volume=pre_market_vol  # <--- ADD THIS LINE
            )

    except Exception as e:
        # print(f"Error processing {symbol}: {e}")
        pass  # Fail gracefully in the worker thread

    return None


def run_premarket_scan(data_client: StockHistoricalDataClient, pre_filtered_stocks: Dict[str, float]) -> List[StockData]:
    """
    5. Get 14 trading days of premarket data, filter on gap and rvol into dataframe (concurrently).
    """
    final_watchlist: List[StockData] = []

    # Use ThreadPoolExecutor for concurrent fetching (highly recommended for performance)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks for each pre-filtered symbol
        futures = {
            executor.submit(process_stock, symbol, close_price, data_client): symbol
            for symbol, close_price in pre_filtered_stocks.items()
        }

        # Process results as they complete
        print("Starting concurrent pre-market analysis (Gap and RVOL checks)...")
        count = 0
        total = len(futures)

        for future in as_completed(futures):
            count += 1
            result = future.result()
            if result:
                final_watchlist.append(result)

            # Print progress every 100 symbols
            if count % 100 == 0 or count == total:
                print(f"  -> Progress: {count}/{total} symbols checked. {len(final_watchlist)} candidates found.")

    return final_watchlist


def output_watchlist(final_watchlist: List[StockData]):
    """
    6. Sort by the absolute gap percentage and output the top 15 stocks to a CSV file.
    """
    if not final_watchlist:
        print("No stocks met the aggressive scanner criteria. No CSV file created.")
        return

    # Sort the final list by the absolute value of the gap percentage in descending order.
    # The sorted() function is used with a lambda key to sort a list of dictionaries by a specific value.
    # The `reverse=True` parameter ensures a descending sort.
    sorted_by_gap = sorted(final_watchlist, key=lambda x: abs(x.gap_pct), reverse=True)

    # Take the top 15 stocks from the sorted list using slicing.
    top_15_stocks = sorted_by_gap[:15]

    csv_filename = "watchlist.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            "Symbol",
            "Latest_Pre-Market_Price",
            "Gap_from_Yesterday's_Close_Pct",
            "Relative_Volume",
            "Premarket_Volume"
        ])

        # Write data rows for the top 15 stocks
        for stock in top_15_stocks:
            writer.writerow([
                stock.symbol,
                f"${stock.price:,.2f}",
                f"{stock.gap_pct:+.2f}%",
                f"{stock.rel_vol:.2f}x",
                f"{stock.premarket_volume:,.0f}"
            ])

    print(f"Successfully wrote the top {len(top_15_stocks)} stocks by gap to {csv_filename}")
    if len(final_watchlist) > 15:
        print(
            f"Note: There were {len(final_watchlist)} stocks that passed the initial filters, but only the top 15 were saved.")


def main():
    """
    The main execution function, structured based on the user's pseudo-code
    and refactored into modular methods.
    """
    print("--- Starting Alpaca Pre-Market Stock Scanner ---")

    # Display the current mode
    if MOCKING_ENABLED:
        mock_dt = get_current_datetime().strftime("%A, %B %d, %Y at %I:%M %p ET")
        print(f"*** WARNING: Running in MOCK MODE for time: {mock_dt} ***")

    # 1. Load API keys
    keys = load_api_keys()

    # 2. Use API keys to initialize Alpaca clients
    trading_client = TradingClient(keys["PAPER_API_KEY"], keys["PAPER_API_SECRET"], paper=True)
    data_client = StockHistoricalDataClient(keys["PAPER_API_KEY"], keys["PAPER_API_SECRET"])
    print("Alpaca clients initialized.")

    # 3. Get list of all stocks
    all_symbols = get_tradable_symbols(trading_client)
    if not all_symbols:
        return

    # 4. Get yesterday's close/volume and filter
    pre_filtered_stocks = pre_filter_stocks(data_client, all_symbols)
    if not pre_filtered_stocks:
        print("No stocks passed the initial price and volume filter.")
        return

    # 5. Run final pre-market analysis (Gap, RVOL)
    final_watchlist = run_premarket_scan(data_client, pre_filtered_stocks)

    # 6. Sort and Output final watchlist
    output_watchlist(final_watchlist)


if __name__ == "__main__":
    main()