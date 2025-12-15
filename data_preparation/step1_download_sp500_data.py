"""
Step 1: Download S&P 500 stock data from Yahoo Finance

Downloads historical price data for all S&P 500 stocks.
Saves to: data/sp500_history.csv

Usage:
    python step1_download_sp500_data.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
from pathlib import Path


def get_sp500_tickers():
    """Get list of S&P 500 ticker symbols"""
    # Read from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    
    # Clean ticker symbols (fix special characters)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    
    return tickers


def download_stock_data(tickers, start_date, end_date):
    """
    Download historical data for given tickers
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume, adj_close
    """
    all_data = []
    
    print(f"Downloading data for {len(tickers)} stocks...")
    print(f"Period: {start_date} to {end_date}")
    
    for ticker in tqdm(tickers):
        try:
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if len(df) == 0:
                print(f"  ⚠ No data for {ticker}")
                continue
            
            # Reset index to get date as column
            df = df.reset_index()
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Select relevant columns
            df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
            
            all_data.append(df)
            
            # Be nice to Yahoo Finance API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  ✗ Error downloading {ticker}: {e}")
            continue
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Downloaded data for {combined_df['ticker'].nunique()} stocks")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        return combined_df
    else:
        print("✗ No data downloaded!")
        return pd.DataFrame()


def main():
    # Configuration
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = Path('data')
    output_file = output_dir / 'sp500_history.csv'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("S&P 500 DATA DOWNLOAD")
    print("="*80)
    
    # Get tickers
    print("\nFetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    print(f"✓ Found {len(tickers)} tickers")
    
    # Download data
    df = download_stock_data(tickers, start_date, end_date)
    
    if not df.empty:
        # Save to CSV
        print(f"\nSaving to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"✓ Saved {len(df):,} rows to {output_file}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"File: {output_file}")
        print(f"Stocks: {df['ticker'].nunique()}")
        print(f"Total rows: {len(df):,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Columns: {', '.join(df.columns)}")
    else:
        print("\n✗ No data to save!")


if __name__ == '__main__':
    main()
