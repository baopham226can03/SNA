"""
Step 2: Convert raw S&P 500 data to Qlib format

Converts downloaded data to Qlib's expected format with proper date handling.
Saves to: data/qlib_format/sp500_data.csv

Usage:
    python step2_prepare_qlib_format.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def prepare_qlib_format(input_file, output_file):
    """
    Convert raw data to Qlib format
    
    Expected Qlib format:
        - Columns: date, instrument (ticker), open, high, low, close, volume, factor (adjustment factor)
        - Date format: YYYY-MM-DD
        - Sorted by: instrument, date
    """
    print("="*80)
    print("QLIB FORMAT PREPARATION")
    print("="*80)
    
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"✓ Loaded {len(df):,} rows")
    print(f"  Stocks: {df['ticker'].nunique()}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Rename ticker to instrument
    df = df.rename(columns={'ticker': 'instrument'})
    
    # Add factor column (adjustment factor = 1.0 for now)
    df['factor'] = 1.0
    
    # Select and reorder columns for Qlib
    qlib_columns = ['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'factor']
    df = df[qlib_columns]
    
    # Sort by instrument and date
    df = df.sort_values(['instrument', 'date']).reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'instrument'], keep='first')
    
    # Remove rows with missing critical data
    print("\nData cleaning...")
    print(f"  Before: {len(df):,} rows")
    
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df[df['volume'] > 0]  # Remove zero volume days
    
    print(f"  After:  {len(df):,} rows")
    
    # Format date as string
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved {len(df):,} rows")
    
    # Summary
    print("\n" + "="*80)
    print("QLIB FORMAT SUMMARY")
    print("="*80)
    print(f"File: {output_file}")
    print(f"Stocks: {df['instrument'].nunique()}")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Sample data:")
    print(df.head(10))


def main():
    input_file = Path('data/sp500_history.csv')
    output_file = Path('data/qlib_format/sp500_data.csv')
    
    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        print("  Run step1_download_sp500_data.py first!")
        return
    
    prepare_qlib_format(input_file, output_file)


if __name__ == '__main__':
    main()
