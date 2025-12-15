"""
Step 3: Generate Alpha158 features from Qlib data

Creates 158 technical features for stock prediction.
Saves to: data/alpha158/

This is a wrapper around d_generate_alpha158.py in the parent directory.

Usage:
    python step3_generate_alpha158.py
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import the actual alpha158 generation module
from d_generate_alpha158 import main as generate_alpha158


def main():
    print("="*80)
    print("ALPHA158 FEATURE GENERATION")
    print("="*80)
    print("\nThis will generate 158 technical features for stock prediction.")
    print("Input:  data/qlib_format/sp500_data.csv")
    print("Output: data/alpha158/")
    print()
    
    # Check if input exists
    input_file = Path('data/qlib_format/sp500_data.csv')
    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        print("  Run step2_prepare_qlib_format.py first!")
        return
    
    # Run alpha158 generation
    generate_alpha158()


if __name__ == '__main__':
    main()
