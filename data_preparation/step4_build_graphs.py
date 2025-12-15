"""
Step 4: Build graph structures (sector and money flow graphs)

Creates graph adjacency matrices for stock relationships.
Saves to: data/graph_data/

This is a wrapper around build_graphs.py and fetch_sectors.py in parent directory.

Usage:
    python step4_build_graphs.py
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def main():
    print("="*80)
    print("GRAPH STRUCTURE GENERATION")
    print("="*80)
    print("\nThis will create stock relationship graphs:")
    print("  1. Sector-based graph (stocks in same sector)")
    print("  2. Money flow graph (price correlation)")
    print()
    print("Output: data/graph_data/")
    print()
    
    # Check if alpha158 data exists
    alpha_file = Path('data/alpha158/alpha158_features.csv')
    if not alpha_file.exists():
        print(f"✗ Alpha158 features not found: {alpha_file}")
        print("  Run step3_generate_alpha158.py first!")
        return
    
    # Step 4.1: Fetch sector information
    print("\n" + "-"*80)
    print("Step 4.1: Fetching sector information...")
    print("-"*80)
    
    from fetch_sectors import main as fetch_sectors
    fetch_sectors()
    
    # Step 4.2: Build graphs
    print("\n" + "-"*80)
    print("Step 4.2: Building graph structures...")
    print("-"*80)
    
    from build_graphs import main as build_graphs
    build_graphs()
    
    print("\n" + "="*80)
    print("✓ Graph generation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
