"""
Run all data preparation steps in sequence

This script runs all data preparation steps from scratch:
1. Download S&P 500 data from Yahoo Finance
2. Convert to Qlib format
3. Generate Alpha158 features
4. Build graph structures

Usage:
    python run_all_data_preparation.py
    
    # Or skip steps that are already done:
    python run_all_data_preparation.py --skip-download
    python run_all_data_preparation.py --start-from 3
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def check_file_exists(filepath, description):
    """Check if a file exists and show info"""
    filepath = Path(filepath)
    if filepath.exists():
        if filepath.is_file():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {description}: {filepath} ({size:.2f} MB)")
        else:
            print(f"  ✓ {description}: {filepath}")
        return True
    else:
        print(f"  ✗ {description}: {filepath} (not found)")
        return False


def run_step(step_num, description, script_name, force=False):
    """Run a data preparation step"""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {description}")
    print("="*80)
    
    # Import and run the script
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False
    
    try:
        # Execute the script
        with open(script_path) as f:
            code = f.read()
        
        # Create a new namespace for execution
        exec_globals = {'__name__': '__main__', '__file__': str(script_path)}
        exec(code, exec_globals)
        
        print(f"\n✓ Step {step_num} completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Step {step_num} failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all data preparation steps')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip step 1 (download) if data already exists')
    parser.add_argument('--start-from', type=int, choices=[1,2,3,4], default=1,
                       help='Start from specific step (1-4)')
    parser.add_argument('--only', type=int, choices=[1,2,3,4],
                       help='Run only specified step')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check what already exists
    print("Checking existing data...")
    check_file_exists('data/sp500_history.csv', 'Raw S&P 500 data')
    check_file_exists('data/qlib_format/sp500_data.csv', 'Qlib format data')
    check_file_exists('data/alpha158/alpha158_features.csv', 'Alpha158 features')
    check_file_exists('data/graph_data/sector_adj_matrix.npy', 'Sector graph')
    check_file_exists('data/graph_data/money_flow_adj_matrix.npy', 'Money flow graph')
    print()
    
    # Define steps
    steps = [
        (1, "Download S&P 500 Data", "step1_download_sp500_data.py"),
        (2, "Prepare Qlib Format", "step2_prepare_qlib_format.py"),
        (3, "Generate Alpha158 Features", "step3_generate_alpha158.py"),
        (4, "Build Graph Structures", "step4_build_graphs.py"),
    ]
    
    # Determine which steps to run
    if args.only:
        steps_to_run = [s for s in steps if s[0] == args.only]
    else:
        steps_to_run = [s for s in steps if s[0] >= args.start_from]
        
        # Skip download if requested
        if args.skip_download and Path('data/sp500_history.csv').exists():
            steps_to_run = [s for s in steps_to_run if s[0] != 1]
            print("⚠ Skipping step 1 (download) - data already exists")
    
    print(f"\nWill run {len(steps_to_run)} step(s):")
    for step_num, desc, _ in steps_to_run:
        print(f"  • Step {step_num}: {desc}")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Run steps
    start_time = datetime.now()
    success = True
    
    for step_num, description, script_name in steps_to_run:
        success = run_step(step_num, description, script_name)
        
        if not success:
            print("\n" + "="*80)
            print(f"✗ Pipeline failed at step {step_num}")
            print("="*80)
            sys.exit(1)
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("You can now proceed to training:")
    print("  python train.py")
    print("="*80)


if __name__ == '__main__':
    main()
