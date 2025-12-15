"""
Comprehensive Evaluation and Comparison Script

Compare results between:
- Baseline HSGNN (train.py)
- LLM-Enhanced HSGNN (train_llm_enhanced.py)

Generate detailed performance reports, visualizations, and statistical tests.

Usage:
    python evaluate_results.py --baseline outputs/baseline --llm outputs/llm_enhanced
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from dataset import create_dataloaders
from model import create_model
from model_llm_dynamic_graph import create_model_llm_dynamic


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, baseline_dir, llm_dir, data_dir='data'):
        """
        Args:
            baseline_dir: Directory with baseline model
            llm_dir: Directory with LLM-enhanced model
            data_dir: Data directory
        """
        self.baseline_dir = Path(baseline_dir)
        self.llm_dir = Path(llm_dir)
        self.data_dir = data_dir
        
        # Load configs
        with open(self.baseline_dir / 'config.json') as f:
            self.baseline_config = json.load(f)
        
        with open(self.llm_dir / 'config.json') as f:
            self.llm_config = json.load(f)
        
        # Load test results
        with open(self.baseline_dir / 'test_results.json') as f:
            self.baseline_results = json.load(f)
        
        with open(self.llm_dir / 'test_results.json') as f:
            self.llm_results = json.load(f)
        
        print("✓ Loaded model configs and results")
    
    def compare_metrics(self):
        """Compare basic metrics"""
        
        print("\n" + "="*80)
        print("METRICS COMPARISON")
        print("="*80)
        
        metrics = ['test_loss', 'test_rank_ic', 'best_val_rank_ic']
        
        comparison = []
        for metric in metrics:
            baseline_val = self.baseline_results.get(metric, 0)
            llm_val = self.llm_results.get(metric, 0)
            
            if 'loss' in metric:
                improvement = (baseline_val - llm_val) / baseline_val * 100
                better = "✓" if llm_val < baseline_val else "✗"
            else:
                improvement = (llm_val - baseline_val) / abs(baseline_val) * 100 if baseline_val != 0 else 0
                better = "✓" if llm_val > baseline_val else "✗"
            
            comparison.append({
                'Metric': metric,
                'Baseline': f"{baseline_val:.6f}",
                'LLM-Enhanced': f"{llm_val:.6f}",
                'Improvement': f"{improvement:+.2f}%",
                'Better': better
            })
        
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        
        return df
    
    def detailed_predictions_analysis(self, device='cpu'):
        """Analyze predictions on test set"""
        
        print("\n" + "="*80)
        print("DETAILED PREDICTIONS ANALYSIS")
        print("="*80)
        
        device = torch.device(device)
        
        # Load dataloaders
        print("\nLoading test data...")
        _, _, test_loader = create_dataloaders(
            data_dir=self.data_dir,
            batch_size=16,
            window_size=self.baseline_config.get('window_size', 150),
            num_workers=0
        )
        
        # Load models
        print("Loading baseline model...")
        baseline_checkpoint = torch.load(
            self.baseline_dir / 'best_model.pt',
            map_location=device,
            weights_only=False
        )
        
        baseline_model = create_model(
            num_alpha_features=119,
            num_risk_features=3,
            hidden_dim=self.baseline_config.get('hidden_dim', 64),
            num_gat_layers=self.baseline_config.get('num_gat_layers', 2),
            num_heads=self.baseline_config.get('num_heads', 4),
            top_k=self.baseline_config.get('top_k', 10),
            dropout=0.0,
            device=device
        )
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        baseline_model.eval()
        
        print("Loading LLM-enhanced model...")
        llm_checkpoint = torch.load(
            self.llm_dir / 'best_model.pt',
            map_location=device,
            weights_only=False
        )
        
        llm_model = create_model_llm_dynamic(
            num_alpha_features=119,
            num_risk_features=3,
            hidden_dim=self.llm_config.get('hidden_dim', 64),
            num_gat_layers=self.llm_config.get('num_gat_layers', 2),
            num_heads=self.llm_config.get('num_heads', 4),
            top_k=self.llm_config.get('top_k', 10),
            dropout=0.0,
            use_llm=self.llm_config.get('use_llm', False),
            llm_provider=self.llm_config.get('llm_provider', 'local'),
            device=device
        )
        llm_model.load_state_dict(llm_checkpoint['model_state_dict'])
        llm_model.eval()
        
        # Collect predictions
        print("Generating predictions...")
        
        baseline_preds_list = []
        llm_preds_list = []
        targets_list = []
        masks_list = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                targets = batch['y']
                mask = batch['mask']
                
                # Baseline predictions
                baseline_preds = baseline_model(batch)
                
                # LLM predictions
                llm_preds = llm_model(batch)
                
                baseline_preds_list.append(baseline_preds.cpu().numpy())
                llm_preds_list.append(llm_preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                masks_list.append(mask.cpu().numpy())
        
        # Concatenate
        baseline_preds = np.concatenate(baseline_preds_list, axis=0)
        llm_preds = np.concatenate(llm_preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)
        
        # Flatten and filter valid
        valid_mask = masks.flatten() > 0.5
        baseline_preds_flat = baseline_preds.flatten()[valid_mask]
        llm_preds_flat = llm_preds.flatten()[valid_mask]
        targets_flat = targets.flatten()[valid_mask]
        
        # Remove NaN/Inf
        finite_mask = (np.isfinite(baseline_preds_flat) & 
                      np.isfinite(llm_preds_flat) & 
                      np.isfinite(targets_flat))
        baseline_preds_flat = baseline_preds_flat[finite_mask]
        llm_preds_flat = llm_preds_flat[finite_mask]
        targets_flat = targets_flat[finite_mask]
        
        print(f"\nTotal valid predictions: {len(targets_flat):,}")
        
        # Compute detailed metrics
        results = {}
        
        # 1. Rank IC (Spearman)
        baseline_rankic = stats.spearmanr(baseline_preds_flat, targets_flat)[0]
        llm_rankic = stats.spearmanr(llm_preds_flat, targets_flat)[0]
        
        results['Rank IC'] = {
            'Baseline': baseline_rankic,
            'LLM': llm_rankic,
            'Improvement': (llm_rankic - baseline_rankic) / abs(baseline_rankic) * 100
        }
        
        # 2. MSE
        baseline_mse = np.mean((baseline_preds_flat - targets_flat) ** 2)
        llm_mse = np.mean((llm_preds_flat - targets_flat) ** 2)
        
        results['MSE'] = {
            'Baseline': baseline_mse,
            'LLM': llm_mse,
            'Improvement': (baseline_mse - llm_mse) / baseline_mse * 100
        }
        
        # 3. MAE
        baseline_mae = np.mean(np.abs(baseline_preds_flat - targets_flat))
        llm_mae = np.mean(np.abs(llm_preds_flat - targets_flat))
        
        results['MAE'] = {
            'Baseline': baseline_mae,
            'LLM': llm_mae,
            'Improvement': (baseline_mae - llm_mae) / baseline_mae * 100
        }
        
        # 4. Pearson correlation
        baseline_pearson = np.corrcoef(baseline_preds_flat, targets_flat)[0, 1]
        llm_pearson = np.corrcoef(llm_preds_flat, targets_flat)[0, 1]
        
        results['Pearson Corr'] = {
            'Baseline': baseline_pearson,
            'LLM': llm_pearson,
            'Improvement': (llm_pearson - baseline_pearson) / abs(baseline_pearson) * 100
        }
        
        # Print results
        print("\nDetailed Metrics:")
        print("-" * 80)
        for metric, values in results.items():
            print(f"\n{metric}:")
            print(f"  Baseline:     {values['Baseline']:.6f}")
            print(f"  LLM-Enhanced: {values['LLM']:.6f}")
            print(f"  Improvement:  {values['Improvement']:+.2f}%")
        
        return results, {
            'baseline_preds': baseline_preds_flat,
            'llm_preds': llm_preds_flat,
            'targets': targets_flat
        }
    
    def statistical_significance_test(self, predictions):
        """Test if improvement is statistically significant"""
        
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        baseline_preds = predictions['baseline_preds']
        llm_preds = predictions['llm_preds']
        targets = predictions['targets']
        
        # Compute errors
        baseline_errors = np.abs(baseline_preds - targets)
        llm_errors = np.abs(llm_preds - targets)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(baseline_errors, llm_errors)
        
        print(f"\nPaired t-test (MAE comparison):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        
        if p_value < 0.001:
            print(f"  ✓ Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"  ✓ Very significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"  ✓ Significant (p < 0.05)")
        else:
            print(f"  ✗ Not significant (p >= 0.05)")
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pvalue = stats.wilcoxon(baseline_errors, llm_errors)
        
        print(f"\nWilcoxon signed-rank test:")
        print(f"  W-statistic: {w_stat:.4f}")
        print(f"  p-value:     {w_pvalue:.6f}")
        
        if w_pvalue < 0.05:
            print(f"  ✓ Significant improvement (p < 0.05)")
        else:
            print(f"  ✗ Not significant (p >= 0.05)")
        
        return {
            't_test': {'t_stat': t_stat, 'p_value': p_value},
            'wilcoxon': {'w_stat': w_stat, 'p_value': w_pvalue}
        }
    
    def generate_visualizations(self, predictions, output_dir='outputs/evaluation'):
        """Generate comparison visualizations"""
        
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_preds = predictions['baseline_preds']
        llm_preds = predictions['llm_preds']
        targets = predictions['targets']
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Scatter plots comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Baseline
        axes[0].scatter(targets, baseline_preds, alpha=0.3, s=1)
        axes[0].plot([targets.min(), targets.max()], 
                     [targets.min(), targets.max()], 
                     'r--', linewidth=2, label='Perfect prediction')
        axes[0].set_xlabel('True Returns')
        axes[0].set_ylabel('Predicted Returns')
        axes[0].set_title(f'Baseline HSGNN (Rank IC: {stats.spearmanr(baseline_preds, targets)[0]:.4f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # LLM
        axes[1].scatter(targets, llm_preds, alpha=0.3, s=1)
        axes[1].plot([targets.min(), targets.max()], 
                     [targets.min(), targets.max()], 
                     'r--', linewidth=2, label='Perfect prediction')
        axes[1].set_xlabel('True Returns')
        axes[1].set_ylabel('Predicted Returns')
        axes[1].set_title(f'LLM-Enhanced HSGNN (Rank IC: {stats.spearmanr(llm_preds, targets)[0]:.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_scatter.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'predictions_scatter.png'}")
        plt.close()
        
        # 2. Error distributions
        baseline_errors = baseline_preds - targets
        llm_errors = llm_preds - targets
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].hist(baseline_errors, bins=100, alpha=0.7, label='Baseline', color='blue')
        axes[0].hist(llm_errors, bins=100, alpha=0.7, label='LLM-Enhanced', color='orange')
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative errors
        axes[1].hist(np.abs(baseline_errors), bins=100, alpha=0.7, 
                    label=f'Baseline (MAE: {np.mean(np.abs(baseline_errors)):.6f})',
                    cumulative=True, density=True, color='blue')
        axes[1].hist(np.abs(llm_errors), bins=100, alpha=0.7,
                    label=f'LLM-Enhanced (MAE: {np.mean(np.abs(llm_errors)):.6f})',
                    cumulative=True, density=True, color='orange')
        axes[1].set_xlabel('Absolute Error')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Absolute Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'error_distributions.png'}")
        plt.close()
        
        # 3. Metrics comparison bar chart
        metrics = ['Rank IC', 'Pearson Corr', 'MSE', 'MAE']
        baseline_vals = [
            stats.spearmanr(baseline_preds, targets)[0],
            np.corrcoef(baseline_preds, targets)[0, 1],
            np.mean((baseline_preds - targets) ** 2),
            np.mean(np.abs(baseline_preds - targets))
        ]
        llm_vals = [
            stats.spearmanr(llm_preds, targets)[0],
            np.corrcoef(llm_preds, targets)[0, 1],
            np.mean((llm_preds - targets) ** 2),
            np.mean(np.abs(llm_preds - targets))
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            x = ['Baseline', 'LLM-Enhanced']
            y = [baseline_vals[i], llm_vals[i]]
            
            colors = ['blue', 'orange']
            axes[i].bar(x, y, color=colors, alpha=0.7)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for j, val in enumerate(y):
                axes[i].text(j, val, f'{val:.6f}', 
                           ha='center', va='bottom' if val > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'metrics_comparison.png'}")
        plt.close()
        
        print(f"\n✓ All visualizations saved to {output_dir}/")
    
    def generate_report(self, output_dir='outputs/evaluation'):
        """Generate comprehensive evaluation report"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Run all analyses
        metrics_comparison = self.compare_metrics()
        
        try:
            detailed_results, predictions = self.detailed_predictions_analysis()
            significance_tests = self.statistical_significance_test(predictions)
            self.generate_visualizations(predictions, output_dir)
            has_predictions = True
        except Exception as e:
            print(f"\n⚠ Warning: Could not generate detailed analysis: {e}")
            print("  Continuing with basic metrics only...")
            has_predictions = False
        
        # Save report to file
        report_path = output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HSGNN MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("Baseline Model: " + str(self.baseline_dir) + "\n")
            f.write("LLM-Enhanced Model: " + str(self.llm_dir) + "\n\n")
            
            f.write("-"*80 + "\n")
            f.write("BASIC METRICS COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(metrics_comparison.to_string(index=False) + "\n\n")
            
            if has_predictions:
                f.write("-"*80 + "\n")
                f.write("DETAILED METRICS\n")
                f.write("-"*80 + "\n")
                for metric, values in detailed_results.items():
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Baseline:     {values['Baseline']:.6f}\n")
                    f.write(f"  LLM-Enhanced: {values['LLM']:.6f}\n")
                    f.write(f"  Improvement:  {values['Improvement']:+.2f}%\n")
                
                f.write("\n" + "-"*80 + "\n")
                f.write("STATISTICAL SIGNIFICANCE\n")
                f.write("-"*80 + "\n")
                f.write(f"\nPaired t-test:\n")
                f.write(f"  t-statistic: {significance_tests['t_test']['t_stat']:.4f}\n")
                f.write(f"  p-value:     {significance_tests['t_test']['p_value']:.6f}\n")
                
                f.write(f"\nWilcoxon test:\n")
                f.write(f"  W-statistic: {significance_tests['wilcoxon']['w_stat']:.4f}\n")
                f.write(f"  p-value:     {significance_tests['wilcoxon']['p_value']:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n\n")
            
            if has_predictions:
                improvement = detailed_results['Rank IC']['Improvement']
                pvalue = significance_tests['t_test']['p_value']
                
                if improvement > 0 and pvalue < 0.05:
                    f.write("✓ LLM-Enhanced model shows STATISTICALLY SIGNIFICANT improvement\n")
                    f.write(f"  - Rank IC improvement: {improvement:+.2f}%\n")
                    f.write(f"  - Statistical significance: p = {pvalue:.6f}\n")
                elif improvement > 0:
                    f.write("⚠ LLM-Enhanced model shows improvement but NOT statistically significant\n")
                    f.write(f"  - Rank IC improvement: {improvement:+.2f}%\n")
                    f.write(f"  - p-value: {pvalue:.6f} (>= 0.05)\n")
                else:
                    f.write("✗ LLM-Enhanced model does NOT show improvement\n")
            else:
                f.write("⚠ Detailed analysis not available. See basic metrics above.\n")
        
        print(f"\n✓ Report saved to: {report_path}")
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare HSGNN models')
    
    parser.add_argument('--baseline', type=str, required=True,
                       help='Baseline model directory (e.g., outputs/baseline)')
    parser.add_argument('--llm', type=str, required=True,
                       help='LLM-enhanced model directory (e.g., outputs/llm_enhanced)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        baseline_dir=args.baseline,
        llm_dir=args.llm,
        data_dir=args.data_dir
    )
    
    # Generate comprehensive report
    report_path = evaluator.generate_report(output_dir=args.output_dir)
    
    print(f"\n📊 Evaluation complete!")
    print(f"   Report: {report_path}")
    print(f"   Visualizations: {args.output_dir}/")


if __name__ == '__main__':
    main()
