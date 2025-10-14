"""
Results Comparison Tool for Experiment Analysis
Aggregates and compares results across all experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from pathlib import Path
import mlflow

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def extract_results_from_csv(results_dir='results2'):
    """Extract metrics from CSV result files."""
    results = {
        'level1': [],
        'fault_type': [],
        'fault_location': [],
        'distance': [],
        'attack_type': []
    }
    
    # Read each result file
    for key in results.keys():
        csv_path = os.path.join(results_dir, f'{key}_results.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found")
            continue
        
        df = pd.read_csv(csv_path, header=None)
        
        # Parse experiments from the CSV
        current_exp = None
        for idx, row in df.iterrows():
            row_str = str(row[0]) if not pd.isna(row[0]) else ''
            
            # Detect experiment number
            if row_str.startswith('Exp '):
                try:
                    current_exp = int(row_str.split()[1])
                except:
                    pass
            
            # Extract average testing score
            if 'Average Testing Score' in row_str or 'Average Test' in row_str:
                if current_exp is not None and len(row) > 1:
                    try:
                        score = float(str(row[1]).strip())
                        results[key].append({
                            'exp_num': current_exp,
                            'metric': key,
                            'accuracy': score
                        })
                    except:
                        pass
    
    return results


def extract_results_from_mlflow(experiment_name="Hierarchical Power System Classifier v2"):
    """Extract results from MLflow tracking."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found in MLflow")
            return None
        
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print("No runs found in MLflow")
            return None
        
        print(f"Found {len(runs)} runs in MLflow")
        
        # Extract key metrics
        metrics_columns = [col for col in runs.columns if col.startswith('metrics.')]
        params_columns = [col for col in runs.columns if col.startswith('params.')]
        
        return runs[['run_id', 'start_time', 'end_time'] + params_columns + metrics_columns]
    
    except Exception as e:
        print(f"Error extracting from MLflow: {e}")
        return None


def create_comparison_plots(results_df, output_dir='comparison_plots'):
    """Create comparison plots across experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy vs Number of PMUs
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance vs Number of PMUs', fontsize=16, fontweight='bold')
    
    metrics = ['level1', 'fault_type', 'fault_location', 'attack_type']
    metric_titles = {
        'level1': 'Level 1: Normal/Attack/Fault',
        'fault_type': 'Fault Type Classification',
        'fault_location': 'Fault Location Classification',
        'attack_type': 'Attack Type Classification'
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        metric_data = results_df[results_df['metric'] == metric].sort_values('exp_num')
        
        if not metric_data.empty:
            ax.plot(metric_data['exp_num'], metric_data['accuracy'], 
                   marker='o', linewidth=2, markersize=8, label=metric)
            ax.set_xlabel('Experiment Number', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(metric_titles.get(metric, metric), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
    
    # Remove empty subplots
    for idx in range(len(metrics), 6):
        fig.delaxes(axes[idx // 3, idx % 3])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_comparison.png")
    plt.close()
    
    # Plot 2: Heatmap of all results
    pivot_data = results_df.pivot(index='exp_num', columns='metric', values='accuracy')
    
    if not pivot_data.empty:
        fig, ax = plt.subplots(figsize=(10, 12))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
        ax.set_title('Experiment Results Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Classification Task', fontsize=12)
        ax.set_ylabel('Experiment Number', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/results_heatmap.png")
        plt.close()
    
    # Plot 3: Best performing experiments
    best_per_metric = results_df.loc[results_df.groupby('metric')['accuracy'].idxmax()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(best_per_metric['metric'], best_per_metric['accuracy'], 
                  color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_ylabel('Best Accuracy (%)', fontsize=12)
    ax.set_xlabel('Classification Task', fontsize=12)
    ax.set_title('Best Performance per Task', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bar, exp_num in zip(bars, best_per_metric['exp_num']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.2f}%\n(Exp {exp_num})',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_performance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/best_performance.png")
    plt.close()


def generate_comparison_report(results_dir='results2', output_dir='comparison_plots'):
    """Generate comprehensive comparison report."""
    print("=" * 80)
    print("GENERATING EXPERIMENT COMPARISON REPORT")
    print("=" * 80)
    
    # Extract results from CSV files
    results = extract_results_from_csv(results_dir)
    
    # Combine all results into a single DataFrame
    all_results = []
    for metric, data in results.items():
        all_results.extend(data)
    
    if not all_results:
        print("No results found. Please run experiments first.")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Create comparison plots
    os.makedirs(output_dir, exist_ok=True)
    create_comparison_plots(results_df, output_dir)
    
    # Generate statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_stats = results_df.groupby('metric')['accuracy'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    print(summary_stats)
    
    # Best experiment per metric
    print("\n" + "=" * 80)
    print("BEST PERFORMING EXPERIMENTS")
    print("=" * 80)
    
    for metric in results_df['metric'].unique():
        metric_data = results_df[results_df['metric'] == metric]
        best_exp = metric_data.loc[metric_data['accuracy'].idxmax()]
        print(f"{metric:20s}: Exp {best_exp['exp_num']:2d} with {best_exp['accuracy']:.2f}% accuracy")
    
    # Save summary to CSV
    summary_path = os.path.join(output_dir, 'experiment_comparison_summary.csv')
    results_df.to_csv(summary_path, index=False)
    print(f"\n✅ Detailed results saved to: {summary_path}")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(stats_path)
    print(f"✅ Summary statistics saved to: {stats_path}")
    
    # Try to extract from MLflow
    print("\n" + "=" * 80)
    print("EXTRACTING MLFLOW DATA")
    print("=" * 80)
    
    mlflow_data = extract_results_from_mlflow()
    if mlflow_data is not None:
        mlflow_path = os.path.join(output_dir, 'mlflow_runs_summary.csv')
        mlflow_data.to_csv(mlflow_path, index=False)
        print(f"✅ MLflow data saved to: {mlflow_path}")
    
    print("\n" + "=" * 80)
    print("COMPARISON REPORT COMPLETE")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--results-dir', default='results2', help='Directory containing result CSV files')
    parser.add_argument('--output-dir', default='comparison_plots', help='Directory to save comparison plots')
    
    args = parser.parse_args()
    
    generate_comparison_report(args.results_dir, args.output_dir)


