"""
Terminal-based MLflow Results Viewer
Alternative to MLflow UI when browser access isn't working.
"""

import mlflow
import pandas as pd
import os
from datetime import datetime


def view_mlflow_results():
    """Display MLflow experiment results in terminal."""

    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")

    # Get experiment
    experiment_name = "Hierarchical Power System Classifier v2"

    print("\n" + "="*80)
    print("MLflow Results Viewer")
    print("="*80)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' not found")
        print("\nAvailable experiments:")
        for exp in mlflow.search_experiments():
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        return
    
    print(f"\n📊 Experiment: {experiment_name}")
    print(f"📁 Experiment ID: {experiment.experiment_id}")
    print(f"📂 Location: {experiment.artifact_location}")
    
    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    if runs.empty:
        print("\n⚠️  No runs found in this experiment")
        return
    
    print(f"\n✅ Total Runs: {len(runs)}")
    
    # Show run statuses
    status_counts = runs['status'].value_counts()
    print("\nRun Status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Extract parameter and metric columns
    param_cols = [col for col in runs.columns if col.startswith('params.')]
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    
    # Show parameter summary
    if param_cols:
        print("\n" + "="*80)
        print("PARAMETERS")
        print("="*80)
        for col in param_cols[:10]:  # Show first 10 params
            param_name = col.replace('params.', '')
            unique_values = runs[col].nunique()
            print(f"  {param_name}: {unique_values} unique value(s)")
            if unique_values <= 5:
                print(f"    Values: {runs[col].dropna().unique().tolist()}")
    
    # Show metric summary
    if metric_cols:
        print("\n" + "="*80)
        print("METRICS SUMMARY")
        print("="*80)
        
        for metric_col in sorted(metric_cols):
            metric_name = metric_col.replace('metrics.', '')
            values = runs[metric_col].dropna()
            
            if not values.empty:
                print(f"\n📈 {metric_name}:")
                print(f"  Count: {len(values)}")
                print(f"  Mean:  {values.mean():.4f}")
                print(f"  Std:   {values.std():.4f}")
                print(f"  Min:   {values.min():.4f}")
                print(f"  Max:   {values.max():.4f}")
                
                # Show best run for this metric
                if values.max() > 0:
                    best_idx = values.idxmax()
                    best_run = runs.loc[best_idx]
                    run_id_short = best_run['run_id'][:8]
                    print(f"  Best:  {values.max():.4f} (Run: {run_id_short}...)")
    
    # Show recent runs
    print("\n" + "="*80)
    print("RECENT RUNS (Last 10)")
    print("="*80)
    
    recent_runs = runs.head(10)
    
    for idx, run in recent_runs.iterrows():
        run_id_short = run['run_id'][:8]
        start_time = run['start_time']
        status = run['status']
        
        print(f"\n🔹 Run {run_id_short}... | {status} | {start_time}")
        
        # Show key metrics if available
        key_metrics = ['level1_avg_test_accuracy', 'fault_type_avg_test_accuracy', 
                      'fault_location_avg_test_accuracy', 'attack_type_avg_test_accuracy']
        
        for metric in key_metrics:
            metric_col = f'metrics.{metric}'
            if metric_col in run and pd.notna(run[metric_col]):
                print(f"  {metric}: {run[metric_col]:.4f}")
        
        # Show PMU list if available
        if 'params.pmu_list' in run and pd.notna(run['params.pmu_list']):
            pmu_str = str(run['params.pmu_list'])
            if len(pmu_str) > 50:
                pmu_str = pmu_str[:50] + "..."
            print(f"  PMUs: {pmu_str}")
    
    # Export to CSV
    print("\n" + "="*80)
    print("EXPORTING DATA")
    print("="*80)
    
    output_file = f'mlflow_runs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    runs.to_csv(output_file, index=False)
    print(f"\n✅ Full data exported to: {output_file}")
    print(f"   Columns: {len(runs.columns)}")
    print(f"   Rows: {len(runs)}")
    
    # Create simplified summary
    summary_cols = ['run_id', 'start_time', 'end_time', 'status']
    summary_cols.extend([col for col in metric_cols if 'avg_test_accuracy' in col or 'r2' in col])
    summary_cols.extend(['params.pmu_list', 'params.cv_folds'])
    
    available_summary_cols = [col for col in summary_cols if col in runs.columns]
    
    if available_summary_cols:
        summary_file = f'mlflow_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        runs[available_summary_cols].to_csv(summary_file, index=False)
        print(f"✅ Summary exported to: {summary_file}")
    
    # Show best overall runs
    print("\n" + "="*80)
    print("TOP 5 RUNS BY LEVEL 1 ACCURACY")
    print("="*80)
    
    if 'metrics.level1_avg_test_accuracy' in runs.columns:
        top_runs = runs.nlargest(5, 'metrics.level1_avg_test_accuracy')
        
        for idx, run in top_runs.iterrows():
            run_id_short = run['run_id'][:8]
            accuracy = run['metrics.level1_avg_test_accuracy']
            print(f"\n🏆 Run {run_id_short}... | Accuracy: {accuracy:.4f}")
            
            # Show all available accuracies
            for metric in metric_cols:
                if 'accuracy' in metric and pd.notna(run[metric]):
                    metric_name = metric.replace('metrics.', '').replace('_avg_test_accuracy', '')
                    print(f"  {metric_name}: {run[metric]:.4f}")
    
    print("\n" + "="*80)
    print("Done! You can now:")
    print("  1. Open the CSV files in Excel/Numbers")
    print("  2. Run: python compare_results.py")
    print("  3. View plots in: comparison_plots/")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    try:
        view_mlflow_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you're in the project directory:")
        print("  cd /Users/sarahbanat/Desktop/Research/39BusWAttacks")
        print("\nAnd that mlruns/ directory exists:")
        print("  ls mlruns/")
        sys.exit(1)

