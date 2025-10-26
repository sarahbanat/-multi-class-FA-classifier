import logging
import time
import pandas as pd
from datetime import datetime
from src.train_pipeline import run_training_pipeline
import src.train_pipeline as pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def parse_exp_list(file_path='exp_list.txt'):
    experiments = []
    
    logging.info(f"Reading experiment list from {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) == 2:
            exp_num = int(parts[0])
            pmu_str = parts[1]
            pmu_list = [int(p.strip()) for p in pmu_str.split(',')]
            experiments.append({
                'exp_num': exp_num,
                'pmu_list': pmu_list,
                'num_pmus': len(pmu_list)
            })
    
    logging.info(f"Loaded {len(experiments)} experiments")
    return experiments


def run_all_experiments(start_from=0, end_at=None, skip_existing=False, level1_config=None):
    experiments = parse_exp_list()
    
    experiments = [e for e in experiments if e['exp_num'] >= start_from]
    if end_at is not None:
        experiments = [e for e in experiments if e['exp_num'] <= end_at]
    
    logging.info(f"Running {len(experiments)} experiments (Exp {start_from} to {end_at or 'end'})")
    
    results_summary = []
    
    for i, exp_config in enumerate(experiments):
        exp_num = exp_config['exp_num']
        pmu_list = exp_config['pmu_list']
        num_pmus = exp_config['num_pmus']
        
        logging.info("=" * 80)
        logging.info(f"EXPERIMENT {exp_num}/{len(experiments)-1} - Using {num_pmus} PMUs: {pmu_list}")
        logging.info("=" * 80)
        
        pipeline.NEEDED_PMUS = pmu_list
        pipeline.EXP_NUM = exp_num
        pipeline.EXPERIMENT_NAME = f"Exp_{exp_num:02d}_PMUs_{num_pmus}"

        if hasattr(pipeline, "configure_level1"):
            if level1_config:
                pipeline.configure_level1(**level1_config)
            else:
                pipeline.configure_level1()
        
        start_time = time.time()
        
        try:
            run_training_pipeline()
            
            elapsed_time = time.time() - start_time
            
            results_summary.append({
                'exp_num': exp_num,
                'num_pmus': num_pmus,
                'pmu_list': str(pmu_list),
                'status': 'SUCCESS',
                'time_seconds': round(elapsed_time, 2),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            logging.info(f"Experiment {exp_num} completed successfully in {elapsed_time:.2f}s")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            
            results_summary.append({
                'exp_num': exp_num,
                'num_pmus': num_pmus,
                'pmu_list': str(pmu_list),
                'status': 'FAILED',
                'error': str(e),
                'time_seconds': round(elapsed_time, 2),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            logging.error(f"Experiment {exp_num} failed: {e}", exc_info=True)
            logging.info("Continuing to next experiment...")
    
    summary_df = pd.DataFrame(results_summary)
    summary_path = f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    summary_df.to_csv(summary_path, index=False)
    
    logging.info("=" * 80)
    logging.info("EXPERIMENT RUN COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Summary saved to: {summary_path}")
    logging.info(f"Total experiments: {len(results_summary)}")
    logging.info(f"Successful: {sum(1 for r in results_summary if r['status'] == 'SUCCESS')}")
    logging.info(f"Failed: {sum(1 for r in results_summary if r['status'] == 'FAILED')}")
    
    return results_summary


def run_single_experiment(exp_num, level1_config=None):
    experiments = parse_exp_list()
    exp_config = next((e for e in experiments if e['exp_num'] == exp_num), None)
    
    if exp_config is None:
        logging.error(f"Experiment {exp_num} not found in exp_list.txt")
        return
    
    pmu_list = exp_config['pmu_list']
    num_pmus = exp_config['num_pmus']
    
    logging.info(f"Running Experiment {exp_num} with {num_pmus} PMUs: {pmu_list}")
    
    pipeline.NEEDED_PMUS = pmu_list
    pipeline.EXP_NUM = exp_num
    pipeline.EXPERIMENT_NAME = f"Exp_{exp_num:02d}_PMUs_{num_pmus}"

    if hasattr(pipeline, "configure_level1"):
        if level1_config:
            pipeline.configure_level1(**level1_config)
        else:
            pipeline.configure_level1()
    
    run_training_pipeline()
    
    logging.info(f"Experiment {exp_num} completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run power system experiments')
    parser.add_argument('--exp', type=int, help='Run a single experiment number')
    parser.add_argument('--start', type=int, default=0, help='Start from experiment number')
    parser.add_argument('--end', type=int, help='End at experiment number')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    
    args = parser.parse_args()
    
    if args.list:
        experiments = parse_exp_list()
        print("\nAvailable Experiments:")
        print("-" * 80)
        for exp in experiments:
            print(f"Exp {exp['exp_num']:2d}: {exp['num_pmus']:2d} PMUs - {exp['pmu_list'][:5]}{'...' if len(exp['pmu_list']) > 5 else ''}")
        print("-" * 80)
    
    elif args.exp is not None:
        run_single_experiment(args.exp)
    
    else:
        run_all_experiments(start_from=args.start, end_at=args.end)


