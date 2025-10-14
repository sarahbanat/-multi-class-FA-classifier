import mlflow
import numpy as np
import json
import logging
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def safe_metric_log(metric_key, metric_value, step=None):
    """Logs metric to MLflow, handling potential non-finite values."""
    if metric_value is None or not np.isfinite(metric_value):
        logging.warning(f"Skipping non-finite metric '{metric_key}': {metric_value}")
    else:
        mlflow.log_metric(metric_key, metric_value, step=step)


def classification_report_to_csv(report_dict, writer):

    first_class_label = next((k for k in report_dict if k not in ('accuracy', 'macro avg', 'weighted avg')), None)
    if not first_class_label:
        logging.warning("Could not find any class labels in classification report for CSV header.")
        writer.writerow(["Classification Report Error: No class labels found"]) 
        return

    header = ['Class'] + list(report_dict[first_class_label].keys())
    writer.writerow(header)
    for class_label, metrics in report_dict.items():
        if class_label not in ('accuracy', 'macro avg', 'weighted avg'):
             row_values = [f"{metrics.get(key, 'N/A'):.4f}" if isinstance(metrics.get(key), (int, float)) else metrics.get(key, 'N/A') for key in header[1:]]
             writer.writerow([class_label] + row_values)
    for avg_type in ['macro avg', 'weighted avg']:
         if avg_type in report_dict:
              metrics = report_dict[avg_type]
              row = [avg_type] + [f"{metrics[key]:.4f}" for key in header[1:] if key in metrics]
              if 'support' in header[1:] and 'support' in metrics:
                   row.insert(header[1:].index('support') + 1, metrics['support'])
              elif 'support' in header[1:]: 
                   row.insert(header[1:].index('support') + 1, '') 
              writer.writerow(row)

    if 'accuracy' in report_dict:
         writer.writerow(['accuracy', f"{report_dict['accuracy']:.4f}"] + [''] * (len(header) - 2))


def log_classification_results(y_true, y_pred, scores_cv, prefix: str, exp_name: str, exp_num: int, file_directory: str):

    logging.info(f"Logging classification results for prefix: {prefix}")

    avg_test_score = scores_cv.get('test_score', [np.nan]).mean()
    avg_train_score = scores_cv.get('train_score', [np.nan]).mean()
    safe_metric_log(f"{prefix}_avg_test_accuracy", avg_test_score)
    safe_metric_log(f"{prefix}_avg_train_accuracy", avg_train_score)

    run_artifact_dir = None
    try:
        active_run = mlflow.active_run()
        if active_run:
            run_id = active_run.info.run_id
            run_artifact_dir = os.path.join(file_directory, "run_artifacts", run_id)
            os.makedirs(run_artifact_dir, exist_ok=True)
            logging.info(f"Saving run-specific artifacts to: {run_artifact_dir}")
        else:
             logging.warning("No active MLflow run found. Saving artifacts to main directory.")
             run_artifact_dir = file_directory 
    except Exception as e:
        logging.error(f"Error determining/creating artifact directory: {e}")
        run_artifact_dir = file_directory 

    report_dict = {} 
    
    try:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_path = os.path.join(run_artifact_dir, f"{prefix}_classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(report_path)
        logging.info(f"Logged classification report artifact: {report_path}")
        safe_metric_log(f"{prefix}_report_accuracy", report_dict.get('accuracy', np.nan))
    except Exception as e:
        logging.error(f"Failed to generate/log classification report artifact for {prefix}: {e}")

    cm = None 
    try:
        cm = confusion_matrix(y_true, y_pred)
        cm_path_txt = os.path.join(run_artifact_dir, f"{prefix}_confusion_matrix.txt")
        np.savetxt(cm_path_txt, cm, fmt='%d')
        mlflow.log_artifact(cm_path_txt)
        logging.info(f"Logged confusion matrix artifact (txt): {cm_path_txt}")

        try:
            plt.figure(figsize=(10, 7))
            tick_labels = sorted(list(set(y_true) | set(y_pred))) 
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
            plt.title(f'{prefix.replace("_", " ").title()} Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            cm_path_png = os.path.join(run_artifact_dir, f"{prefix}_confusion_matrix.png")
            plt.savefig(cm_path_png)
            mlflow.log_artifact(cm_path_png)
            plt.close()
            logging.info(f"Logged confusion matrix artifact (png): {cm_path_png}")
        except Exception as e:
            logging.warning(f"Failed to plot/log confusion matrix for {prefix}: {e}")

    except Exception as e:
        logging.error(f"Failed to generate/log confusion matrix artifacts for {prefix}: {e}")

    csv_file_name = os.path.join(file_directory, f"{prefix}_results.csv")
    logging.info(f"Writing/Appending classification results to CSV: {csv_file_name}")
    file_exists = os.path.isfile(csv_file_name)
    try:
        with open(csv_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Experiment Name"]) 
            
            writer.writerow([''])
            writer.writerow(['-----------------------------------'])
            writer.writerow([''])
            writer.writerow([exp_name])
            writer.writerow([f"Exp {exp_num}"])
            writer.writerow([f"Results for: {prefix}"])
            writer.writerow([''])

            if 'train_score' in scores_cv:
                writer.writerow(["Training Scores (Accuracy):"])
                writer.writerow([f"{s*100:.4f}" for s in scores_cv['train_score']])
                writer.writerow(["Average Training Score:", f"{avg_train_score*100:.4f}"])
                writer.writerow([''])
            if 'test_score' in scores_cv:
                writer.writerow(["Testing Scores (Accuracy):"])
                writer.writerow([f"{s*100:.4f}" for s in scores_cv['test_score']])
                writer.writerow(["Average Testing Score:", f"{avg_test_score*100:.4f}"])
                writer.writerow([''])

            if report_dict:
                writer.writerow(["Classification Report:"])
                classification_report_to_csv(report_dict, writer)
                writer.writerow([''])
            else:
                writer.writerow(["Classification Report: Not Available"])
                writer.writerow([''])

            if cm is not None:
                writer.writerow(["Confusion Matrix:"])
                
                unique_labels = sorted(list(set(y_true) | set(y_pred)))
                writer.writerow(['Actual \ Predicted'] + [str(label) for label in unique_labels])
                 
                for i, label in enumerate(unique_labels):
                     writer.writerow([str(label)] + list(cm[i]))
                writer.writerow([''])

                try:
                    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
                    writer.writerow(["Normalized Confusion Matrix:"])
                     
                    writer.writerow(['Actual \ Predicted'] + [str(label) for label in unique_labels])
                     
                    for i, label in enumerate(unique_labels):
                        writer.writerow([str(label)] + [f"{val:.4f}" for val in cm_normalized[i]])
                    writer.writerow([''])
                except Exception as e:
                     logging.warning(f"Could not compute/write normalized confusion matrix for {prefix}: {e}")
                     writer.writerow(["Normalized Confusion Matrix: Not Available"])
                     writer.writerow([''])

            else:
                writer.writerow(["Confusion Matrix: Not Available"])
                writer.writerow([''])

    except IOError as e:
        logging.error(f"Error writing classification results to CSV {csv_file_name}: {e}")
    except Exception as e:
         logging.error(f"An unexpected error occurred during CSV writing for {prefix}: {e}")


def log_regression_results(scores_cv, prefix: str, exp_name: str, exp_num: int, file_directory: str):
    """Logs (MLflow) regression metrics and writes detailed CSV results.

    Args:
        scores_cv: Dictionary of scores from cross_validate.
        prefix: Prefix for metric names (e.g., 'distance').
        exp_name: Name of the experiment for CSV reporting.
        exp_num: Number of the experiment for CSV reporting.
        file_directory: Directory to save CSV results.
    """
    logging.info(f"Logging regression results for prefix: {prefix}")

    for score_name, values in scores_cv.items():
        if not isinstance(values, (list, np.ndarray)):
            values = [values]
            
        avg_value = np.mean([v for v in values if np.isfinite(v)]) 

        if score_name.startswith('test_'):
            metric_key = f"{prefix}_{score_name.replace('test_', '').replace('neg_', '')}"
            log_value = -avg_value if 'neg_' in score_name else avg_value
            safe_metric_log(metric_key, log_value)
        elif score_name.startswith('train_'): 
             metric_key = f"{prefix}_{score_name.replace('train_', '').replace('neg_', '')}_train"
             log_value = -avg_value if 'neg_' in score_name else avg_value
             safe_metric_log(metric_key, log_value)

    csv_file_name = os.path.join(file_directory, f"{prefix}_results.csv")
    logging.info(f"Writing regression results to CSV: {csv_file_name}")
    file_exists = os.path.isfile(csv_file_name)
    metric_map = { 
        'neg_mean_squared_error': 'Mean Squared Error',
        'neg_root_mean_squared_error': 'Root Mean Squared Error',
        'r2': 'R2 Score',
        'neg_mean_absolute_error': 'Mean Absolute Error',
        'neg_mean_absolute_percentage_error': 'Mean Absolute Percentage Error'
    }
    readable_metric_names = { 
        'fit_time': 'Fit Time (s)',
        'score_time': 'Score Time (s)',
        **metric_map 
    }
    
    header_written = file_exists 

    try:
        with open(csv_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            
            if not header_written:
                writer.writerow(["Experiment Name"]) 
            
            writer.writerow([''])
            writer.writerow(['-----------------------------------'])
            writer.writerow([''])
            writer.writerow([exp_name])
            writer.writerow([f"Exp {exp_num}"])
            writer.writerow([f"Results for: {prefix}"])
            writer.writerow([''])

            dynamic_header = ["Metric"]
            
            num_folds = 0
            if scores_cv:
                first_metric_values = next(iter(scores_cv.values()))
                if isinstance(first_metric_values, (list, np.ndarray)):
                    num_folds = len(first_metric_values)
            
            for i in range(num_folds):
                dynamic_header.append(f"Fold {i+1}")
            dynamic_header.append("Average")
            dynamic_header.append("Std Dev")
            writer.writerow(dynamic_header)

            for score_key, values in scores_cv.items():
                if not isinstance(values, (list, np.ndarray)): 
                    values = [values]

                is_neg_metric = score_key.startswith('neg_')
                display_values = [-v if is_neg_metric and np.isfinite(v) else v for v in values]
                
                readable_name = readable_metric_names.get(score_key.replace('test_', '').replace('train_', ''), score_key)
                if score_key.startswith('train_'):
                     readable_name += " (Train)"
                
                row = [readable_name]
                
                finite_display_values = [v for v in display_values if np.isfinite(v)]
                if not finite_display_values: 
                    avg_score = np.nan
                    std_dev = np.nan
                else:
                    avg_score = np.mean(finite_display_values)
                    std_dev = np.std(finite_display_values)

                for val in display_values:
                    row.append(f"{val:.4f}" if np.isfinite(val) else "N/A")
                
                row.append(f"{avg_score:.4f}" if np.isfinite(avg_score) else "N/A")
                row.append(f"{std_dev:.4f}" if np.isfinite(std_dev) else "N/A")
                writer.writerow(row)
            
            writer.writerow(['']) 

    except IOError as e:
        logging.error(f"Error writing regression results to CSV {csv_file_name}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during regression CSV writing for {prefix}: {e}", exc_info=True) 