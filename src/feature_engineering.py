import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import fnmatch
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FLAG_COL_IDX = 0
FAULT_TYPE_COL_IDX = 3
FAULT_LOCATION_COL_IDX = 4
DISTANCE_COL_IDX = 5 # 'X' column
ATTACK_TYPE_COL_IDX = 6
FIRST_FEATURE_COL_IDX = 8 # Start of V1_Mag

def get_features_labels(df: pd.DataFrame, pmu_list: List[int]) -> Tuple:

    logging.info(f"Selecting features for PMUs: {pmu_list}")
    all_column_names = list(df.columns)
    feature_column_names = all_column_names[FIRST_FEATURE_COL_IDX:]
    
    if not pmu_list: 
        logging.warning("PMU list is empty, using all measurement features.")
        selected_feature_cols = feature_column_names
    else:
        selected_feature_cols = []
        for i in pmu_list:
            pattern_v = f'V{i}_*'
            pattern_i = f'I{i}_*'
            selected_feature_cols.extend(fnmatch.filter(feature_column_names, pattern_v))
            selected_feature_cols.extend(fnmatch.filter(feature_column_names, pattern_i))
        
        if not selected_feature_cols:
             raise ValueError("No feature columns selected. Check PMU list and column names.")

    logging.info(f"Number of selected feature columns: {len(selected_feature_cols)}")
    X = df[selected_feature_cols].copy()

    logging.info("Scaling features using StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logging.info("Extracting labels")
    y_level1 = df.iloc[:, FLAG_COL_IDX].copy()
    y_fault_type = df.iloc[:, FAULT_TYPE_COL_IDX].copy()
    y_fault_location = df.iloc[:, FAULT_LOCATION_COL_IDX].copy()
    y_distance = df.iloc[:, DISTANCE_COL_IDX].copy()
    y_attack_type = df.iloc[:, ATTACK_TYPE_COL_IDX].copy()

    assert X_scaled.shape[0] == len(y_level1), "Mismatch between feature and label rows"

    return X_scaled, y_level1, y_fault_type, y_fault_location, y_distance, y_attack_type


if __name__ == '__main__':
    from data_loader import load_data
    data_path = 'data/39_FullDataset_normal_faullt_attack.csv'
    needed_pmus = [1]
    df = load_data(data_path)
    X_s, yl1, yft, yfl, yd, yat = get_features_labels(df, needed_pmus)
    print("Features shape:", X_s.shape)
    print("Level 1 labels sample:", yl1.head())
    print("Fault Type labels sample:", yft.head()) 