# Direct survival evaluation script for SIGMA study. Evaluates log-hazard eta from Cox head directly.
import os
import pickle
import numpy as np
import pandas as pd
import argparse
from sksurv.metrics import concordance_index_censored

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    args = parser.parse_args()

    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_DIR = f"{BASE}/processed"
    RESULTS_DIR = f"{BASE}/results_final"
    OUTPUT_DIR = f"{BASE}/evaluation"

    np.random.seed(42)

    print(f"[DIRECT_SURVIVAL] Config: severity={args.severity} | condition={args.condition}")

    survival = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")
    T_all = survival['T'].values.astype(np.float32)
    E_all = survival['E'].values.astype(np.float32)

    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)

    methods = ['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
    
    out_path = f"{OUTPUT_DIR}/direct_survival_{args.severity}_{args.condition}"
    os.makedirs(out_path, exist_ok=True)

    summary_data = []

    for method in methods:
        c_indices = []
        
        for k, fold_dict in enumerate(folds):
            test_idx = fold_dict['test']
            T_test = T_all[test_idx]
            E_test = E_all[test_idx]

            res_path = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}/fold_{k+1}_results.pkl"
            
            if not os.path.exists(res_path):
                print(f"Warning: Result file missing for {method} fold {k+1}. Skipping.")
                continue

            with open(res_path, 'rb') as f:
                res = pickle.load(f)
            
            if 'eta' not in res:
                continue
            eta_test = res['eta']
            eta_test = np.nan_to_num(eta_test, nan=0.0, posinf=1e6, neginf=-1e6)
            eta_test = eta_test.flatten()
            
            c_idx = concordance_index_censored(E_test.astype(bool), T_test, eta_test)[0]
            c_indices.append(c_idx)
            
            print(f"[DIRECT_SURVIVAL] {method} | Fold {k+1}/5 | C-index: {c_idx:.4f}")

        if len(c_indices) > 0:
            mean_c = np.mean(c_indices)
            std_c = np.std(c_indices)
            print(f"[DIRECT_SURVIVAL] Summary | {method} | C-index: {mean_c:.4f} \u00b1 {std_c:.4f}")
            
            pd.DataFrame({
                'Fold': range(1, len(c_indices) + 1),
                'C-index': c_indices
            }).to_csv(f"{out_path}/{method}_direct_survival.csv", index=False)
            
            summary_data.append({
                'Method': method,
                'Mean_C-index': mean_c,
                'Std_C-index': std_c
            })
            
    if summary_data:
        pd.DataFrame(summary_data).to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
