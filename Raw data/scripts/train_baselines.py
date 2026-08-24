# Baseline imputation script for SIGMA study. Implements MICE, missForest, and mean imputation.

import os, pickle, argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

def missforest_impute(X_train, X_test):
    imp_mean = SimpleImputer(strategy='mean')
    X_train_imp = imp_mean.fit_transform(X_train)
    X_test_imp = imp_mean.transform(X_test)
    
    missing_cols = np.where(np.isnan(X_train).sum(axis=0) > 0)[0]
    if len(missing_cols) == 0:
        return X_train_imp, X_test_imp
        
    for _ in range(10):
        X_train_imp_old = X_train_imp.copy()
        for c in missing_cols:
            y_train = X_train_imp[:, c]
            X_train_features = np.delete(X_train_imp, c, axis=1)
            X_test_features = np.delete(X_test_imp, c, axis=1)
            
            if HAS_CUML:
                rf = cuRF(n_estimators=50, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            observed_rows = ~np.isnan(X_train[:, c])
            rf.fit(X_train_features[observed_rows], y_train[observed_rows])
            
            missing_rows_train = np.isnan(X_train[:, c])
            if missing_rows_train.sum() > 0:
                X_train_imp[missing_rows_train, c] = rf.predict(X_train_features[missing_rows_train])
                
            missing_rows_test = np.isnan(X_test[:, c])
            if missing_rows_test.sum() > 0:
                X_test_imp[missing_rows_test, c] = rf.predict(X_test_features[missing_rows_test])
                
        diff = np.max(np.abs(X_train_imp - X_train_imp_old))
        if diff < 1e-3:
            break
            
    return X_train_imp, X_test_imp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    args = parser.parse_args()

    BASE = "/Users/azmannads/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    FOLDS_DIR = f"{PROCESSED_DIR}/folds"
    OUTPUT_DIR = f"{BASE}/results"
    
    out_path = f"{OUTPUT_DIR}/baselines_{args.severity}_{args.condition}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print(f"[BASELINES] Config: severity={args.severity} | condition={args.condition}")

    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    
    with open(f"{FOLDS_DIR}/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)

    X_c = clinical.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    
    genomic_cols = genomic.drop(columns=['PATIENT_ID']).columns.tolist()
    expr_cols = [c for c in genomic_cols if not c.endswith('_CNA')]
    X_expr = genomic[expr_cols].values.astype(np.float32)

    masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
    masks_g_expr = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_genomic.npy")
    
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]
    
    mask_c_features = masks_c[severity_idx]
    age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
    
    mask_g_expr_full = masks_g_expr[severity_idx]
    
    if args.condition == 'joint':
        age = X_c[:, 0]
        tau = np.median(age)
        age_indicator = (age > tau).astype(np.float32)
        
        calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
        
        def get_params(modality, severity):
            row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
            return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
            
        alpha_c, beta_c = get_params('clinical', args.severity)
        alpha_g, beta_g = get_params('genomic', args.severity)
        
        np.random.seed(42)
        X_c_features = X_c[:, 1:]
        prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features) + beta_c)
        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)
        age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
        mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
        
        prob_g = expit(alpha_g * age_indicator[:, None] * np.abs(X_expr) + beta_g)
        mask_g_expr_joint = np.random.binomial(1, prob_g).astype(np.float32)
        mask_g_expr_full = mask_g_expr_joint
        
    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = np.nan
    
    X_expr_masked = X_expr.copy()
    X_expr_masked[mask_g_expr_full == 1] = np.nan

    methods = ['Mean', 'MICE', 'missForest']
    results = {m: {'rmse_c': [], 'mae_c': [], 'rmse_g': [], 'mae_g': []} for m in methods}

    for k, fold_dict in enumerate(folds, 1):
        np.random.seed(42)
        train_idx = fold_dict['train']
        test_idx = fold_dict['test']
        
        X_c_train, X_c_test = X_c_masked[train_idx], X_c_masked[test_idx]
        X_expr_train, X_expr_test = X_expr_masked[train_idx], X_expr_masked[test_idx]
        
        X_c_test_true = X_c[test_idx]
        X_expr_test_true = X_expr[test_idx]
        
        mask_c_test = mask_c_full[test_idx]
        mask_g_expr_test = mask_g_expr_full[test_idx]
        
        for method in methods:
            if method == 'Mean':
                imp_c = SimpleImputer(strategy='mean')
                x_hat_c_train = imp_c.fit_transform(X_c_train)
                x_hat_c = imp_c.transform(X_c_test)
                
                imp_g = SimpleImputer(strategy='mean')
                x_hat_g_train = imp_g.fit_transform(X_expr_train)
                x_hat_g_expr = imp_g.transform(X_expr_test)
                
            elif method == 'MICE':
                imp_c = IterativeImputer(max_iter=10, random_state=42)
                x_hat_c_train = imp_c.fit_transform(X_c_train)
                x_hat_c = imp_c.transform(X_c_test)
                
                imp_g = IterativeImputer(max_iter=10, random_state=42)
                x_hat_g_train = imp_g.fit_transform(X_expr_train)
                x_hat_g_expr = imp_g.transform(X_expr_test)
                
            elif method == 'missForest':
                x_hat_c_train, x_hat_c = missforest_impute(X_c_train, X_c_test)
                x_hat_g_train, x_hat_g_expr = missforest_impute(X_expr_train, X_expr_test)
            
            mask_c_test_features = mask_c_test[:, 1:]
            x_hat_c_features = x_hat_c[:, 1:]
            x_c_test_features = X_c_test_true[:, 1:]
            
            obs_c = mask_c_test_features == 1
            if obs_c.sum() > 0:
                diff_c = x_hat_c_features[obs_c] - x_c_test_features[obs_c]
                rmse_c = np.sqrt(np.mean(diff_c**2))
                mae_c = np.mean(np.abs(diff_c))
            else:
                rmse_c, mae_c = 0.0, 0.0
                
            obs_g = mask_g_expr_test == 1
            if obs_g.sum() > 0:
                diff_g = x_hat_g_expr[obs_g] - X_expr_test_true[obs_g]
                rmse_g = np.sqrt(np.mean(diff_g**2))
                mae_g = np.mean(np.abs(diff_g))
            else:
                rmse_g, mae_g = 0.0, 0.0
                
            print(f"[BASELINES] {method} | Fold {k}/5 | RMSE_c: {rmse_c:.4f} | MAE_c: {mae_c:.4f} | RMSE_g: {rmse_g:.4f} | MAE_g: {mae_g:.4f}")
            
            results[method]['rmse_c'].append(rmse_c)
            results[method]['mae_c'].append(mae_c)
            results[method]['rmse_g'].append(rmse_g)
            results[method]['mae_g'].append(mae_g)
            
            res_dict = {
                'x_hat_c': x_hat_c,
                'x_hat_g': x_hat_g_expr,
                'test_idx': test_idx,
                'rmse_c': rmse_c,
                'mae_c': mae_c,
                'rmse_g': rmse_g,
                'mae_g': mae_g
            }
            with open(f"{out_path}/{method}_fold_{k}_results.pkl", 'wb') as f:
                pickle.dump(res_dict, f)
                
    summary_data = []
    for method in methods:
        mean_rmse_c, std_rmse_c = np.mean(results[method]['rmse_c']), np.std(results[method]['rmse_c'])
        mean_mae_c, std_mae_c = np.mean(results[method]['mae_c']), np.std(results[method]['mae_c'])
        mean_rmse_g, std_rmse_g = np.mean(results[method]['rmse_g']), np.std(results[method]['rmse_g'])
        mean_mae_g, std_mae_g = np.mean(results[method]['mae_g']), np.std(results[method]['mae_g'])
        
        print(f"[BASELINES] Summary | {method} | RMSE_c: {mean_rmse_c:.4f} ± {std_rmse_c:.4f} | MAE_c: {mean_mae_c:.4f} ± {std_mae_c:.4f} | RMSE_g: {mean_rmse_g:.4f} ± {std_rmse_g:.4f} | MAE_g: {mean_mae_g:.4f} ± {std_mae_g:.4f}")
        
        summary_data.append({
            'Method': method,
            'mean_rmse_c': mean_rmse_c, 'std_rmse_c': std_rmse_c,
            'mean_mae_c': mean_mae_c, 'std_mae_c': std_mae_c,
            'mean_rmse_g': mean_rmse_g, 'std_rmse_g': std_rmse_g,
            'mean_mae_g': mean_mae_g, 'std_mae_g': std_mae_g
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
