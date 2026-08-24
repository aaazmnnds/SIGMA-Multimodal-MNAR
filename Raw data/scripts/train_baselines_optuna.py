# Baseline Optuna tuning and training script for SIGMA study. MICE and missForest.
import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import optuna
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy.core.multiarray':
            module = 'numpy'
        return super().find_class(module, name)

def compute_rmse(imputed, truth, mask):
    obs = mask == 1
    diff = imputed[obs] - truth[obs]
    return np.sqrt(np.mean(diff**2)) if obs.sum() > 0 else 0.0

def compute_mae(imputed, truth, mask):
    obs = mask == 1
    diff = imputed[obs] - truth[obs]
    return np.mean(np.abs(diff)) if obs.sum() > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['MICE', 'missForest'])
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    parser.add_argument('--n_trials', type=int, default=50)
    args = parser.parse_args()

    np.random.seed(42)
    
    BASE = "/home/azman/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    RESULTS_DIR = f"{BASE}/results_final"
    
    out_dir = f"{RESULTS_DIR}/{args.method}_{args.severity}_{args.condition}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[BASELINES-OPTUNA] Method: {args.method} | severity={args.severity} | condition={args.condition}")
    
    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    
    X_c = clinical.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g = genomic.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    
    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = NumpyUnpickler(f).load()
        
    from scipy.special import expit
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]

    if args.condition == 'independent':
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
        masks_g = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_genomic.npy")
        mask_c_features = masks_c[severity_idx].astype(np.float32)
        mask_g_expr = masks_g[severity_idx].astype(np.float32)
    else:  # joint
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
        X_expr = X_g[:, :49]
        prob_g = expit(alpha_g * age_indicator[:, None] * np.abs(X_expr) + beta_g)
        mask_g_expr = np.random.binomial(1, prob_g).astype(np.float32)

    age_mask_col = np.zeros((X_c.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
    mask_g_full = np.zeros((X_g.shape[0], X_g.shape[1]), dtype=np.float32)
    mask_g_full[:, :49] = mask_g_expr
    
    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = np.nan
    
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr == 1] = np.nan
    
    train_idx_0 = folds[0]['train']
    val_idx_0 = folds[0]['val']
    
    print(f"[BASELINES-OPTUNA] Running {args.n_trials} Optuna trials on fold 0...")
    
    def objective(trial):
        if args.method == 'MICE':
            max_iter = trial.suggest_int('max_iter', 5, 20)
            imputer = IterativeImputer(max_iter=max_iter, n_nearest_features=10, random_state=42)
            X_combined_train = np.concatenate([X_c_masked[train_idx_0], X_g_masked[train_idx_0]], axis=1)
            X_combined_val = np.concatenate([X_c_masked[val_idx_0], X_g_masked[val_idx_0]], axis=1)
            imputer.fit(X_combined_train)
            X_combined_imp = np.nan_to_num(imputer.transform(X_combined_val), nan=0.0, posinf=0.0, neginf=0.0)
            X_combined_imp = np.clip(X_combined_imp, -10.0, 10.0)
            X_c_imp = X_combined_imp[:, :X_c.shape[1]]
            X_g_imp = X_combined_imp[:, X_c.shape[1]:]
            
        elif args.method == 'missForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 100)
            max_depth = trial.suggest_categorical('max_depth', [5, 10, 20])
            max_iter = trial.suggest_int('max_iter', 5, 10)
            if HAS_CUML:
                rf = cuRF(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            imputer = IterativeImputer(estimator=rf, max_iter=max_iter, random_state=42)
            X_combined_train = np.concatenate([X_c_masked[train_idx_0], X_g_masked[train_idx_0]], axis=1)
            X_combined_val = np.concatenate([X_c_masked[val_idx_0], X_g_masked[val_idx_0]], axis=1)
            imputer.fit(X_combined_train)
            X_combined_imp = np.nan_to_num(imputer.transform(X_combined_val), nan=0.0, posinf=0.0, neginf=0.0)
            X_combined_imp = np.clip(X_combined_imp, -10.0, 10.0)
            X_c_imp = X_combined_imp[:, :X_c.shape[1]]
            X_g_imp = X_combined_imp[:, X_c.shape[1]:]
            
        rmse_c = compute_rmse(X_c_imp, X_c[val_idx_0], mask_c_full[val_idx_0])
        rmse_g = compute_rmse(X_g_imp[:, :49], X_g[val_idx_0][:, :49], mask_g_expr[val_idx_0])
        
        val_obj = rmse_c + rmse_g
        return val_obj

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=args.n_trials)
    
    best_params = study.best_params
    best_loss = study.best_value
    
    params_str = " | ".join([f"{k}={v}" for k, v in best_params.items()])
    print(f"[BASELINES-OPTUNA] Best params: {params_str} | val_rmse={best_loss:.4f}")
    
    with open(f"{out_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print("[BASELINES-OPTUNA] Retraining on all 5 folds...")
    
    results_list = []
    
    for k, fold_dict in enumerate(folds, 1):
        train_idx = fold_dict['train']
        test_idx = fold_dict['test']
        
        if args.method == 'MICE':
            imputer = IterativeImputer(max_iter=best_params['max_iter'], n_nearest_features=10, random_state=42)
        elif args.method == 'missForest':
            if HAS_CUML:
                rf = cuRF(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42, n_jobs=-1)
            imputer = IterativeImputer(estimator=rf, max_iter=best_params['max_iter'], random_state=42)
            
        X_combined_train = np.concatenate([X_c_masked[train_idx], X_g_masked[train_idx]], axis=1)
        X_combined_test = np.concatenate([X_c_masked[test_idx], X_g_masked[test_idx]], axis=1)
        imputer.fit(X_combined_train)
        
        X_combined_imp = np.nan_to_num(imputer.transform(X_combined_test), nan=0.0, posinf=0.0, neginf=0.0)
        X_combined_imp = np.clip(X_combined_imp, -10.0, 10.0)
        
        X_c_imp = X_combined_imp[:, :X_c.shape[1]]
        X_g_imp = X_combined_imp[:, X_c.shape[1]:]
        
        rmse_c = compute_rmse(X_c_imp, X_c[test_idx], mask_c_full[test_idx])
        mae_c = compute_mae(X_c_imp, X_c[test_idx], mask_c_full[test_idx])
        
        rmse_g = compute_rmse(X_g_imp[:, :49], X_g[test_idx][:, :49], mask_g_expr[test_idx])
        mae_g = compute_mae(X_g_imp[:, :49], X_g[test_idx][:, :49], mask_g_expr[test_idx])
        
        print(f"[BASELINES-OPTUNA] Fold {k}/5 | RMSE_c: {rmse_c:.4f} | MAE_c: {mae_c:.4f} | RMSE_g: {rmse_g:.4f} | MAE_g: {mae_g:.4f}")
        
        results_list.append({'Fold': k, 'RMSE_c': rmse_c, 'MAE_c': mae_c, 'RMSE_g': rmse_g, 'MAE_g': mae_g})
        
        fold_res = {
            'test_idx': test_idx,
            'x_hat_c': X_c_imp,
            'x_hat_g': X_g_imp,
            'rmse_c': rmse_c,
            'rmse_g': rmse_g,
            'mae_c': mae_c,
            'mae_g': mae_g
        }
        with open(f"{out_dir}/fold_{k}_results.pkl", 'wb') as f:
            pickle.dump(fold_res, f)
            
    res_df = pd.DataFrame(results_list)
    mean_rmse_c = res_df['RMSE_c'].mean()
    std_rmse_c = res_df['RMSE_c'].std()
    mean_rmse_g = res_df['RMSE_g'].mean()
    std_rmse_g = res_df['RMSE_g'].std()
    
    print(f"[BASELINES-OPTUNA] Summary | RMSE_c: {mean_rmse_c:.4f} ± {std_rmse_c:.4f} | RMSE_g: {mean_rmse_g:.4f} ± {std_rmse_g:.4f}")
    res_df.to_csv(f"{out_dir}/summary.csv", index=False)

if __name__ == '__main__':
    main()
