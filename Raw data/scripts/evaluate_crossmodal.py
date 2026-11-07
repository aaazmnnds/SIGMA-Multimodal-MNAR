# Cross-modal imputation evaluation script for SIGMA study.
import os, pickle, argparse, json
from math import sqrt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sigma_model import SIGMA, device
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    parser.add_argument('--mechanism', type=str, choices=['threshold', 'selfmask'], default='threshold')
    args = parser.parse_args()
    
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_DIR = f"{BASE}/processed"
    if args.mechanism == 'selfmask':
        RESULTS_DIR = f"{BASE}/results_selfmask"
        OUTPUT_DIR = f"{BASE}/evaluation_selfmask"
    else:
        RESULTS_DIR = f"{BASE}/results_final"
        OUTPUT_DIR = f"{BASE}/evaluation"
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    
    out_path = f"{OUTPUT_DIR}/crossmodal_{args.severity}_{args.condition}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    print(f"[CROSSMODAL] Config: severity={args.severity} | condition={args.condition} | mechanism={args.mechanism}")
    
    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    
    X_c = clinical.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g = genomic.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_expr = X_g[:, :49]
    
    if args.mechanism == 'selfmask':
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_selfmask_clinical.npy")
        masks_g_expr = np.load(f"{SIMULATION_DIR}/mnar_masks_selfmask_genomic.npy")
    else:
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
        masks_g_expr = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_genomic.npy")
    
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]
    
    mask_c_features = masks_c[severity_idx]
    age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
    
    mask_g_expr_full = masks_g_expr[severity_idx]
    
    if args.condition == 'joint':
        if args.mechanism == 'selfmask':
            calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results_selfmask.csv")
        else:
            calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
        
        def get_params(modality, severity):
            row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
            return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
            
        alpha_c, beta_c = get_params('clinical', args.severity)
        alpha_g, beta_g = get_params('genomic', args.severity)
        
        np.random.seed(42)
        X_c_features_arr = X_c[:, 1:]
        if args.mechanism == 'selfmask':
            prob_c = expit(alpha_c * np.abs(X_c_features_arr) + beta_c)
            prob_g = expit(alpha_g * np.abs(X_expr) + beta_g)
        else:
            age = X_c[:, 0]
            tau = np.median(age)
            age_indicator = (age > tau).astype(np.float32)
            prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features_arr) + beta_c)
            prob_g = expit(alpha_g * age_indicator[:, None] * np.abs(X_expr) + beta_g)

        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)
        age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
        mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
        
        mask_g_expr_joint = np.random.binomial(1, prob_g).astype(np.float32)
        mask_g_expr_full = mask_g_expr_joint

    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = 0.0
    
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr_full == 1] = 0.0
    
    mask_g_full = np.zeros_like(X_g)
    mask_g_full[:, :49] = mask_g_expr_full
    
    p_c = X_c.shape[1]
    p_g = X_g.shape[1]
    
    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)
        
    methods = ['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
    all_summary = []
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    for method in methods:
        if method not in ('EF', 'CMA', 'PoE'):
            continue
        method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"
        if not os.path.exists(method_dir):
            print(f"Warning: {method_dir} does not exist. Skipping {method}.")
            continue
            
        method_results = []
        for k, fold_dict in enumerate(folds, 1):
            model_path = f"{method_dir}/fold_{k}_model.pt"
            res_path = f"{method_dir}/fold_{k}_results.pkl"
            params_path = f"{method_dir}/fold_{k}_best_params.json"
            
            if not os.path.exists(model_path) or not os.path.exists(res_path):
                continue
                
            import json
            best_params_path = f"{method_dir}/best_params.json"
            with open(best_params_path) as f:
                best_params = json.load(f)
            d = best_params['d']
                    
            model = SIGMA(p_c, p_g, d, method).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            with open(res_path, 'rb') as f:
                res = pickle.load(f)
            test_idx = res['test_idx']
            
            x_c_test = torch.FloatTensor(X_c_masked[test_idx]).to(device)
            mask_c_test = torch.FloatTensor(mask_c_full[test_idx]).to(device)
            x_g_test = torch.FloatTensor(X_g_masked[test_idx]).to(device)
            mask_g_test = torch.FloatTensor(mask_g_full[test_idx]).to(device)
            
            with torch.no_grad():
                # Clinical to genomic
                in_c = torch.cat([x_c_test, mask_c_test], dim=1)
                mu_c, log_var_c = model.enc_c(in_c)
                z_c = mu_c
                x_hat_g_from_c = model.dec_g(z_c).cpu().numpy()
                
                # Genomic to clinical
                in_g = torch.cat([x_g_test, mask_g_test], dim=1)
                mu_g, log_var_g = model.enc_g(in_g)
                z_g = mu_g
                x_hat_c_from_g = model.dec_c(z_g).cpu().numpy()
                
            x_g_true = X_g[test_idx]
            x_c_true = X_c[test_idx]
            
            mask_g_test_np = mask_g_full[test_idx]
            mask_c_test_np = mask_c_full[test_idx]
            
            # C2G metrics
            g_eval_mask = (mask_g_test_np[:, :49] == 1)
            if np.sum(g_eval_mask) > 0:
                c2g_mse = np.mean((x_g_true[:, :49][g_eval_mask] - x_hat_g_from_c[:, :49][g_eval_mask])**2)
                c2g_rmse = np.sqrt(c2g_mse)
                c2g_mae = np.mean(np.abs(x_g_true[:, :49][g_eval_mask] - x_hat_g_from_c[:, :49][g_eval_mask]))
            else:
                c2g_rmse = np.nan
                c2g_mae = np.nan
                
            # G2C metrics
            c_eval_mask = (mask_c_test_np[:, 1:] == 1)
            if np.sum(c_eval_mask) > 0:
                g2c_mse = np.mean((x_c_true[:, 1:][c_eval_mask] - x_hat_c_from_g[:, 1:][c_eval_mask])**2)
                g2c_rmse = np.sqrt(g2c_mse)
                g2c_mae = np.mean(np.abs(x_c_true[:, 1:][c_eval_mask] - x_hat_c_from_g[:, 1:][c_eval_mask]))
            else:
                g2c_rmse = np.nan
                g2c_mae = np.nan
                
            # PoE does not support unimodal cross-modal reconstruction
            if method == 'PoE':
                g2c_rmse = np.nan
                g2c_mae = np.nan

            print(f"[CROSSMODAL] {method} | Fold {k}/5 | C2G_RMSE: {c2g_rmse:.4f}, G2C_RMSE: {g2c_rmse:.4f}")
            
            res_fold = {
                'method': method,
                'fold': k,
                'C2G_RMSE': c2g_rmse,
                'C2G_MAE': c2g_mae,
                'G2C_RMSE': g2c_rmse,
                'G2C_MAE': g2c_mae
            }
            method_results.append(res_fold)
            
            with open(f"{out_path}/{method}_fold_{k}_crossmodal.pkl", 'wb') as f:
                pickle.dump(res_fold, f)
                
        if len(method_results) > 0:
            df_m = pd.DataFrame(method_results)
            df_m.to_csv(f"{out_path}/{method}_crossmodal.csv", index=False)
            
            m_c2g_r, s_c2g_r = df_m['C2G_RMSE'].mean(), df_m['C2G_RMSE'].std()
            m_g2c_r, s_g2c_r = df_m['G2C_RMSE'].mean(), df_m['G2C_RMSE'].std()
            
            print(f"[CROSSMODAL] Summary | {method} | C2G_RMSE: {m_c2g_r:.4f}±{s_c2g_r:.4f} | G2C_RMSE: {m_g2c_r:.4f}±{s_g2c_r:.4f}")
            
            all_summary.append({
                'method': method,
                'C2G_RMSE_mean': m_c2g_r,
                'C2G_RMSE_std': s_c2g_r,
                'C2G_RMSE_ci': 2.776 * s_c2g_r / sqrt(5),
                'C2G_MAE_mean': df_m['C2G_MAE'].mean(),
                'C2G_MAE_std': df_m['C2G_MAE'].std(),
                'C2G_MAE_ci': 2.776 * df_m['C2G_MAE'].std() / sqrt(5),
                'G2C_RMSE_mean': m_g2c_r,
                'G2C_RMSE_std': s_g2c_r,
                'G2C_RMSE_ci': 2.776 * s_g2c_r / sqrt(5),
                'G2C_MAE_mean': df_m['G2C_MAE'].mean(),
                'G2C_MAE_std': df_m['G2C_MAE'].std(),
                'G2C_MAE_ci': 2.776 * df_m['G2C_MAE'].std() / sqrt(5)
            })
            
    if len(all_summary) > 0:
        pd.DataFrame(all_summary).to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
