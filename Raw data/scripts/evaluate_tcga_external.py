import os, pickle, json
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sigma_model import SIGMA, device
import warnings
warnings.filterwarnings('ignore')

def main():
    BASE = "/home/azman/intrinsic_eval_run/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    RESULTS_DIR = f"{BASE}/results_final"
    OUTPUT_DIR = f"{BASE}/evaluation/tcga_external_validation"
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[EXTERNAL] Loading TCGA data...")
    tcga_clin = pd.read_csv(f"{PROCESSED_DIR}/tcga_clinical_external.csv")
    tcga_gen = pd.read_csv(f"{PROCESSED_DIR}/tcga_genomic_external.csv")
    tcga_surv = pd.read_csv(f"{PROCESSED_DIR}/tcga_survival_external.csv")
    tcga_sub = pd.read_csv(f"{PROCESSED_DIR}/tcga_subtype.csv")
    
    X_c_ts = tcga_clin.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g_ts = tcga_gen.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    T_ts = tcga_surv['T'].values.astype(np.float32)
    E_ts = tcga_surv['E'].values.astype(np.float32)
    
    # TCGA subtypes
    tcga_subtype_map = dict(zip(tcga_sub['PATIENT_ID'], tcga_sub['SUBTYPE']))
    subtype_remap = {'BRCA_LumA': 'LumA', 'BRCA_LumB': 'LumB', 'BRCA_Basal': 'Basal', 'BRCA_Her2': 'Her2', 'BRCA_Normal': 'Normal'}
    tcga_subtype_map = {k: subtype_remap.get(v, v) for k, v in tcga_subtype_map.items()}
    valid_subtypes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
    patient_ids_ts = tcga_clin['PATIENT_ID'].values
    y_ts_subtypes = np.array([str(tcga_subtype_map.get(pid, 'NC')).replace('HER2', 'Her2') for pid in patient_ids_ts])
    test_sub_mask = np.isin(y_ts_subtypes, valid_subtypes)
    
    # METABRIC for downstream training
    mb_clin = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    mb_surv = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")
    T_tr_all = mb_surv['T'].values.astype(np.float32)
    E_tr_all = mb_surv['E'].values.astype(np.float32)
    
    mb_clin_raw = pd.read_csv(f"{BASE}/original_datasets/brca_metabric/data_clinical_patient.txt", sep='\t', comment='#')
    mb_subtype_map = dict(zip(mb_clin_raw['PATIENT_ID'], mb_clin_raw['CLAUDIN_SUBTYPE']))
    y_tr_subtypes_all = np.array([mb_subtype_map.get(pid, 'NC') for pid in mb_clin['PATIENT_ID'].values])
    
    mb_gen = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    X_c_mb = mb_clin.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g_mb = mb_gen.drop(columns=['PATIENT_ID']).values.astype(np.float32)

    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", "rb") as f:
        folds = pickle.load(f)

    # Load pre-computed METABRIC MNAR masks for severe independent condition (severity index 2)
    MNAR_DIR = f"{BASE}/mnar_simulation"
    masks_c_all = np.load(f"{MNAR_DIR}/mnar_masks_metabric_clinical.npy")  # shape (3, N, 8)
    masks_g_all = np.load(f"{MNAR_DIR}/mnar_masks_metabric_genomic.npy")   # shape (3, N, 49)
    sev_idx = 2  # severe
    mask_c_features_mb = masks_c_all[sev_idx].astype(np.float32)  # (N, 8)
    mask_c_full = np.concatenate([np.zeros((X_c_mb.shape[0], 1), dtype=np.float32), mask_c_features_mb], axis=1)  # (N, 9)
    mask_g_full = np.zeros((X_g_mb.shape[0], X_g_mb.shape[1]), dtype=np.float32)
    mask_g_full[:, :49] = masks_g_all[sev_idx].astype(np.float32)
    X_c_masked = X_c_mb.copy()
    X_c_masked[mask_c_full == 1] = 0.0
    X_g_masked = X_g_mb.copy()
    X_g_masked[mask_g_full == 1] = 0.0
        
    print("[EXTERNAL] Imputing background real-world missingness using MICE...")
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    mice_c = IterativeImputer(max_iter=10, random_state=42)
    X_c_ts = mice_c.fit_transform(X_c_ts).astype(np.float32)
    
    mice_g = IterativeImputer(max_iter=10, random_state=42)
    X_g_ts = mice_g.fit_transform(X_g_ts).astype(np.float32)
        
    # Simulate MNAR Severe Independent on TCGA
    calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
    def get_params(modality, severity):
        row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
        return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
    
    alpha_c, beta_c = get_params('clinical', 'severe')
    alpha_g, beta_g = get_params('genomic', 'severe')
    
    np.random.seed(42)
    # Age indicator for TCGA
    age_ts = X_c_ts[:, 0]
    tau_ts = np.median(age_ts)
    age_indicator_ts = (age_ts > tau_ts).astype(np.float32)
    
    # Clinical mask
    X_c_features_ts = X_c_ts[:, 1:]
    prob_c_ts = expit(alpha_c * age_indicator_ts[:, None] * np.abs(X_c_features_ts) + beta_c)
    mask_c_features_ts = np.random.binomial(1, prob_c_ts).astype(np.float32)
    mask_c_full_ts = np.concatenate([np.zeros((X_c_ts.shape[0], 1), dtype=np.float32), mask_c_features_ts], axis=1)
    
    # Genomic mask
    X_expr_ts = X_g_ts[:, :49]
    prob_g_ts = expit(alpha_g * age_indicator_ts[:, None] * np.abs(X_expr_ts) + beta_g)
    mask_g_expr_ts = np.random.binomial(1, prob_g_ts).astype(np.float32)
    mask_g_full_ts = np.zeros_like(X_g_ts)
    mask_g_full_ts[:, :49] = mask_g_expr_ts
    
    X_c_masked_ts = X_c_ts.copy()
    X_c_masked_ts[mask_c_full_ts == 1] = 0.0
    
    X_g_masked_ts = X_g_ts.copy()
    X_g_masked_ts[mask_g_full_ts == 1] = 0.0
    
    binary_c_cols = [1, 2, 3, 4, 5, 6]
    
    def postprocess_clinical(x):
        x_post = x.copy()
        x_post[:, binary_c_cols] = expit(x_post[:, binary_c_cols])
        return x_post
        
    summary_results = []
    
    methods = ['EF', 'CMA', 'PoE']
    for method in methods:
        method_dir = f"{RESULTS_DIR}/{method}_severe_independent"
        if not os.path.exists(method_dir):
            continue
            
        with open(f"{method_dir}/best_params.json") as f:
            best_params = json.load(f)
        d = best_params['d']
        
        metrics = {'RMSE_c': [], 'MAE_c': [], 'RMSE_g': [], 'MAE_g': [], 'C2G_RMSE': [], 'RSF': [], 'PAM50': []}
        
        for k, fold_dict in enumerate(folds, 1):
            model_path = f"{method_dir}/fold_{k}_model.pt"
            res_path = f"{method_dir}/fold_{k}_results.pkl"
            
            if not os.path.exists(model_path) or not os.path.exists(res_path):
                continue
                
            model = SIGMA(X_c_ts.shape[1], X_g_ts.shape[1], d, method).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            x_c_t = torch.FloatTensor(X_c_masked_ts).to(device)
            m_c_t = torch.FloatTensor(mask_c_full_ts).to(device)
            x_g_t = torch.FloatTensor(X_g_masked_ts).to(device)
            m_g_t = torch.FloatTensor(mask_g_full_ts).to(device)
            
            with torch.no_grad():
                if method == 'EF':
                    in_c = torch.cat([x_c_t, m_c_t], dim=1)
                    in_g = torch.cat([x_g_t, m_g_t], dim=1)
                    mu_c, log_var_c = model.enc_c(in_c)
                    mu_g, log_var_g = model.enc_g(in_g)
                    z_c = mu_c
                    z_g = mu_g
                    z_shared = model.fusion_module(z_c, z_g)
                    x_hat_c_raw = model.dec_c(z_shared).cpu().numpy()
                    x_hat_g_raw = model.dec_g(z_shared).cpu().numpy()
                    # C2G: use clinical encoder only -> genomic decoder
                    x_hat_g_c2g_raw = model.dec_g(z_c).cpu().numpy()
                elif method == 'CMA':
                    in_c = torch.cat([x_c_t, m_c_t], dim=1)
                    in_g = torch.cat([x_g_t, m_g_t], dim=1)
                    mu_c, log_var_c = model.enc_c(in_c)
                    mu_g, log_var_g = model.enc_g(in_g)
                    z_c = mu_c
                    z_g = mu_g
                    z_shared = model.fusion_module(z_c, z_g)
                    x_hat_c_raw = model.dec_c(z_shared).cpu().numpy()
                    x_hat_g_raw = model.dec_g(z_shared).cpu().numpy()
                    # C2G: use clinical encoder only -> genomic decoder
                    x_hat_g_c2g_raw = model.dec_g(z_c).cpu().numpy()
                elif method == 'PoE':
                    in_c = torch.cat([x_c_t, m_c_t], dim=1)
                    in_g = torch.cat([x_g_t, m_g_t], dim=1)
                    mu_c, log_var_c = model.enc_c(in_c)
                    var_c = torch.exp(log_var_c)
                    mu_g, log_var_g = model.enc_g(in_g)
                    var_g = torch.exp(log_var_g)
                    
                    # Include standard normal prior (precision = 1)
                    var_joint = 1.0 / (1.0 + 1.0/var_c + 1.0/var_g)
                    mu_joint = var_joint * (mu_c/var_c + mu_g/var_g)
                    
                    x_hat_c_raw = model.dec_c(mu_joint).cpu().numpy()
                    x_hat_g_raw = model.dec_g(mu_joint).cpu().numpy()
                    
                    x_hat_g_c2g_raw = model.dec_g(mu_c).cpu().numpy()
            
            x_hat_c_ts = postprocess_clinical(x_hat_c_raw)
            x_hat_g_ts = x_hat_g_raw
            x_hat_g_c2g = x_hat_g_c2g_raw
            
            # Imputation Metrics
            mask_c_bool = (mask_c_full_ts == 1)
            rmse_c = np.sqrt(np.mean((X_c_ts[mask_c_bool] - x_hat_c_ts[mask_c_bool])**2))
            mae_c = np.mean(np.abs(X_c_ts[mask_c_bool] - x_hat_c_ts[mask_c_bool]))
            
            mask_g_bool = (mask_g_full_ts[:, :49] == 1)
            rmse_g = np.sqrt(np.mean((X_g_ts[:, :49][mask_g_bool] - x_hat_g_ts[:, :49][mask_g_bool])**2))
            mae_g = np.mean(np.abs(X_g_ts[:, :49][mask_g_bool] - x_hat_g_ts[:, :49][mask_g_bool]))
            
            c2g_rmse = np.sqrt(np.mean((X_g_ts[:, :49][mask_g_bool] - x_hat_g_c2g[:, :49][mask_g_bool])**2))
            
            # Downstream Tasks (Train on METABRIC, Test on TCGA)
            # Run fresh forward pass on METABRIC training data
            train_idx = np.concatenate([fold_dict['train'], fold_dict['val']])
            x_c_tr = torch.FloatTensor(X_c_masked[train_idx]).to(device)
            m_c_tr = torch.FloatTensor(mask_c_full[train_idx]).to(device)
            x_g_tr = torch.FloatTensor(X_g_masked[train_idx]).to(device)
            m_g_tr = torch.FloatTensor(mask_g_full[train_idx]).to(device)
            with torch.no_grad():
                x_hat_c_mb_raw, x_hat_g_mb_raw, _, _, _, _, _, _ = model(x_c_tr, m_c_tr, x_g_tr, m_g_tr)
            X_c_tr_mb = postprocess_clinical(x_hat_c_mb_raw.cpu().numpy())
            X_g_tr_mb = x_hat_g_mb_raw.cpu().numpy()
            
            # RSF
            X_tr_rsf = np.concatenate([X_c_tr_mb, X_g_tr_mb[:, :49]], axis=1)
            X_ts_rsf = np.concatenate([x_hat_c_ts, x_hat_g_ts[:, :49]], axis=1)
            y_tr_rsf = Surv.from_arrays(E_tr_all[train_idx].astype(bool), T_tr_all[train_idx])
            
            # Impute any remaining NaNs in TCGA features that were originally NaN and not masked
            # Because x_hat only replaces masked items if we don't overwrite, but our x_hat replaces everything.
            # Wait, x_hat is the direct output, so it has NO NaNs!
            rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
            rsf.fit(X_tr_rsf, y_tr_rsf)
            risk = rsf.predict(X_ts_rsf)
            c_idx = concordance_index_censored(E_ts.astype(bool), T_ts, risk)[0]
            
            # PAM50
            X_tr_pam = X_g_tr_mb[:, :49]
            y_tr_pam = y_tr_subtypes_all[train_idx]
            valid_train = np.isin(y_tr_pam, valid_subtypes)
            X_tr_pam = X_tr_pam[valid_train]
            y_tr_pam = y_tr_pam[valid_train]
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_tr_pam, y_tr_pam)
            
            X_ts_pam = x_hat_g_ts[:, :49][test_sub_mask]
            y_ts_pam = y_ts_subtypes[test_sub_mask]
            y_pred = rf.predict(X_ts_pam)
            acc = accuracy_score(y_ts_pam, y_pred)
            
            metrics['RMSE_c'].append(rmse_c)
            metrics['MAE_c'].append(mae_c)
            metrics['RMSE_g'].append(rmse_g)
            metrics['MAE_g'].append(mae_g)
            metrics['C2G_RMSE'].append(c2g_rmse)
            metrics['RSF'].append(c_idx)
            metrics['PAM50'].append(acc)
            
        if len(metrics['RMSE_c']) > 0:
            m = {k: np.mean(v) for k, v in metrics.items()}
            s = {k: np.std(v) for k, v in metrics.items()}
            
            summary_results.append({
                'method': method,
                'RMSE_c_mean': m['RMSE_c'], 'RMSE_c_std': s['RMSE_c'],
                'RMSE_g_mean': m['RMSE_g'], 'RMSE_g_std': s['RMSE_g'],
                'C2G_RMSE_mean': m['C2G_RMSE'], 'C2G_RMSE_std': s['C2G_RMSE'],
                'RSF_mean': m['RSF'], 'RSF_std': s['RSF'],
                'PAM50_mean': m['PAM50'], 'PAM50_std': s['PAM50']
            })
            
            print(f"[TCGA-EXT] {method} | RMSE_c: {m['RMSE_c']:.4f} \u00b1 {s['RMSE_c']:.4f} | "
                  f"RMSE_g: {m['RMSE_g']:.4f} \u00b1 {s['RMSE_g']:.4f} | "
                  f"C2G: {m['C2G_RMSE']:.4f} \u00b1 {s['C2G_RMSE']:.4f} | "
                  f"C-index RSF: {m['RSF']:.4f} \u00b1 {s['RSF']:.4f} | "
                  f"PAM50 Acc: {m['PAM50']:.4f} \u00b1 {s['PAM50']:.4f}")
                  
    if len(summary_results) > 0:
        pd.DataFrame(summary_results).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
        print(f"\nSaved summary to {OUTPUT_DIR}/summary.csv")

if __name__ == '__main__':
    main()
