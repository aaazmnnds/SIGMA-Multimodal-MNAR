# Survival evaluation script for SIGMA study.
import os, pickle, argparse
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cox_partial_likelihood(eta, t, e):
    idx = torch.argsort(t, descending=True)
    eta = eta[idx]
    e = e[idx]
    log_risk = torch.logcumsumexp(eta, dim=0)
    loss = -torch.sum((eta - log_risk) * e) / (torch.sum(e) + 1e-8)
    return loss

class DeepSurv(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class DeepHit(nn.Module):
    def __init__(self, in_features, num_bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, num_bins),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

def train_deepsurv(X_tr, T_tr, E_tr, X_ts, epochs=100, lr=1e-3):
    model = DeepSurv(X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    X = torch.FloatTensor(X_tr).to(device)
    T = torch.FloatTensor(T_tr).to(device)
    E = torch.FloatTensor(E_tr).to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        eta = model(X).squeeze()
        loss = cox_partial_likelihood(eta, T, E)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        risk = model(torch.FloatTensor(X_ts).to(device)).cpu().numpy().squeeze()
    return risk

def train_deephit(X_tr, T_tr, E_tr, X_ts, epochs=100, lr=1e-3, num_bins=10):
    max_t = T_tr.max() + 1e-5
    bins = np.linspace(0, max_t, num_bins + 1)
    T_bin_tr = np.digitize(T_tr, bins) - 1
    T_bin_tr = np.clip(T_bin_tr, 0, num_bins - 1)
    
    model = DeepHit(X_tr.shape[1], num_bins).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.FloatTensor(X_tr).to(device)
    T_b = torch.LongTensor(T_bin_tr).to(device)
    E = torch.FloatTensor(E_tr).to(device)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        pmf = model(X)
        nll_uncensored = -torch.log(pmf.gather(1, T_b.unsqueeze(1)).squeeze() + 1e-8)
        mask = torch.arange(num_bins).unsqueeze(0).to(device) >= T_b.unsqueeze(1)
        cdf_after = torch.sum(pmf * mask, dim=1)
        nll_censored = -torch.log(cdf_after + 1e-8)
        loss = torch.mean(E * nll_uncensored + (1 - E) * nll_censored)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        pmf = model(torch.FloatTensor(X_ts).to(device)).cpu().numpy()
    
    expected_time = np.sum(pmf * np.arange(num_bins), axis=1)
    risk = -expected_time
    return risk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    parser.add_argument('--mechanism', type=str, choices=['threshold', 'selfmask'], default='threshold')
    parser.add_argument('--task', type=str, choices=['survival', 'imputation'], default='survival')
    args = parser.parse_args()
    
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_DIR = f"{BASE}/processed"
    if args.mechanism == 'selfmask':
        RESULTS_DIR = f"{BASE}/results_selfmask"
        OUTPUT_DIR = f"{BASE}/evaluation_selfmask"
    else:
        RESULTS_DIR = f"{BASE}/results_final"
        OUTPUT_DIR = f"{BASE}/evaluation"
    
    out_path = f"{OUTPUT_DIR}/survival_{args.severity}_{args.condition}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    print(f"[SURVIVAL] Config: severity={args.severity} | condition={args.condition} | mechanism={args.mechanism}")
    
    if args.task == 'imputation':
        import pickle, json, csv
        methods = ['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary_rows = []
        for method in methods:
            method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"
            if not os.path.exists(method_dir):
                print(f"[IMPUTATION] Warning: {method_dir} not found, skipping.")
                continue
            rmse_c_vals, rmse_g_vals, mae_c_vals, mae_g_vals = [], [], [], []
            for k in range(1, 6):
                res_path = f"{method_dir}/fold_{k}_results.pkl"
                if not os.path.exists(res_path):
                    continue
                with open(res_path, 'rb') as f:
                    res = pickle.load(f)
                clinical_only = {'MNARSVAE', 'StandardVAE'}
                rmse_c_vals.append(res['rmse_c'])
                if 'rmse_g' in res and method not in clinical_only:
                    rmse_g_vals.append(res['rmse_g'])
                mae_c_vals.append(res['mae_c'])
                if 'mae_g' in res and method not in clinical_only:
                    mae_g_vals.append(res['mae_g'])
            if len(rmse_c_vals) < 2:
                print(f"[IMPUTATION] Warning: insufficient folds for {method}, skipping.")
                continue
            import numpy as np
            n = len(rmse_c_vals)
            t = 2.776  # t_{0.025, 4}
            def ci(vals):
                a = np.mean(vals); s = np.std(vals, ddof=1)
                h = t * s / np.sqrt(n)
                return a, h
            rc_m, rc_h = ci(rmse_c_vals)
            rg_m, rg_h = ci(rmse_g_vals) if rmse_g_vals else (None, None)
            mc_m, mc_h = ci(mae_c_vals)
            mg_m, mg_h = ci(mae_g_vals) if mae_g_vals else (None, None)
            rg_str = f"{rg_m:.4f}±{rg_h:.4f}" if rg_m is not None else "N/A"
            mg_str = f"{mg_m:.4f}±{mg_h:.4f}" if mg_m is not None else "N/A"
            print(f"[IMPUTATION] {method} | RMSE_c: {rc_m:.4f}±{rc_h:.4f} | RMSE_g: {rg_str} | MAE_c: {mc_m:.4f}±{mc_h:.4f} | MAE_g: {mg_str}")
            summary_rows.append({'method': method, 'RMSE_c_mean': rc_m, 'RMSE_c_ci': rc_h, 'RMSE_g_mean': rg_m, 'RMSE_g_ci': rg_h, 'MAE_c_mean': mc_m, 'MAE_c_ci': mc_h, 'MAE_g_mean': mg_m, 'MAE_g_ci': mg_h})
        out_path = f"{OUTPUT_DIR}/imputation_summary_{args.severity}_{args.condition}.csv"
        if summary_rows:
            keys = summary_rows[0].keys()
            with open(out_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(summary_rows)
            print(f"[IMPUTATION] Saved: {out_path}")
        import sys; sys.exit(0)
    
    survival = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")
    T_all = survival['T'].values.astype(np.float32)
    E_all = survival['E'].values.astype(np.float32)
    
    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)
        
    methods = ['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
    
    times_auc = np.array([12.0, 36.0, 60.0])
    all_summary = []
    
    for method in methods:
        method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"
        if not os.path.exists(method_dir):
            print(f"Warning: {method_dir} does not exist. Skipping {method}.")
            continue
            
        # Initialize full arrays using first available fold result
        full_x_c = None
        full_x_g = None
        
        # Pre-scan to get shapes
        for k_scan in range(1, 6):
            scan_path = f"{method_dir}/fold_{k_scan}_results.pkl"
            
            if os.path.exists(scan_path):
                with open(scan_path, 'rb') as f:
                    scan_res = pickle.load(f)
                n_total = len(T_all)
                full_x_c = np.full((n_total, scan_res['x_hat_c'].shape[1]), np.nan)
                if 'x_hat_g' in scan_res and scan_res['x_hat_g'] is not None:
                    xg_arr = np.asarray(scan_res['x_hat_g'])
                    if xg_arr.ndim == 2 and xg_arr.shape[1] > 0:
                        full_x_g = np.full((n_total, xg_arr.shape[1]), np.nan)
                break
        
        if full_x_c is None:
            print(f"Warning: No result files found for {method}. Skipping.")
            continue
            
        for k, fold_dict in enumerate(folds, 1):
            res_path = f"{method_dir}/fold_{k}_results.pkl"
            if not os.path.exists(res_path):
                continue
            with open(res_path, 'rb') as f:
                res = pickle.load(f)
            idx = res['test_idx']
            x_hat_c = res['x_hat_c']
            
            full_x_c[idx] = x_hat_c
            if full_x_g is not None and 'x_hat_g' in res:
                full_x_g[idx] = res['x_hat_g']
                
        method_results = []
        for k, fold_dict in enumerate(folds, 1):
            train_idx = np.concatenate([fold_dict['train'], fold_dict['val']])
            test_idx = fold_dict['test']
            
            xc_tr, xc_ts = full_x_c[train_idx], full_x_c[test_idx]
            xg_tr = full_x_g[train_idx] if full_x_g is not None else None
            xg_ts = full_x_g[test_idx] if full_x_g is not None else None
            
            if method.startswith(('EF', 'CMA', 'PoE')):
                X_tr = np.concatenate([xc_tr, xg_tr[:, :49]], axis=1)
                X_ts = np.concatenate([xc_ts, xg_ts[:, :49]], axis=1)
            elif method in ['StandardVAE', 'MNARSVAE'] or method.startswith('SIGMA_clinical'):
                X_tr = xc_tr
                X_ts = xc_ts
            elif method.startswith('SIGMA_genomic'):
                X_tr = xg_tr[:, :49] if xg_tr is not None else xc_tr
                X_ts = xg_ts[:, :49] if xg_ts is not None else xc_ts
            else:
                # Classical baselines and GAIN/MIDA: use clinical + expression features only
                X_tr = np.concatenate([xc_tr, xg_tr[:, :49]], axis=1) if xg_tr is not None else xc_tr
                X_ts = np.concatenate([xc_ts, xg_ts[:, :49]], axis=1) if xg_ts is not None else xc_ts
                
            imp = SimpleImputer(strategy='mean')
            X_tr = imp.fit_transform(X_tr)
            X_ts = imp.transform(X_ts)
            
            T_tr, E_tr = T_all[train_idx], E_all[train_idx]
            T_ts, E_ts = T_all[test_idx], E_all[test_idx]
            
            y_tr = Surv.from_arrays(E_tr.astype(bool), T_tr)
            y_ts = Surv.from_arrays(E_ts.astype(bool), T_ts)
            
            np.random.seed(42)
            torch.manual_seed(42)
            
            # RSF
            rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
            rsf.fit(X_tr, y_tr)
            risk_rsf = rsf.predict(X_ts)
            
            # XGBoost
            xgb = XGBRegressor(objective='survival:cox', n_estimators=100, random_state=42)
            y_xgb_tr = T_tr.copy()
            y_xgb_tr[E_tr == 0] = -y_xgb_tr[E_tr == 0]
            xgb.fit(X_tr, y_xgb_tr)
            risk_xgb = xgb.predict(X_ts)
            risk_xgb = np.clip(risk_xgb, -1e6, 1e6)
            risk_xgb = np.nan_to_num(risk_xgb, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # DeepSurv
            risk_ds = train_deepsurv(X_tr, T_tr, E_tr, X_ts)
            
            # DeepHit
            risk_dh = train_deephit(X_tr, T_tr, E_tr, X_ts)
            
            def evaluate_risk(risk):
                c_idx = concordance_index_censored(E_ts.astype(bool), T_ts, risk)[0]
                try:
                    t_min_ts = y_ts['time'][y_ts['event']].min() if y_ts['event'].any() else 0
                    t_max_ts = y_ts['time'].max()
                    valid_times = times_auc[(times_auc > t_min_ts) & (times_auc < t_max_ts)]
                    if len(valid_times) == 0:
                        a1, a3, a5 = np.nan, np.nan, np.nan
                    else:
                        auc_vals, _ = cumulative_dynamic_auc(y_tr, y_ts, risk, valid_times)
                        auc_dict = dict(zip(valid_times, auc_vals))
                        a1 = auc_dict.get(12.0, np.nan)
                        a3 = auc_dict.get(36.0, np.nan)
                        a5 = auc_dict.get(60.0, np.nan)
                except Exception as e:
                    a1, a3, a5 = np.nan, np.nan, np.nan
                return c_idx, a1, a3, a5
            
            c_rsf, a1_rsf, a3_rsf, a5_rsf = evaluate_risk(risk_rsf)
            c_xgb, a1_xgb, a3_xgb, a5_xgb = evaluate_risk(risk_xgb)
            c_ds, a1_ds, a3_ds, a5_ds = evaluate_risk(risk_ds)
            c_dh, a1_dh, a3_dh, a5_dh = evaluate_risk(risk_dh)
            
            try:
                survs = rsf.predict_survival_function(X_ts)
                times = rsf.unique_times_
                t_min = max(y_ts['time'].min(), y_tr['time'].min()) + 1e-5
                t_max = min(y_ts['time'].max(), y_tr['time'].max()) - 1e-5
                mask = (times > t_min) & (times < t_max)
                eval_times = times[mask]
                if len(eval_times) < 2:
                    ibs = np.nan
                else:
                    preds = np.asarray([[fn(t) for t in eval_times] for fn in survs])
                    ibs = integrated_brier_score(y_tr, y_ts, preds, eval_times)
            except Exception as e:
                print(f"    IBS computation failed: {e}")
                ibs = np.nan
                
            # IBS computed from RSF survival function estimate only.
            # XGBoost, DeepSurv, and DeepHit output risk scores without calibrated survival curves.
                
            print(f"[SURVIVAL] Method: {method} | Fold {k}/5 | C-RSF: {c_rsf:.4f} | C-XGB: {c_xgb:.4f} | C-DS: {c_ds:.4f} | C-DH: {c_dh:.4f} | IBS: {ibs:.4f}")
            
            res_fold = {
                'method': method,
                'fold': k,
                'C_index_RSF': c_rsf,
                'C_index_XGB': c_xgb,
                'C_index_DeepSurv': c_ds,
                'C_index_DeepHit': c_dh,
                'IBS_RSF': ibs,
                'AUC_1yr_RSF': a1_rsf,
                'AUC_3yr_RSF': a3_rsf,
                'AUC_5yr_RSF': a5_rsf,
                'AUC_1yr_XGB': a1_xgb,
                'AUC_3yr_XGB': a3_xgb,
                'AUC_5yr_XGB': a5_xgb,
                'AUC_1yr_DeepSurv': a1_ds,
                'AUC_3yr_DeepSurv': a3_ds,
                'AUC_5yr_DeepSurv': a5_ds,
                'AUC_1yr_DeepHit': a1_dh,
                'AUC_3yr_DeepHit': a3_dh,
                'AUC_5yr_DeepHit': a5_dh
            }
            method_results.append(res_fold)
            
            with open(f"{out_path}/{method}_fold_{k}_survival.pkl", 'wb') as f:
                pickle.dump(res_fold, f)
                
        if len(method_results) > 0:
            df_m = pd.DataFrame(method_results)
            summary_dict = {'method': method}
            
            for col in ['C_index_RSF', 'C_index_XGB', 'C_index_DeepSurv', 'C_index_DeepHit']:
                mean_val = df_m[col].mean()
                std_val = df_m[col].std()
                print(f"[SURVIVAL] Summary | {method} | {col}: {mean_val:.4f} ± {std_val:.4f}")
                
            for col in df_m.columns:
                if col not in ['method', 'fold']:
                    summary_dict[f"{col}_mean"] = df_m[col].mean()
                    summary_dict[f"{col}_std"] = df_m[col].std()
                    summary_dict[f"{col}_ci"] = 2.776 * df_m[col].std() / sqrt(5)
                    
            all_summary.append(summary_dict)
            
    if len(all_summary) > 0:
        pd.DataFrame(all_summary).to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
