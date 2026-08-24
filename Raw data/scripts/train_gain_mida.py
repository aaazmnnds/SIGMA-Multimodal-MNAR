# GAIN and MIDA baseline imputation script for SIGMA study.
import os, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit

class GAIN_Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )
    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1))

class GAIN_Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            nn.Sigmoid()
        )
    def forward(self, x_hat, h):
        return self.net(torch.cat([x_hat, h], dim=1))

class MIDANet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

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
    
    out_path = f"{OUTPUT_DIR}/gain_mida_{args.severity}_{args.condition}"
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    print(f"[GAIN-MIDA] Config: severity={args.severity} | condition={args.condition}")
    
    clinical = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    survival = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")
    
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
        X_c_features_arr = X_c[:, 1:]
        prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features_arr) + beta_c)
        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)
        age_mask_col = np.zeros((mask_c_features.shape[0], 1), dtype=np.float32)
        mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
        prob_g = expit(alpha_g * age_indicator[:, None] * np.abs(X_expr) + beta_g)
        mask_g_expr_joint = np.random.binomial(1, prob_g).astype(np.float32)
        mask_g_expr_full = mask_g_expr_joint
        
    X_c_masked = X_c.copy()
    X_c_masked[mask_c_full == 1] = 0.0
    
    X_g = genomic.drop(columns=['PATIENT_ID']).values.astype(np.float32)
    X_g_masked = X_g.copy()
    X_g_masked[:, :49][mask_g_expr_full == 1] = 0.0
    mask_g_full = np.zeros_like(X_g)
    mask_g_full[:, :49] = mask_g_expr_full
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_gain(X_train, M_train, iters=5000, batch_size=128, alpha=100.0):
        dim = X_train.shape[1]
        G = GAIN_Generator(dim).to(device)
        D = GAIN_Discriminator(dim).to(device)
        G_opt = torch.optim.Adam(G.parameters(), lr=0.001)
        D_opt = torch.optim.Adam(D.parameters(), lr=0.001)
        n_samples = X_train.shape[0]
        
        for _ in range(iters):
            idx = np.random.choice(n_samples, batch_size)
            X_mb = torch.FloatTensor(X_train[idx]).to(device)
            # M_train represents the missingness mask (1=missing). 
            # We define M_obs (1=observed) for GAIN formulas.
            M_obs = 1.0 - torch.FloatTensor(M_train[idx]).to(device)
            Z_mb = torch.rand(batch_size, dim).to(device)
            hint_mask = (torch.rand(batch_size, dim).to(device) < 0.9).float()
            H_mb = hint_mask * M_obs + (1 - hint_mask) * 0.5
            X_corrupted = M_obs * X_mb + (1 - M_obs) * Z_mb
            
            # Discriminator step
            G_sample = G(X_corrupted, M_obs)
            X_hat = M_obs * X_mb + (1 - M_obs) * G_sample
            D_prob = D(X_hat.detach(), H_mb)
            D_loss = -torch.mean(M_obs * torch.log(D_prob + 1e-8) + (1 - M_obs) * torch.log(1 - D_prob + 1e-8))
            
            D_opt.zero_grad()
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            D_opt.step()
            
            # Generator step
            G_sample = G(X_corrupted, M_obs)
            X_hat = M_obs * X_mb + (1 - M_obs) * G_sample
            D_prob = D(X_hat, H_mb)
            G_loss_temp = -torch.mean((1 - M_obs) * torch.log(D_prob + 1e-8))
            MSE_loss = torch.mean((M_obs * G_sample - M_obs * X_mb) ** 2) / (torch.mean(M_obs) + 1e-8)
            G_loss = G_loss_temp + alpha * MSE_loss
            
            G_opt.zero_grad()
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            G_opt.step()
            
        return G
        
    def test_gain(G, X_test, M_test):
        dim = X_test.shape[1]
        G.eval()
        with torch.no_grad():
            X_ts = torch.FloatTensor(X_test).to(device)
            M_obs = 1.0 - torch.FloatTensor(M_test).to(device)
            Z_ts = torch.rand(X_ts.shape[0], dim).to(device)
            X_corrupted = M_obs * X_ts + (1 - M_obs) * Z_ts
            G_sample = G(X_corrupted, M_obs)
            X_hat = M_obs * X_ts + (1 - M_obs) * G_sample
        return X_hat.cpu().numpy()
        
    def train_mida(X_train, M_train, epochs=200, batch_size=128):
        dim = X_train.shape[1]
        model = MIDANet(dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(M_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for ep in range(epochs):
            model.train()
            for x_mb, m_mb in loader:
                x_mb = x_mb.to(device)
                M_obs = 1.0 - m_mb.to(device)
                drop_mask = (torch.rand(x_mb.shape).to(device) > 0.5).float()
                M_corrupted = M_obs * drop_mask
                X_corrupted = x_mb * M_corrupted
                x_hat = model(X_corrupted)
                loss = torch.sum(((x_hat - x_mb) * M_obs)**2) / (torch.sum(M_obs) + 1e-8)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                
        return model
        
    def test_mida(model, X_test, M_test, n_passes=5):
        model.eval()
        X_ts = torch.FloatTensor(X_test).to(device)
        M_obs = 1.0 - torch.FloatTensor(M_test).to(device)
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                drop_mask = (torch.rand(X_ts.shape).to(device) > 0.5).float()
                M_corrupted = M_obs * drop_mask
                X_corrupted = X_ts * M_corrupted
                preds.append(model(X_corrupted))
            mean_pred = torch.stack(preds).mean(dim=0)
            X_hat = M_obs * X_ts + (1 - M_obs) * mean_pred
        return X_hat.cpu().numpy()
        
    results_gain = []
    results_mida = []
    
    for k, fold_dict in enumerate(folds, 1):
        np.random.seed(42)
        torch.manual_seed(42)
        train_idx = fold_dict['train']
        test_idx = fold_dict['test']
        
        # GAIN Clinical
        print(f"[GAIN-MIDA] GAIN | Fold {k}/5 | Training...")
        G_c = train_gain(X_c_masked[train_idx], mask_c_full[train_idx])
        X_c_hat_gain = test_gain(G_c, X_c_masked[test_idx], mask_c_full[test_idx])
        
        # GAIN Genomic
        G_g = train_gain(X_g_masked[train_idx, :49], mask_g_full[train_idx, :49])
        X_g_expr_hat_gain = test_gain(G_g, X_g_masked[test_idx, :49], mask_g_full[test_idx, :49])
        X_g_hat_gain = X_g_masked[test_idx].copy()
        X_g_hat_gain[:, :49] = X_g_expr_hat_gain
        
        # Eval GAIN
        x_c_test_true = X_c[test_idx]
        x_g_test_true = X_g[test_idx]
        mask_c_test = mask_c_full[test_idx]
        mask_g_test = mask_g_full[test_idx]
        
        obs_c = mask_c_test[:, 1:] == 1
        diff_c = X_c_hat_gain[:, 1:][obs_c] - x_c_test_true[:, 1:][obs_c]
        rmse_c_gain = np.sqrt(np.mean(diff_c**2)) if obs_c.sum() > 0 else 0.0
        mae_c_gain = np.mean(np.abs(diff_c)) if obs_c.sum() > 0 else 0.0
        
        obs_g = mask_g_test[:, :49] == 1
        diff_g = X_g_hat_gain[:, :49][obs_g] - x_g_test_true[:, :49][obs_g]
        rmse_g_gain = np.sqrt(np.mean(diff_g**2)) if obs_g.sum() > 0 else 0.0
        mae_g_gain = np.mean(np.abs(diff_g)) if obs_g.sum() > 0 else 0.0
        
        print(f"[GAIN-MIDA] GAIN | Fold {k}/5 | RMSE_c: {rmse_c_gain:.4f} | MAE_c: {mae_c_gain:.4f} | RMSE_g: {rmse_g_gain:.4f} | MAE_g: {mae_g_gain:.4f}")
        
        with open(f"{out_path}/gain_fold_{k}_results.pkl", 'wb') as f:
            pickle.dump({'x_hat_c': X_c_hat_gain, 'x_hat_g': X_g_hat_gain, 'test_idx': test_idx, 
                         'rmse_c': rmse_c_gain, 'mae_c': mae_c_gain, 'rmse_g': rmse_g_gain, 'mae_g': mae_g_gain}, f)
        
        results_gain.append({'fold': k, 'rmse_c': rmse_c_gain, 'mae_c': mae_c_gain, 'rmse_g': rmse_g_gain, 'mae_g': mae_g_gain})
        
        # MIDA Clinical
        print(f"[GAIN-MIDA] MIDA | Fold {k}/5 | Training...")
        M_c = train_mida(X_c_masked[train_idx], mask_c_full[train_idx])
        X_c_hat_mida = test_mida(M_c, X_c_masked[test_idx], mask_c_full[test_idx])
        
        # MIDA Genomic
        M_g = train_mida(X_g_masked[train_idx, :49], mask_g_full[train_idx, :49])
        X_g_expr_hat_mida = test_mida(M_g, X_g_masked[test_idx, :49], mask_g_full[test_idx, :49])
        X_g_hat_mida = X_g_masked[test_idx].copy()
        X_g_hat_mida[:, :49] = X_g_expr_hat_mida
        
        # Eval MIDA
        diff_c = X_c_hat_mida[:, 1:][obs_c] - x_c_test_true[:, 1:][obs_c]
        rmse_c_mida = np.sqrt(np.mean(diff_c**2)) if obs_c.sum() > 0 else 0.0
        mae_c_mida = np.mean(np.abs(diff_c)) if obs_c.sum() > 0 else 0.0
        
        diff_g = X_g_hat_mida[:, :49][obs_g] - x_g_test_true[:, :49][obs_g]
        rmse_g_mida = np.sqrt(np.mean(diff_g**2)) if obs_g.sum() > 0 else 0.0
        mae_g_mida = np.mean(np.abs(diff_g)) if obs_g.sum() > 0 else 0.0
        
        print(f"[GAIN-MIDA] MIDA | Fold {k}/5 | RMSE_c: {rmse_c_mida:.4f} | MAE_c: {mae_c_mida:.4f} | RMSE_g: {rmse_g_mida:.4f} | MAE_g: {mae_g_mida:.4f}")
        
        with open(f"{out_path}/mida_fold_{k}_results.pkl", 'wb') as f:
            pickle.dump({'x_hat_c': X_c_hat_mida, 'x_hat_g': X_g_hat_mida, 'test_idx': test_idx, 
                         'rmse_c': rmse_c_mida, 'mae_c': mae_c_mida, 'rmse_g': rmse_g_mida, 'mae_g': mae_g_mida}, f)
                         
        results_mida.append({'fold': k, 'rmse_c': rmse_c_mida, 'mae_c': mae_c_mida, 'rmse_g': rmse_g_mida, 'mae_g': mae_g_mida})
        
    df_gain = pd.DataFrame(results_gain)
    df_mida = pd.DataFrame(results_mida)
    summary_data = []
    
    for method, df in zip(['GAIN', 'MIDA'], [df_gain, df_mida]):
        summary_data.append({
            'method': method,
            'mean_rmse_c': df['rmse_c'].mean(), 'std_rmse_c': df['rmse_c'].std(),
            'mean_mae_c': df['mae_c'].mean(), 'std_mae_c': df['mae_c'].std(),
            'mean_rmse_g': df['rmse_g'].mean(), 'std_rmse_g': df['rmse_g'].std(),
            'mean_mae_g': df['mae_g'].mean(), 'std_mae_g': df['mae_g'].std()
        })
        
    pd.DataFrame(summary_data).to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
