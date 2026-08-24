import re

filepath = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts/train_baselines_optuna.py"
with open(filepath, 'r') as f:
    content = f.read()

replacement = """    from scipy.special import expit
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
"""

content = re.sub(r'    masks_c = np\.load\(f"\{BASE\}/mnar_simulation/mnar_masks_metabric_clinical\.npy"\).*?mask_g_full\[:, :49\] = mask_g_expr\n', replacement, content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(content)
