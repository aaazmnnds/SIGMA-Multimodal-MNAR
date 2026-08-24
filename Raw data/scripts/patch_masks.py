import re

base_dir = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts"

def patch_file(filename, has_g=True):
    filepath = f"{base_dir}/{filename}"
    with open(filepath, 'r') as f:
        content = f.read()

    # The block we are looking for is:
    # masks_c = np.load(f"{BASE}/mnar_simulation/mnar_masks_metabric_clinical.npy")
    # OR masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
    # AND lines around severity_idx
    
    # We will find the start of the mask loading.
    
    if has_g:
        replacement = """from scipy.special import expit
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
    mask_g_full[:, :49] = mask_g_expr"""
    else:
        replacement = """from scipy.special import expit
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]

    if args.condition == 'independent':
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
        mask_c_features = masks_c[severity_idx].astype(np.float32)
    else:  # joint
        age = X_c[:, 0]
        tau = np.median(age)
        age_indicator = (age > tau).astype(np.float32)
        calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
        def get_params(modality, severity):
            row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
            return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
        alpha_c, beta_c = get_params('clinical', args.severity)
        np.random.seed(42)
        X_c_features = X_c[:, 1:]
        prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features) + beta_c)
        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)

    age_mask_col = np.zeros((X_c.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)"""

    # We need to find where the old mask logic starts.
    # It usually starts with "masks_c = np.load(..."
    
    lines = content.split('\n')
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "masks_c = np.load(" in line:
            start_idx = i
            break
            
    if start_idx != -1:
        # Find where it ends (where age_mask_col or mask_c_full is built or mask_g_expr_full is built)
        # Actually let's just trace down to where the new replacement should end
        # We need to remove all existing mask logic until X_c_masked = X_c.copy() or similar
        for i in range(start_idx, len(lines)):
            if "X_c_masked = X_c.copy()" in lines[i] or "if args.condition == 'joint':" in lines[i]:
                end_idx = i
                break
                
        if end_idx != -1:
            # If the script has a joint condition block, we need to remove it completely
            if "if args.condition == 'joint':" in lines[end_idx]:
                # find the end of the joint block
                # it ends right before `train_idx_0 = folds[0]['train']` or `print(f"[...`
                for j in range(end_idx, len(lines)):
                    if "print(f\"[" in lines[j] or "train_idx_0 =" in lines[j] or "X_c_masked = X_c.copy()" in lines[j]:
                        end_idx = j
                        break
                        
            new_lines = lines[:start_idx] + replacement.split('\n') + lines[end_idx:]
            
            # remove duplicate masks_g_expr or masks_g loading if any still exist
            final_lines = []
            skip = False
            for line in new_lines:
                if "masks_c = np.load" in line and "SIMULATION_DIR" not in line and "BASE" not in line:
                    continue # handled by regex
                final_lines.append(line)
                
            # Actually, to be very safe, let's write a targeted regex per file, or just use python string replacement.
            pass

def targeted_patch(filename, has_g):
    filepath = f"{base_dir}/{filename}"
    with open(filepath, 'r') as f:
        content = f.read()

    replacement = """from scipy.special import expit
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
    
    mask_g_expr_full = mask_g_expr  # alias just in case
"""

    if not has_g:
        replacement = """from scipy.special import expit
    SIMULATION_DIR = f"{BASE}/mnar_simulation"
    severity_idx = {'light': 0, 'moderate': 1, 'severe': 2}[args.severity]

    if args.condition == 'independent':
        masks_c = np.load(f"{SIMULATION_DIR}/mnar_masks_metabric_clinical.npy")
        mask_c_features = masks_c[severity_idx].astype(np.float32)
    else:  # joint
        age = X_c[:, 0]
        tau = np.median(age)
        age_indicator = (age > tau).astype(np.float32)
        calib = pd.read_csv(f"{SIMULATION_DIR}/calibration_results.csv")
        def get_params(modality, severity):
            row = calib[(calib['modality'] == modality) & (calib['severity'] == severity)]
            return float(row['alpha'].iloc[0]), float(row['beta'].iloc[0])
        alpha_c, beta_c = get_params('clinical', args.severity)
        np.random.seed(42)
        X_c_features = X_c[:, 1:]
        prob_c = expit(alpha_c * age_indicator[:, None] * np.abs(X_c_features) + beta_c)
        mask_c_features = np.random.binomial(1, prob_c).astype(np.float32)

    age_mask_col = np.zeros((X_c.shape[0], 1), dtype=np.float32)
    mask_c_full = np.concatenate([age_mask_col, mask_c_features], axis=1)
"""

    if filename == 'train_sigma_optuna.py':
        # Remove from `masks_c = np.load...` down to the end of `if args.condition == 'joint': ...`
        pattern = r'    masks_c = np\.load\(f"\{SIMULATION_DIR\}/mnar_masks_metabric_clinical\.npy"\).*?mask_g_expr_full = mask_g_expr\n'
        # Wait, the end is not known cleanly. I will use a manual substring replace.
        content = re.sub(r'    masks_c = np\.load\(f"\{SIMULATION_DIR\}/mnar_masks_metabric_clinical\.npy"\).*?            mask_g_expr_full = mask_g_expr\n', replacement, content, flags=re.DOTALL)
        
    elif filename == 'train_mnarsvae_optuna.py':
        # Remove from `masks_c = np.load...` to `mask_c_full = np.concatenate...`
        content = re.sub(r'    masks_c = np\.load\(f"\{BASE\}/mnar_simulation/mnar_masks_metabric_clinical\.npy"\).*?mask_c_full = np\.concatenate\(\[age_mask_col, mask_c_full\], axis=1\)\n', replacement, content, flags=re.DOTALL)
        
    elif filename == 'train_myevae_optuna.py':
        content = re.sub(r'    masks_c = np\.load\(f"\{SIMULATION_DIR\}/mnar_masks_metabric_clinical\.npy"\).*?            mask_g_expr_full = mask_g_expr\n', replacement, content, flags=re.DOTALL)
        
    elif filename == 'train_dl_baselines_optuna.py':
        content = re.sub(r'    masks_c = np\.load\(f"\{BASE\}/mnar_simulation/mnar_masks_metabric_clinical\.npy"\).*?mask_g_full\[:, :49\] = mask_g_expr\n', replacement, content, flags=re.DOTALL)

    with open(filepath, 'w') as f:
        f.write(content)

targeted_patch('train_sigma_optuna.py', True)
targeted_patch('train_mnarsvae_optuna.py', False)
targeted_patch('train_myevae_optuna.py', True)
targeted_patch('train_dl_baselines_optuna.py', True)
