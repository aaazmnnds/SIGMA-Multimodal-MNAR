import re

base_dir = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts"
for file in ["train_sigma_optuna.py", "train_mnarsvae_optuna.py", "train_myevae_optuna.py", "train_dl_baselines_optuna.py"]:
    filepath = f"{base_dir}/{file}"
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the missing indentation for the first import
    content = content.replace("from scipy.special import expit\n    SIMULATION_DIR", "    from scipy.special import expit\n    SIMULATION_DIR")
    
    with open(filepath, 'w') as f:
        f.write(content)
