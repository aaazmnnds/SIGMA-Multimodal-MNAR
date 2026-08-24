import re

def patch_sigma(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    old_code = "        x_hat_c_features = x_hat_c_test.numpy()[:, 1:]"
    new_code = """        x_hat_c_np = x_hat_c_test.numpy()
        bin_cols = [1, 2, 3, 4, 5, 6]
        x_hat_c_np[:, bin_cols] = 1 / (1 + np.exp(-x_hat_c_np[:, bin_cols]))
        x_hat_c_features = x_hat_c_np[:, 1:]"""
    
    content = content.replace(old_code, new_code)
    
    # Also update the dictionary
    content = content.replace("'x_hat_c': x_hat_c_test.numpy(),", "'x_hat_c': x_hat_c_np,")
    
    with open(filepath, 'w') as f:
        f.write(content)

def patch_mnarsvae(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    old_code = "        x_hat_c_np = np.concatenate(x_hat_c_full)"
    new_code = """        x_hat_c_np = np.concatenate(x_hat_c_full)
        bin_cols = [1, 2, 3, 4, 5, 6]
        x_hat_c_np[:, bin_cols] = 1 / (1 + np.exp(-x_hat_c_np[:, bin_cols]))"""
        
    content = content.replace(old_code, new_code)
    
    with open(filepath, 'w') as f:
        f.write(content)

def patch_myevae(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    old_code = "            x_hat_c = x_hat_c.cpu().numpy()"
    new_code = """            x_hat_c = x_hat_c.cpu().numpy()
            bin_cols = [1, 2, 3, 4, 5, 6]
            x_hat_c[:, bin_cols] = 1 / (1 + np.exp(-x_hat_c[:, bin_cols]))"""
            
    content = content.replace(old_code, new_code)
    
    with open(filepath, 'w') as f:
        f.write(content)

def patch_dl(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Optuna loop 
    old_code_1 = """                X_c_imp = x_hat.cpu().numpy()
                eta_ts = eta.cpu().numpy()"""
    new_code_1 = """                X_c_imp = x_hat.cpu().numpy()
                if args.method == 'StandardVAE':
                    bin_cols = [1, 2, 3, 4, 5, 6]
                    X_c_imp[:, bin_cols] = 1 / (1 + np.exp(-X_c_imp[:, bin_cols]))
                eta_ts = eta.cpu().numpy()"""
                
    content = content.replace(old_code_1, new_code_1)
    
    # Retraining loop
    # For StandardVAE we need to apply sigmoid before the mix
    # Let's find: X_c_imp, _, _, _, eta_ts = model(x_test)
    
    old_code_2 = """                X_c_imp, _, _, _, eta_ts = model(x_test)
                M_obs = 1.0 - m_test
                X_c_imp = M_obs * x_test + (1 - M_obs) * X_c_imp"""
    
    new_code_2 = """                X_c_imp, _, _, _, eta_ts = model(x_test)
                if args.method == 'StandardVAE':
                    bin_cols = [1, 2, 3, 4, 5, 6]
                    X_c_imp[:, bin_cols] = torch.sigmoid(X_c_imp[:, bin_cols])
                M_obs = 1.0 - m_test
                X_c_imp = M_obs * x_test + (1 - M_obs) * X_c_imp"""
                
    content = content.replace(old_code_2, new_code_2)
    
    with open(filepath, 'w') as f:
        f.write(content)

base_dir = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts"
patch_sigma(f"{base_dir}/train_sigma_optuna.py")
patch_mnarsvae(f"{base_dir}/train_mnarsvae_optuna.py")
patch_myevae(f"{base_dir}/train_myevae_optuna.py")
patch_dl(f"{base_dir}/train_dl_baselines_optuna.py")

print("All patched!")
