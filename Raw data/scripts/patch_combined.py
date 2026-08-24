import re

filepath = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts/train_baselines_optuna.py"
with open(filepath, 'r') as f:
    content = f.read()

# MICE Optuna block
mice_optuna_old = """            imputer_c = IterativeImputer(max_iter=max_iter, random_state=42)
            imputer_g = IterativeImputer(max_iter=max_iter, random_state=42)
            imputer_c.fit(X_c_masked[train_idx_0])
            imputer_g.fit(X_g_masked[train_idx_0])
            X_c_imp = np.nan_to_num(imputer_c.transform(X_c_masked[val_idx_0]), nan=0.0, posinf=0.0, neginf=0.0)
            X_c_imp = np.clip(X_c_imp, -10.0, 10.0)
            X_g_imp = np.nan_to_num(imputer_g.transform(X_g_masked[val_idx_0]), nan=0.0, posinf=0.0, neginf=0.0)
            X_g_imp = np.clip(X_g_imp, -10.0, 10.0)"""

mice_optuna_new = """            imputer = IterativeImputer(max_iter=max_iter, random_state=42)
            X_combined_train = np.concatenate([X_c_masked[train_idx_0], X_g_masked[train_idx_0]], axis=1)
            X_combined_val = np.concatenate([X_c_masked[val_idx_0], X_g_masked[val_idx_0]], axis=1)
            imputer.fit(X_combined_train)
            X_combined_imp = np.nan_to_num(imputer.transform(X_combined_val), nan=0.0, posinf=0.0, neginf=0.0)
            X_combined_imp = np.clip(X_combined_imp, -10.0, 10.0)
            X_c_imp = X_combined_imp[:, :X_c.shape[1]]
            X_g_imp = X_combined_imp[:, X_c.shape[1]:]"""

content = content.replace(mice_optuna_old, mice_optuna_new)

# missForest Optuna block
mf_optuna_old = """            if HAS_CUML:
                rf_c = cuRF(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf_g = cuRF(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            else:
                rf_c = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                rf_g = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            imputer_c = IterativeImputer(estimator=rf_c, max_iter=max_iter, random_state=42)
            imputer_g = IterativeImputer(estimator=rf_g, max_iter=max_iter, random_state=42)
            imputer_c.fit(X_c_masked[train_idx_0])
            imputer_g.fit(X_g_masked[train_idx_0])
            X_c_imp = np.nan_to_num(imputer_c.transform(X_c_masked[val_idx_0]), nan=0.0, posinf=0.0, neginf=0.0)
            X_c_imp = np.clip(X_c_imp, -10.0, 10.0)
            X_g_imp = np.nan_to_num(imputer_g.transform(X_g_masked[val_idx_0]), nan=0.0, posinf=0.0, neginf=0.0)
            X_g_imp = np.clip(X_g_imp, -10.0, 10.0)"""

mf_optuna_new = """            if HAS_CUML:
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
            X_g_imp = X_combined_imp[:, X_c.shape[1]:]"""

content = content.replace(mf_optuna_old, mf_optuna_new)

# Retraining loop
retrain_old = """        if args.method == 'MICE':
            imputer_c = IterativeImputer(max_iter=best_params['max_iter'], random_state=42)
            imputer_g = IterativeImputer(max_iter=best_params['max_iter'], random_state=42)
        elif args.method == 'missForest':
            if HAS_CUML:
                rf_c = cuRF(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
                rf_g = cuRF(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
            else:
                rf_c = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42, n_jobs=-1)
                rf_g = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42, n_jobs=-1)
            imputer_c = IterativeImputer(estimator=rf_c, max_iter=best_params['max_iter'], random_state=42)
            imputer_g = IterativeImputer(estimator=rf_g, max_iter=best_params['max_iter'], random_state=42)
            
        imputer_c.fit(X_c_masked[train_idx])
        imputer_g.fit(X_g_masked[train_idx])
        
        X_c_imp = np.nan_to_num(imputer_c.transform(X_c_masked[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        X_c_imp = np.clip(X_c_imp, -10.0, 10.0)
        X_g_imp = np.nan_to_num(imputer_g.transform(X_g_masked[test_idx]), nan=0.0, posinf=0.0, neginf=0.0)
        X_g_imp = np.clip(X_g_imp, -10.0, 10.0)"""

retrain_new = """        if args.method == 'MICE':
            imputer = IterativeImputer(max_iter=best_params['max_iter'], random_state=42)
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
        X_g_imp = X_combined_imp[:, X_c.shape[1]:]"""

content = content.replace(retrain_old, retrain_new)

with open(filepath, 'w') as f:
    f.write(content)

