# Integration quality evaluation script for SIGMA study.
import os, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--severity', type=str, required=True, choices=['light', 'moderate', 'severe'])
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    args = parser.parse_args()
    
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_DIR = f"{BASE}/processed"
    RESULTS_DIR = f"{BASE}/results_final"
    OUTPUT_DIR = f"{BASE}/evaluation"
    
    out_path = f"{OUTPUT_DIR}/integration_{args.severity}_{args.condition}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    print(f"[INTEGRATION] Config: severity={args.severity} | condition={args.condition}")
    
    # Load PAM50 subtypes for METABRIC
    clinical_raw_path = f"{BASE}/original_datasets/brca_metabric/data_clinical_patient.txt"
    try:
        clinical_raw = pd.read_csv(clinical_raw_path, sep='\t', comment='#')
        # Map subtypes
        valid_subtypes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
        clinical_raw = clinical_raw[clinical_raw['CLAUDIN_SUBTYPE'].isin(valid_subtypes)]
        subtype_map = dict(zip(clinical_raw['PATIENT_ID'], clinical_raw['CLAUDIN_SUBTYPE']))
    except Exception as e:
        print(f"Warning: Could not load or parse clinical raw data for subtypes: {e}")
        subtype_map = {}
        
    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)
        
    clinical_processed = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    patient_ids = clinical_processed['PATIENT_ID'].values
    
    methods = ['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
    all_summary = []
    
    np.random.seed(42)
    
    for method in methods:
        unimodal_methods = ['MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
        if method in unimodal_methods:
            print(f"[INTEGRATION] {method} | No latent space — skipping integration metrics")
            all_summary.append({
                'method': method,
                'CCA_corr_mean': float('nan'),
                'CCA_corr_std': float('nan'),
                'Silhouette_mean': float('nan'),
                'Silhouette_std': float('nan')
            })
            continue
            
        method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"
        if not os.path.exists(method_dir):
            print(f"Warning: {method_dir} does not exist. Skipping {method}.")
            continue
            
        method_results = []
        for k, fold_dict in enumerate(folds, 1):
            res_path = f"{method_dir}/fold_{k}_results.pkl"
            train_res_path = f"{method_dir}/fold_{k}_train_results.pkl"
            if not os.path.exists(res_path):
                continue
                
            with open(res_path, 'rb') as f:
                res = pickle.load(f)
                
            z_c_test = res.get('z_c', None)
            z_g_test = res.get('z_g', None)
            z_shared = res.get('z_shared', res.get('z', None))
            test_idx = res['test_idx']
            
            # Extract train latents if available
            z_c_train, z_g_train = None, None
            if os.path.exists(train_res_path):
                with open(train_res_path, 'rb') as f:
                    train_res = pickle.load(f)
                z_c_train = train_res.get('z_c', None)
                z_g_train = train_res.get('z_g', None)
            else:
                z_c_train = res.get('z_c_train', None)
                z_g_train = res.get('z_g_train', None)
                
            # Compute CCA correlation
            corr = np.nan
            if z_c_test is not None and z_g_test is not None:
                if z_c_train is not None and z_g_train is not None:
                    try:
                        cca = CCA(n_components=1)
                        cca.fit(z_c_train, z_g_train)
                        z_c_t, z_g_t = cca.transform(z_c_test, z_g_test)
                        corr = np.corrcoef(z_c_t[:, 0], z_g_t[:, 0])[0, 1]
                    except:
                        pass
                else:
                    # Fallback: fit CCA on first half of test data, transform second half
                    # This avoids inflation from fitting and transforming on the same data
                    try:
                        n = len(z_c_test)
                        split = n // 2
                        if split > 1:
                            cca = CCA(n_components=1)
                            cca.fit(z_c_test[:split], z_g_test[:split])
                            z_c_t, z_g_t = cca.transform(z_c_test[split:], z_g_test[split:])
                            corr = np.corrcoef(z_c_t[:, 0], z_g_t[:, 0])[0, 1]
                    except:
                        pass
                        
            # Compute Silhouette Score
            sil_score = np.nan
            if z_shared is not None:
                test_patient_ids = patient_ids[test_idx]
                test_subtypes = [subtype_map.get(pid, 'Unknown') for pid in test_patient_ids]
                
                # Filter unknown
                valid_mask = [st != 'Unknown' for st in test_subtypes]
                if sum(valid_mask) > 1:
                    z_shared_valid = z_shared[valid_mask]
                    subtypes_valid = np.array(test_subtypes)[valid_mask]
                    
                    if len(np.unique(subtypes_valid)) > 1:
                        le = LabelEncoder()
                        labels_encoded = le.fit_transform(subtypes_valid)
                        sil_score = silhouette_score(z_shared_valid, labels_encoded)
                        
            def fmt(v):
                return f"{v:.4f}" if not np.isnan(v) else "nan"
                
            print(f"[INTEGRATION] {method} | Fold {k}/5 | CCA_corr: {fmt(corr)} | Silhouette: {fmt(sil_score)}")
            
            res_fold = {
                'method': method,
                'fold': k,
                'CCA_corr': corr,
                'Silhouette': sil_score
            }
            method_results.append(res_fold)
            
            with open(f"{out_path}/{method}_fold_{k}_integration.pkl", 'wb') as f:
                pickle.dump(res_fold, f)
                
        if len(method_results) > 0:
            df_m = pd.DataFrame(method_results)
            mean_cca = df_m['CCA_corr'].mean()
            std_cca = df_m['CCA_corr'].std()
            mean_sil = df_m['Silhouette'].mean()
            std_sil = df_m['Silhouette'].std()
            
            def fmt(v):
                return f"{v:.4f}" if not np.isnan(v) else "nan"
                
            print(f"[INTEGRATION] Summary | {method} | CCA_corr: {fmt(mean_cca)} ± {fmt(std_cca)} | Silhouette: {fmt(mean_sil)} ± {fmt(std_sil)}")
            df_m.to_csv(f"{out_path}/{method}_integration.csv", index=False)
            
            all_summary.append({
                'method': method,
                'CCA_corr_mean': mean_cca,
                'CCA_corr_std': std_cca,
                'Silhouette_mean': mean_sil,
                'Silhouette_std': std_sil
            })
            
    if len(all_summary) > 0:
        pd.DataFrame(all_summary).to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
