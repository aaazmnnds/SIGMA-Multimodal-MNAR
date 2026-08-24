# PAM50 subtype classification script for SIGMA study. Biological validity evaluation.
import os
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True, choices=['independent', 'joint'])
    args = parser.parse_args()

    BASE = "/Users/azmannads/MO-SVAE/Raw data"
    PROCESSED_DIR = f"{BASE}/processed"
    RESULTS_DIR = f"{BASE}/results_final"
    OUTPUT_DIR = f"{BASE}/evaluation"
    
    print(f"[PAM50] Config: severity=severe | condition={args.condition}")

    # Load from METABRIC clinical raw file
    clinical_raw = pd.read_csv(f"{BASE}/original_datasets/brca_metabric/data_clinical_patient.txt", sep='\t', comment='#')
    valid_subtypes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
    subtype_map = dict(zip(clinical_raw['PATIENT_ID'], clinical_raw['CLAUDIN_SUBTYPE']))

    clinical_processed = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    patient_ids = clinical_processed['PATIENT_ID'].values
    patient_subtypes = np.array([subtype_map.get(pid, 'NC') for pid in patient_ids])
    
    # Load folds
    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", "rb") as f:
        folds = pickle.load(f)
        
    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    genomic_cols = genomic.drop(columns=['PATIENT_ID']).columns.tolist()
    expr_cols = [c for c in genomic_cols if not c.endswith('_CNA')]
    x_g_orig = genomic[expr_cols].values.astype(np.float32)

    methods = ['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
    
    out_path = f"{OUTPUT_DIR}/pam50_{args.condition}"
    os.makedirs(out_path, exist_ok=True)
    
    summary_data = []

    for method in methods:
        accuracies = []
        f1_scores = []
        
        for k, fold_dict in enumerate(folds):
            train_idx = np.concatenate([fold_dict['train'], fold_dict['val']])
            test_idx = fold_dict['test']
            # Training data for RF
            X_train_expr = x_g_orig[train_idx, :49]
            y_train_labels = patient_subtypes[train_idx]
            
            # Filter train to known subtypes
            train_mask = np.isin(y_train_labels, valid_subtypes)
            X_train_expr = X_train_expr[train_mask]
            y_train_labels = y_train_labels[train_mask]
            
            y_test_labels = patient_subtypes[test_idx]
            test_mask = np.isin(y_test_labels, valid_subtypes)
            y_test_labels = y_test_labels[test_mask]
            
            if method == 'Ground Truth':
                X_test_imputed_expr = x_g_orig[test_idx, :49][test_mask]
            else:
                res_path = f"{RESULTS_DIR}/{method}_severe_{args.condition}/fold_{k+1}_results.pkl"
                if not os.path.exists(res_path):
                    print(f"Warning: Result file missing for {method} fold {k+1}. Skipping.")
                    continue
                    
                with open(res_path, 'rb') as f:
                    res = pickle.load(f)
                
                if 'x_hat_g' not in res or res['x_hat_g'] is None:
                    continue
                x_hat_g = res['x_hat_g']
                X_test_imputed_expr = x_hat_g[:, :49][test_mask]
                
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X_train_expr, y_train_labels)
            y_pred = rf_clf.predict(X_test_imputed_expr)
            
            acc = accuracy_score(y_test_labels, y_pred)
            f1 = f1_score(y_test_labels, y_pred, average='macro')
            
            accuracies.append(acc)
            f1_scores.append(f1)
            
            print(f"[PAM50] {method} | Fold {k+1}/5 | Accuracy: {acc:.4f} | F1: {f1:.4f}")
            
        if len(accuracies) > 0:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            
            print(f"[PAM50] Summary | {method} | Accuracy: {mean_acc:.4f} \u00b1 {std_acc:.4f} | F1: {mean_f1:.4f} \u00b1 {std_f1:.4f}")
            
            pd.DataFrame({
                'Fold': range(1, len(accuracies) + 1),
                'Accuracy': accuracies,
                'F1': f1_scores
            }).to_csv(f"{out_path}/{method}_pam50_results.csv", index=False)
            
            summary_data.append({
                'Method': method,
                'Mean_Accuracy': mean_acc,
                'Std_Accuracy': std_acc,
                'Mean_F1': mean_f1,
                'Std_F1': std_f1
            })
            
    if summary_data:
        pd.DataFrame(summary_data).to_csv(f"{out_path}/summary.csv", index=False)

if __name__ == '__main__':
    main()
