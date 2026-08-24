# Evaluation script for SIGMA ablation variants.
# Evaluates downstream RSF survival prediction and PAM50 subtype classification.

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore')

def main():
    # Use script directory to derive base directory portably across environments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(script_dir)
    PROCESSED_DIR = f"{BASE}/processed"
    RESULTS_DIR = f"{BASE}/results_final"
    EVAL_DIR = f"{BASE}/evaluation"

    # Define ablation variants to evaluate
    variants = [
        'ablation_no_cross_severe_independent',
        'ablation_no_cox_severe_independent',
        'ablation_no_mask_severe_independent',
        'ablation_clinical_only_severe_independent',
        'ablation_genomic_only_severe_independent'
    ]

    # Load survival targets and cross-validation folds
    survival_df = pd.read_csv(f"{PROCESSED_DIR}/metabric_survival.csv")
    T_all = survival_df['T'].values.astype(np.float32)
    E_all = survival_df['E'].values.astype(np.float32)

    with open(f"{PROCESSED_DIR}/folds/metabric_folds.pkl", 'rb') as f:
        folds = pickle.load(f)

    # Load PAM50 reference data
    clinical_raw = pd.read_csv(f"{BASE}/original_datasets/brca_metabric/data_clinical_patient.txt", sep='\t', comment='#')
    valid_subtypes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
    subtype_map = dict(zip(clinical_raw['PATIENT_ID'], clinical_raw['CLAUDIN_SUBTYPE']))

    clinical_processed = pd.read_csv(f"{PROCESSED_DIR}/metabric_clinical.csv")
    patient_ids = clinical_processed['PATIENT_ID'].values
    patient_subtypes = np.array([subtype_map.get(pid, 'NC') for pid in patient_ids])

    genomic = pd.read_csv(f"{PROCESSED_DIR}/metabric_genomic.csv")
    genomic_cols = genomic.drop(columns=['PATIENT_ID']).columns.tolist()
    expr_cols = [c for c in genomic_cols if not c.endswith('_CNA')]
    x_g_orig = genomic[expr_cols].values.astype(np.float32)

    # Setup output directories
    surv_out_dir = f"{EVAL_DIR}/ablation_survival_severe_independent"
    pam50_out_dir = f"{EVAL_DIR}/ablation_pam50_severe_independent"
    os.makedirs(surv_out_dir, exist_ok=True)
    os.makedirs(pam50_out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("  SIGMA ABLATION DOWNSTREAM EVALUATION (RSF & PAM50)")
    print(f"{'='*65}")

    surv_summary_list = []
    pam50_summary_list = []

    for variant in variants:
        variant_dir = f"{RESULTS_DIR}/{variant}"
        if not os.path.exists(variant_dir):
            print(f"\nWarning: Result directory missing for {variant}. Skipping.")
            continue

        print(f"\n--- Evaluating Variant: {variant} ---")

        # -----------------------------------------------------------------
        # 1. Survival Evaluation (RSF C-index)
        # -----------------------------------------------------------------
        # Reconstruct full-cohort imputed matrices across test folds
        full_x_c = None
        full_x_g = None

        for k in range(1, 6):
            fold_file = f"{variant_dir}/fold_{k}_results.pkl"
            if os.path.exists(fold_file):
                with open(fold_file, 'rb') as f:
                    res = pickle.load(f)
                if full_x_c is None and 'x_hat_c' in res and res['x_hat_c'] is not None:
                    full_x_c = np.full((len(T_all), res['x_hat_c'].shape[1]), np.nan)
                if full_x_g is None and 'x_hat_g' in res and res['x_hat_g'] is not None:
                    full_x_g = np.full((len(T_all), res['x_hat_g'].shape[1]), np.nan)
                if full_x_c is not None and full_x_g is not None:
                    break

        for k in range(1, 6):
            fold_file = f"{variant_dir}/fold_{k}_results.pkl"
            if not os.path.exists(fold_file):
                continue
            with open(fold_file, 'rb') as f:
                res = pickle.load(f)
            idx = res['test_idx']
            if full_x_c is not None and 'x_hat_c' in res and res['x_hat_c'] is not None:
                full_x_c[idx] = res['x_hat_c']
            if full_x_g is not None and 'x_hat_g' in res and res['x_hat_g'] is not None:
                full_x_g[idx] = res['x_hat_g']

        c_indices = []
        for k, fold_dict in enumerate(folds, 1):
            train_idx = np.concatenate([fold_dict['train'], fold_dict['val']])
            test_idx = fold_dict['test']

            xc_tr = full_x_c[train_idx] if full_x_c is not None else None
            xc_ts = full_x_c[test_idx] if full_x_c is not None else None
            xg_tr = full_x_g[train_idx] if full_x_g is not None else None
            xg_ts = full_x_g[test_idx] if full_x_g is not None else None

            if 'clinical_only' in variant:
                X_tr = xc_tr
                X_ts = xc_ts
            elif 'genomic_only' in variant:
                X_tr = xg_tr[:, :49] if xg_tr is not None else None
                X_ts = xg_ts[:, :49] if xg_ts is not None else None
            else:
                X_tr = np.concatenate([xc_tr, xg_tr[:, :49]], axis=1) if (xc_tr is not None and xg_tr is not None) else None
                X_ts = np.concatenate([xc_ts, xg_ts[:, :49]], axis=1) if (xc_ts is not None and xg_ts is not None) else None

            if X_tr is None or X_ts is None:
                continue

            imp = SimpleImputer(strategy='mean')
            X_tr = imp.fit_transform(X_tr)
            X_ts = imp.transform(X_ts)

            T_tr, E_tr = T_all[train_idx], E_all[train_idx]
            T_ts, E_ts = T_all[test_idx], E_all[test_idx]

            y_tr = Surv.from_arrays(E_tr.astype(bool), T_tr)

            rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
            rsf.fit(X_tr, y_tr)
            risk_rsf = rsf.predict(X_ts)

            c_idx = concordance_index_censored(E_ts.astype(bool), T_ts, risk_rsf)[0]
            c_indices.append(c_idx)
            print(f"[Survival RSF] {variant} | Fold {k}/5 | C-index: {c_idx:.4f}")

        if len(c_indices) > 0:
            mean_c = np.mean(c_indices)
            std_c = np.std(c_indices)
            print(f"[Survival RSF] Summary | {variant} | C-index: {mean_c:.4f} ± {std_c:.4f}")

            pd.DataFrame({
                'Fold': range(1, len(c_indices) + 1),
                'C_index_RSF': c_indices
            }).to_csv(f"{surv_out_dir}/{variant}_survival_results.csv", index=False)

            surv_summary_list.append({
                'Variant': variant,
                'Mean_C_index_RSF': mean_c,
                'Std_C_index_RSF': std_c
            })

        # -----------------------------------------------------------------
        # 2. PAM50 Classification Evaluation
        # -----------------------------------------------------------------
        if 'clinical_only' in variant:
            print(f"[PAM50] Skipping {variant} (clinical_only has no genomic expression).")
        else:
            pam_accuracies = []
            pam_f1_scores = []

            for k, fold_dict in enumerate(folds):
                train_idx = np.concatenate([fold_dict['train'], fold_dict['val']])
                test_idx = fold_dict['test']

                X_train_expr = x_g_orig[train_idx, :49]
                y_train_labels = patient_subtypes[train_idx]

                train_mask = np.isin(y_train_labels, valid_subtypes)
                X_train_expr = X_train_expr[train_mask]
                y_train_labels = y_train_labels[train_mask]

                y_test_labels = patient_subtypes[test_idx]
                test_mask = np.isin(y_test_labels, valid_subtypes)
                y_test_labels = y_test_labels[test_mask]

                fold_file = f"{variant_dir}/fold_{k+1}_results.pkl"
                if not os.path.exists(fold_file):
                    print(f"Warning: Result file missing for {variant} fold {k+1}. Skipping PAM50 fold.")
                    continue

                with open(fold_file, 'rb') as f:
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

                pam_accuracies.append(acc)
                pam_f1_scores.append(f1)
                print(f"[PAM50] {variant} | Fold {k+1}/5 | Accuracy: {acc:.4f} | F1: {f1:.4f}")

            if len(pam_accuracies) > 0:
                mean_acc = np.mean(pam_accuracies)
                std_acc = np.std(pam_accuracies)
                mean_f1 = np.mean(pam_f1_scores)
                std_f1 = np.std(pam_f1_scores)

                print(f"[PAM50] Summary | {variant} | Accuracy: {mean_acc:.4f} ± {std_acc:.4f} | F1: {mean_f1:.4f} ± {std_f1:.4f}")

                pd.DataFrame({
                    'Fold': range(1, len(pam_accuracies) + 1),
                    'Accuracy': pam_accuracies,
                    'F1': pam_f1_scores
                }).to_csv(f"{pam50_out_dir}/{variant}_pam50_results.csv", index=False)

                pam50_summary_list.append({
                    'Variant': variant,
                    'Mean_Accuracy': mean_acc,
                    'Std_Accuracy': std_acc,
                    'Mean_F1': mean_f1,
                    'Std_F1': std_f1
                })

    # Save overall summary tables
    if surv_summary_list:
        pd.DataFrame(surv_summary_list).to_csv(f"{surv_out_dir}/ablation_survival_summary.csv", index=False)
    if pam50_summary_list:
        pd.DataFrame(pam50_summary_list).to_csv(f"{pam50_out_dir}/ablation_pam50_summary.csv", index=False)

    print(f"\n{'='*65}")
    print("  EVALUATION COMPLETE")
    print(f"  Survival results saved to: {surv_out_dir}")
    print(f"  PAM50 results saved to:    {pam50_out_dir}")
    print(f"{'='*65}\n")

if __name__ == '__main__':
    main()
