# Figure generation script for SIGMA study.
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Numpy1xUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATION_DIR = f"{BASE}/evaluation"
RESULTS_DIR = f"{BASE}/results_final"
OUTPUT_DIR = f"{BASE}/figures"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png'
})

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    severities = ['light', 'moderate', 'severe']
    conditions = ['independent', 'joint']
    
    methods_plot = {
        'SIGMA EF': {'color': 'red', 'linestyle': '-', 'marker': 'o'},
        'SIGMA CMA': {'color': 'blue', 'linestyle': '-', 'marker': 's'},
        'SIGMA PoE': {'color': 'green', 'linestyle': '-', 'marker': '^'},
        'MICE': {'color': 'gray', 'linestyle': '--', 'marker': 'o'},
        'missForest': {'color': 'brown', 'linestyle': '--', 'marker': 's'},
        'GAIN': {'color': 'orange', 'linestyle': '--', 'marker': 'D'},
        'MIDA': {'color': 'purple', 'linestyle': '--', 'marker': '^'},
        'MyeVAE': {'color': 'magenta', 'linestyle': ':', 'marker': 'v'}
    }
    
    def get_summary_stats(method, metric, severity, condition):
        try:
            vals = []
            prefix = method.replace('SIGMA ', '')
            for k in range(1, 6):
                fpath = f"{RESULTS_DIR}/{prefix}_{severity}_{condition}/fold_{k}_results.pkl"
                if os.path.exists(fpath):
                    with open(fpath, 'rb') as f:
                        res = Numpy1xUnpickler(f).load()
                    val = res.get(metric, np.nan)
                    if val is not None:
                        vals.append(float(val))
            vals = np.array(vals)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                return np.mean(vals), np.std(vals)
            return np.nan, np.nan
        except Exception as e:
            return np.nan, np.nan

    def get_survival_stats(method, severity, condition):
        try:
            path = f"{EVALUATION_DIR}/survival_{severity}_{condition}/summary.csv"
            if not os.path.exists(path): return np.nan, np.nan
            df = pd.read_csv(path)
            
            name_map = {
                'SIGMA EF': 'EF',
                'SIGMA CMA': 'CMA',
                'SIGMA PoE': 'PoE',
                'MyeVAE': 'MyeVAE',
                'MICE': 'MICE',
                'missForest': 'missForest',
                'GAIN': 'GAIN',
                'MIDA': 'MIDA'
            }
            m_name = name_map.get(method, method)
            row = df[df['method'] == m_name]
            if not row.empty:
                return row.iloc[0]['C_index_RSF_mean'], row.iloc[0]['C_index_RSF_std']
            return np.nan, np.nan
        except Exception:
            return np.nan, np.nan

    # Figure 1: RMSE_g by Severity (line plot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_labels = ['Light', 'Moderate', 'Severe']
    x_ticks = [0, 1, 2]
    
    for i, condition in enumerate(conditions):
        ax = axes[i]
        for method, style in methods_plot.items():
            means = []
            stds = []
            for sev in severities:
                m, s = get_summary_stats(method, 'rmse_g', sev, condition)
                means.append(m)
                stds.append(s)
            means = np.array(means)
            stds = np.array(stds)
            
            valid = ~np.isnan(means)
            if np.any(valid):
                ax.plot(np.array(x_ticks)[valid], means[valid], 
                        color=style['color'], linestyle=style['linestyle'], 
                        marker=style['marker'], label=method)
                ax.fill_between(np.array(x_ticks)[valid], 
                                means[valid] - stds[valid], 
                                means[valid] + stds[valid], 
                                color=style['color'], alpha=0.1)
                
        ax.axhline(y=1.0, color='gray', linestyle='-.', label='z-score baseline')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Severity Level')
        ax.set_ylabel('Genomic Imputation RMSE')
        ax.set_title(f'Condition: {condition.capitalize()}')
        if i == 0:
            ax.legend(loc='upper right', framealpha=0.8)
            
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rmse_by_severity.png")
    plt.close()
    print("[FIGURES] Saved rmse_by_severity.png")

    # Figure 2: PAM50 Subtype Classification Accuracy (bar chart)
    def load_pam50_summary(condition):
        path = f"{EVALUATION_DIR}/pam50_{condition}/summary.csv"
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_csv(path)
    
    pam50_ind = load_pam50_summary('independent')
    pam50_jnt = load_pam50_summary('joint')
    
    if not pam50_ind.empty and not pam50_jnt.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        name_map_rev = {
            'EF': 'SIGMA EF', 'CMA': 'SIGMA CMA', 'PoE': 'SIGMA PoE',
            'MyeVAE': 'MyeVAE', 'StandardVAE': 'StandardVAE',
            'MICE': 'MICE', 'missForest': 'missForest',
            'GAIN': 'GAIN', 'MIDA': 'MIDA'
        }
        
        for i, (cond, df) in enumerate([('Independent', pam50_ind), ('Joint', pam50_jnt)]):
            ax = axes[i]
            if 'Method' in df.columns:
                df['Plot_Method'] = df['Method'].map(lambda x: name_map_rev.get(x, x))
                
                methods_order = ['SIGMA EF', 'SIGMA CMA', 'SIGMA PoE', 'MyeVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
                plot_df = df[df['Plot_Method'].isin(methods_order)].copy()
                plot_df['sort_key'] = plot_df['Plot_Method'].map(lambda x: methods_order.index(x) if x in methods_order else 999)
                plot_df = plot_df.sort_values('sort_key')
                
                colors = []
                for m in plot_df['Plot_Method']:
                    if 'SIGMA' in m:
                        colors.append(sns.color_palette("Reds")[3])
                    elif 'VAE' in m:
                        colors.append(sns.color_palette("Purples")[3])
                    else:
                        colors.append(sns.color_palette("Greys")[3])
                
                x_pos = np.arange(len(plot_df))
                
                ax.bar(x_pos, plot_df['Mean_Accuracy'], yerr=plot_df['Std_Accuracy'], color=colors, capsize=5)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(plot_df['Plot_Method'], rotation=45, ha='right')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'PAM50 Accuracy - {cond}')
                ax.set_ylim(0, 0.85)
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/pam50_accuracy.png")
        plt.close()
        print("[FIGURES] Saved pam50_accuracy.png")

    # Figure 3: Survival C-index by Severity (line plot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, condition in enumerate(conditions):
        ax = axes[i]
        for method, style in methods_plot.items():
            means = []
            stds = []
            for sev in severities:
                m, s = get_survival_stats(method, sev, condition)
                means.append(m)
                stds.append(s)
            means = np.array(means)
            stds = np.array(stds)
            
            valid = ~np.isnan(means)
            if np.any(valid):
                ax.plot(np.array(x_ticks)[valid], means[valid], 
                        color=style['color'], linestyle=style['linestyle'], 
                        marker=style['marker'], label=method)
                ax.fill_between(np.array(x_ticks)[valid], 
                                means[valid] - stds[valid], 
                                means[valid] + stds[valid], 
                                color=style['color'], alpha=0.1)
                
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Severity Level')
        ax.set_ylabel('C-index RSF')
        ax.set_title(f'Condition: {condition.capitalize()}')
        if i == 0:
            ax.legend(loc='lower left', framealpha=0.8)
            
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cindex_by_severity.png")
    plt.close()
    print("[FIGURES] Saved cindex_by_severity.png")

    # Figure 4: t-SNE Latent Space Visualization
    try:
        clinical_raw_path = f"{BASE}/original_datasets/brca_metabric/data_clinical_patient.txt"
        clinical_processed_path = f"{BASE}/processed/metabric_clinical.csv"
        if os.path.exists(clinical_raw_path) and os.path.exists(clinical_processed_path):
            clinical_raw = pd.read_csv(clinical_raw_path, sep='\t', comment='#')
            clinical_processed = pd.read_csv(clinical_processed_path)
            subtype_map = dict(zip(clinical_raw['PATIENT_ID'], clinical_raw['CLAUDIN_SUBTYPE']))
            subtypes = np.array([subtype_map.get(pid, 'NC') for pid in clinical_processed['PATIENT_ID'].values])
            
            folds_path = f"{BASE}/processed/folds/metabric_folds.pkl"
            with open(folds_path, 'rb') as f:
                folds = Numpy1xUnpickler(f).load()
            test_idx = folds[0]['test']
            test_subtypes = subtypes[test_idx]
            
            le = LabelEncoder()
            labels = le.fit_transform(test_subtypes)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fusions = ['EF', 'CMA', 'PoE']
            
            for i, fusion in enumerate(fusions):
                ax = axes[i]
                res_path = f"{RESULTS_DIR}/{fusion}_severe_independent/fold_1_results.pkl"
                if os.path.exists(res_path):
                    with open(res_path, 'rb') as f:
                        res = Numpy1xUnpickler(f).load()
                    z_shared = res['z_shared']
                    
                    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                    embedding = reducer.fit_transform(z_shared)
                    
                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=15, alpha=0.8)
                    ax.set_title(f'SIGMA {fusion}')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if i == 2:
                        handles, _ = scatter.legend_elements()
                        ax.legend(handles, le.classes_, title="PAM50 Subtype", loc='center left', bbox_to_anchor=(1, 0.5))
                        
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/tsne_latent.png")
            plt.close()
            print("[FIGURES] Saved tsne_latent.png")
    except Exception as e:
        print(f"[FIGURES] t-SNE failed: {e}")

    # Figure 5: Imputation Error Distribution (boxplot)
    plot_methods = ['SIGMA EF', 'SIGMA CMA', 'SIGMA PoE', 'MICE', 'missForest']
    data = []
    
    for condition in ['independent']:
        for sev in severities:
            for method in plot_methods:
                prefix = method.replace('SIGMA ', '')
                for k in range(1, 6):
                    fpath = f"{RESULTS_DIR}/{prefix}_{sev}_{condition}/fold_{k}_results.pkl"
                    if os.path.exists(fpath):
                        with open(fpath, 'rb') as f:
                            res = Numpy1xUnpickler(f).load()
                        rmse = res.get('rmse_g', np.nan)
                        if rmse is not None and not np.isnan(rmse):
                            data.append({
                                'Severity': sev.capitalize(),
                                'Method': method,
                                'RMSE_g': float(rmse)
                            })
                            
    if data:
        df_box = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Severity', y='RMSE_g', hue='Method', data=df_box, palette='colorblind')
        plt.ylabel('Genomic Imputation RMSE')
        plt.title('Imputation Error Distribution across Folds (Independent Condition)')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/rmse_boxplot.png")
        plt.close()
        print("[FIGURES] Saved rmse_boxplot.png")

if __name__ == '__main__':
    main()
