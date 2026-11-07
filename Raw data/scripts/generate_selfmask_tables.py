"""
generate_selfmask_tables.py

Reads selfmask evaluation summary CSVs from:
  <repo_root>/Raw data/evaluation_selfmask/
for sev in ['light', 'moderate', 'severe'].

Generates complete LaTeX table environments for:
  Table 1 (tab:supp_selfmask_survival): RSF C-index across severity levels.
  Table 2 (tab:supp_selfmask_downstream): XGB, DeepSurv, and DeepHit C-indices across severity levels.
  Table 3 (tab:supp_selfmask_imputation_g): Genomic expression imputation RMSE across severity levels.
  Table 4 (tab:supp_selfmask_imputation_c): Clinical feature imputation RMSE across severity levels.
  Table 5 (tab:supp_selfmask_ibs): IBS for RSF across severity levels.
  Table 6 (tab:supp_selfmask_crossmodal): Cross-modal imputation RMSE across severity levels.
"""

import math
import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_selfmask"))
SEVS = ["light", "moderate", "severe"]
METHODS = ["EF", "CMA", "PoE", "MyeVAE", "MICE", "missForest", "GAIN", "MIDA", "StandardVAE", "MNARSVAE"]
NAME_MAP = {
    "EF": "SIGMA EF",
    "CMA": "SIGMA CMA",
    "PoE": "SIGMA PoE",
    "MyeVAE": "MyeVAE",
    "MICE": "MICE",
    "missForest": "missForest",
    "GAIN": "GAIN",
    "MIDA": "MIDA",
    "StandardVAE": "StandardVAE",
    "MNARSVAE": "MNARSVAE",
}


def load_data():
    dfs = {}
    for sev in SEVS:
        csv_path = os.path.join(BASE_DIR, f"survival_{sev}_independent", "summary.csv")
        df = pd.read_csv(csv_path).set_index("method")
        dfs[sev] = df
    return dfs


def load_imputation_data():
    dfs = {}
    for sev in SEVS:
        csv_path = os.path.join(BASE_DIR, f"imputation_summary_{sev}_independent.csv")
        df = pd.read_csv(csv_path).set_index("method")
        dfs[sev] = df
    return dfs


def load_crossmodal_data():
    dfs = {}
    for sev in SEVS:
        csv_path = os.path.join(BASE_DIR, f"crossmodal_{sev}_independent", "summary.csv")
        df = pd.read_csv(csv_path).set_index("method")
        dfs[sev] = df
    return dfs


def format_cell(mean, std, ci, is_best=False):
    mean_val = round(float(mean), 3)
    std_val = round(float(std), 3)
    ci_val = round(float(ci), 3)
    lower = mean_val - ci_val
    upper = mean_val + ci_val

    cell_content = f"{mean_val:.3f} \\pm {std_val:.3f}\\ [{lower:.3f}, {upper:.3f}]"
    if is_best:
        return f"$\\mathbf{{{cell_content}}}$"
    return f"${cell_content}$"


def generate_table1(dfs):
    # Determine best (highest) mean per column
    best_methods = {}
    for sev in SEVS:
        best_methods[sev] = dfs[sev]["C_index_RSF_mean"].idxmax()

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\rowcolors{2}{gray!10}{white}")
    lines.append(
        r"\caption{RSF C-index under self-masking MNAR across all methods and severity levels (independent condition).}"
    )
    lines.append(r"\label{tab:supp_selfmask_survival}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Light C-index RSF & Moderate C-index RSF & Severe C-index RSF \\")
    lines.append(r"\midrule")

    for m in METHODS:
        disp_name = NAME_MAP[m]
        cells = []
        for sev in SEVS:
            mean = dfs[sev].loc[m, "C_index_RSF_mean"]
            std = dfs[sev].loc[m, "C_index_RSF_std"]
            ci = dfs[sev].loc[m, "C_index_RSF_ci"]
            is_best = m == best_methods[sev]
            cells.append(format_cell(mean, std, ci, is_best=is_best))
        row_str = f"{disp_name} & " + " & ".join(cells) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{1ex}")
    lines.append(
        r"\parbox{\linewidth}{\raggedright\footnotesize Note: Values are reported as mean $\pm$ SD [95\% CI]; CI computed as $\bar{x} \pm t_{0.025,4} \times s/\sqrt{5}$. Best values (higher C-index is better) are highlighted in bold. RSF: Random Survival Forest.\par}"
    )
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_table2(dfs):
    models = ["XGB", "DeepSurv", "DeepHit"]

    # Determine best (highest) mean per column (sev x model)
    best_methods = {}
    for sev in SEVS:
        for mdl in models:
            col = f"C_index_{mdl}_mean"
            best_methods[(sev, mdl)] = dfs[sev][col].idxmax()

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\rowcolors{2}{gray!10}{white}")
    lines.append(
        r"\caption{C-index for XGBoost (XGB), DeepSurv, and DeepHit under self-masking MNAR across all methods and severity levels (independent condition).}"
    )
    lines.append(r"\label{tab:supp_selfmask_downstream}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{Light} & \multicolumn{3}{c}{Moderate} & \multicolumn{3}{c}{Severe} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    lines.append(
        r"Method & XGB & DeepSurv & DeepHit & XGB & DeepSurv & DeepHit & XGB & DeepSurv & DeepHit \\"
    )
    lines.append(r"\midrule")

    for m in METHODS:
        disp_name = NAME_MAP[m]
        cells = []
        for sev in SEVS:
            for mdl in models:
                mean = dfs[sev].loc[m, f"C_index_{mdl}_mean"]
                std = dfs[sev].loc[m, f"C_index_{mdl}_std"]
                ci = dfs[sev].loc[m, f"C_index_{mdl}_ci"]
                is_best = m == best_methods[(sev, mdl)]
                cells.append(format_cell(mean, std, ci, is_best=is_best))
        row_str = f"{disp_name} & " + " & ".join(cells) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{1ex}")
    lines.append(
        r"\parbox{\linewidth}{\raggedright\footnotesize Note: Values are reported as mean $\pm$ SD [95\% CI]; CI computed as $\bar{x} \pm t_{0.025,4} \times s/\sqrt{5}$. Best values (higher C-index is better) are highlighted in bold. XGB: XGBoost; EF: Early Fusion; CMA: Cross-Modal Attention; PoE: Product of Experts; MICE: Multiple Imputation by Chained Equations; GAIN: Generative Adversarial Imputation Networks; MIDA: Multiple Imputation using Denoising Autoencoders.\par}"
    )
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_table3(imp_dfs):
    # Table 3: Genomic RMSE. Best = lowest
    best_methods = {}
    for sev in SEVS:
        best_methods[sev] = imp_dfs[sev]["RMSE_g_mean"].dropna().idxmin()

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\rowcolors{2}{gray!10}{white}")
    lines.append(
        r"\caption{Genomic expression imputation RMSE under self-masking MNAR across all methods and severity levels (independent condition).}"
    )
    lines.append(r"\label{tab:supp_selfmask_imputation_g}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Light RMSE & Moderate RMSE & Severe RMSE \\")
    lines.append(r"\midrule")

    for m in METHODS:
        disp_name = NAME_MAP[m]
        cells = []
        for sev in SEVS:
            if m in ["StandardVAE", "MNARSVAE"]:
                cells.append("---")
            else:
                mean = imp_dfs[sev].loc[m, "RMSE_g_mean"]
                ci = imp_dfs[sev].loc[m, "RMSE_g_ci"]
                std = float(ci) / 2.776 * math.sqrt(5)
                is_best = m == best_methods[sev]
                cells.append(format_cell(mean, std, ci, is_best=is_best))
        row_str = f"{disp_name} & " + " & ".join(cells) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{1ex}")
    lines.append(
        r"\parbox{\linewidth}{\raggedright\footnotesize Note: Values are reported as mean $\pm$ SD [95\% CI]; CI computed as $\bar{x} \pm t_{0.025,4} \times s/\sqrt{5}$. Best values (lower RMSE is better) are highlighted in bold. `---' indicates that StandardVAE and MNARSVAE do not model genomic data. RMSE: Root Mean Square Error; EF: Early Fusion; CMA: Cross-Modal Attention; PoE: Product of Experts; MICE: Multiple Imputation by Chained Equations; GAIN: Generative Adversarial Imputation Networks; MIDA: Multiple Imputation using Denoising Autoencoders.\par}"
    )
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_table4(imp_dfs):
    # Table 4: Clinical RMSE. Best = lowest
    best_methods = {}
    for sev in SEVS:
        best_methods[sev] = imp_dfs[sev]["RMSE_c_mean"].dropna().idxmin()

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\rowcolors{2}{gray!10}{white}")
    lines.append(
        r"\caption{Clinical feature imputation RMSE under self-masking MNAR across all methods and severity levels (independent condition).}"
    )
    lines.append(r"\label{tab:supp_selfmask_imputation_c}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Light RMSE & Moderate RMSE & Severe RMSE \\")
    lines.append(r"\midrule")

    for m in METHODS:
        disp_name = NAME_MAP[m]
        cells = []
        for sev in SEVS:
            mean = imp_dfs[sev].loc[m, "RMSE_c_mean"]
            ci = imp_dfs[sev].loc[m, "RMSE_c_ci"]
            std = float(ci) / 2.776 * math.sqrt(5)
            is_best = m == best_methods[sev]
            cells.append(format_cell(mean, std, ci, is_best=is_best))
        row_str = f"{disp_name} & " + " & ".join(cells) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{1ex}")
    lines.append(
        r"\parbox{\linewidth}{\raggedright\footnotesize Note: Values are reported as mean $\pm$ SD [95\% CI]; CI computed as $\bar{x} \pm t_{0.025,4} \times s/\sqrt{5}$. Best values (lower RMSE is better) are highlighted in bold. RMSE: Root Mean Square Error; EF: Early Fusion; CMA: Cross-Modal Attention; PoE: Product of Experts; MICE: Multiple Imputation by Chained Equations; GAIN: Generative Adversarial Imputation Networks; MIDA: Multiple Imputation using Denoising Autoencoders.\par}"
    )
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_table5(surv_dfs):
    # Table 5: IBS for RSF. Best = lowest
    best_methods = {}
    for sev in SEVS:
        best_methods[sev] = surv_dfs[sev]["IBS_RSF_mean"].dropna().idxmin()

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\rowcolors{2}{gray!10}{white}")
    lines.append(
        r"\caption{IBS for RSF under self-masking MNAR across all methods and severity levels (independent condition).}"
    )
    lines.append(r"\label{tab:supp_selfmask_ibs}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Light IBS & Moderate IBS & Severe IBS \\")
    lines.append(r"\midrule")

    for m in METHODS:
        disp_name = NAME_MAP[m]
        cells = []
        for sev in SEVS:
            mean = surv_dfs[sev].loc[m, "IBS_RSF_mean"]
            std = surv_dfs[sev].loc[m, "IBS_RSF_std"]
            ci = surv_dfs[sev].loc[m, "IBS_RSF_ci"]
            is_best = m == best_methods[sev]
            cells.append(format_cell(mean, std, ci, is_best=is_best))
        row_str = f"{disp_name} & " + " & ".join(cells) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{1ex}")
    lines.append(
        r"\parbox{\linewidth}{\raggedright\footnotesize Note: Values are reported as mean $\pm$ SD [95\% CI]; CI computed as $\bar{x} \pm t_{0.025,4} \times s/\sqrt{5}$. Best values (lower IBS is better) are highlighted in bold. RSF: Random Survival Forest; IBS: Integrated Brier Score; EF: Early Fusion; CMA: Cross-Modal Attention; PoE: Product of Experts; MICE: Multiple Imputation by Chained Equations; GAIN: Generative Adversarial Imputation Networks; MIDA: Multiple Imputation using Denoising Autoencoders.\par}"
    )
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_table6(cross_dfs):
    # Table 6: Crossmodal RMSE. Best = lowest. EF, CMA, PoE only.
    cross_methods = ["EF", "CMA", "PoE"]

    best_methods = {}
    for sev in SEVS:
        best_methods[(sev, "C2G")] = cross_dfs[sev]["C2G_RMSE_mean"].dropna().idxmin()
        best_methods[(sev, "G2C")] = cross_dfs[sev]["G2C_RMSE_mean"].dropna().idxmin()

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\rowcolors{2}{gray!10}{white}")
    lines.append(
        r"\caption{Cross-modal imputation RMSE under self-masking MNAR (independent condition). C2G: clinical-to-genomic; G2C: genomic-to-clinical.}"
    )
    lines.append(r"\label{tab:supp_selfmask_crossmodal}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Light C2G & Light G2C & Moderate C2G & Moderate G2C & Severe C2G & Severe G2C \\"
    )
    lines.append(r"\midrule")

    for m in cross_methods:
        disp_name = NAME_MAP[m]
        cells = []
        for sev in SEVS:
            # C2G
            mean_c2g = cross_dfs[sev].loc[m, "C2G_RMSE_mean"]
            std_c2g = cross_dfs[sev].loc[m, "C2G_RMSE_std"]
            ci_c2g = cross_dfs[sev].loc[m, "C2G_RMSE_ci"]
            is_best_c2g = m == best_methods[(sev, "C2G")]
            cells.append(format_cell(mean_c2g, std_c2g, ci_c2g, is_best=is_best_c2g))

            # G2C
            mean_g2c = cross_dfs[sev].loc[m, "G2C_RMSE_mean"]
            if pd.isna(mean_g2c):
                cells.append("---")
            else:
                std_g2c = cross_dfs[sev].loc[m, "G2C_RMSE_std"]
                ci_g2c = cross_dfs[sev].loc[m, "G2C_RMSE_ci"]
                is_best_g2c = m == best_methods[(sev, "G2C")]
                cells.append(format_cell(mean_g2c, std_g2c, ci_g2c, is_best=is_best_g2c))

        row_str = f"{disp_name} & " + " & ".join(cells) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{1ex}")
    lines.append(
        r"\parbox{\linewidth}{\raggedright\footnotesize Note: Values are reported as mean $\pm$ SD [95\% CI]; CI computed as $\bar{x} \pm t_{0.025,4} \times s/\sqrt{5}$. Best values (lower RMSE is better) are highlighted in bold. `---' indicates that PoE G2C cross-modal reconstruction is not supported by design. RMSE: Root Mean Square Error; C2G: clinical-to-genomic; G2C: genomic-to-clinical; EF: Early Fusion; CMA: Cross-Modal Attention; PoE: Product of Experts.\par}"
    )
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    surv_dfs = load_data()
    imp_dfs = load_imputation_data()
    cross_dfs = load_crossmodal_data()

    table1 = generate_table1(surv_dfs)
    table2 = generate_table2(surv_dfs)
    table3 = generate_table3(imp_dfs)
    table4 = generate_table4(imp_dfs)
    table5 = generate_table5(surv_dfs)
    table6 = generate_table6(cross_dfs)

    print(table1)
    print("\n")
    print(table2)
    print("\n")
    print(table3)
    print("\n")
    print(table4)
    print("\n")
    print(table5)
    print("\n")
    print(table6)


if __name__ == "__main__":
    main()
