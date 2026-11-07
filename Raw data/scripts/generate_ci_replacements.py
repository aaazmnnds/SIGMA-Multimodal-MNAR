"""
generate_ci_replacements.py

Generates structured full-row LaTeX replacements for manuscript tables:
Reverts the current mean ± CI half-width row back to mean ± SD, followed by
a second line displaying the 95% CI intervals [lower, upper].

OLD: current manuscript row (with CI half-width)
NEW: two rows:
     original SD row
      & [lower, upper] ... \\

Bounds: lower = mean - ci, upper = mean + ci, where ci = 2.776 * std / sqrt(5)
Format: 3 decimal places for all values.
Preserves bolding in SD row, unbolded in CI row.
Also generates footnote updates at the end.
"""

import os
import re

TEX_PATH = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/BMC_Bioinformatics_Submission/sn_article_sigma.tex"
REPL_PATH = "/tmp/ci_replacements.txt"
PM_PATTERN = re.compile(r"(\$(?:\\mathbf\{)?)([\d\.\-]+)\s*\\pm\s*([\d\.\-]+)\}?\s*\$")


def load_rules(repl_path):
    table_blocks = {}
    curr = None
    with open(repl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# tab:"):
                curr = line[2:].strip()
                table_blocks[curr] = []
            elif line.startswith("#"):
                curr = None
            elif curr and "|||" in line:
                table_blocks[curr].append(line.split("|||"))
    return table_blocks


def main():
    table_blocks = load_rules(REPL_PATH)

    with open(TEX_PATH, "r", encoding="utf-8") as f:
        tex_content = f.read()

    labels = [
        "tab:survival",
        "tab:direct_survival",
        "tab:survival_ibs",
        "tab:supp_survival_downstream",
        "tab:integration",
        "tab:crossmodal",
        "tab:pam50",
        "tab:tcga_external",
        "tab:imputation_g",
        "tab:imputation_c",
        "tab:supp_mae_g",
        "tab:supp_mae_c",
        "tab:ablation",
    ]

    for lab in labels:
        print(f"# {lab}")
        pos = tex_content.find(f"\\label{{{lab}}}")
        if pos == -1:
            continue
        start = tex_content.rfind("\\begin{table", 0, pos)
        end = tex_content.find("\\end{table", pos)
        end = tex_content.find("}", end) + 1
        t_block = tex_content[start:end]

        rows = [
            l
            for l in t_block.splitlines()
            if "&" in l
            and (r"\pm" in l or "---" in l)
            and not any(k in l for k in ["multicolumn", "Light", "Metric", "Method", "Fusion"])
        ]

        rules = table_blocks[lab]
        rule_idx = 0

        for r in rows:
            tokens = [x.strip() for x in r.split("&")]
            lead = tokens[0]

            # In tab:ablation, only C-index RSF and PAM50 Acc have rules
            if lab == "tab:ablation" and ("RMSE" in lead):
                continue

            cells = tokens[1:]
            cells[-1] = re.sub(r"\\\\$", "", cells[-1]).strip()

            sd_cells = []
            ci_cells = []

            for c in cells:
                if "---" in c:
                    sd_cells.append("---")
                    ci_cells.append("---")
                else:
                    old_sd, old_ci = rules[rule_idx]
                    rule_idx += 1

                    m_sd = PM_PATTERN.search(old_sd)
                    m_ci = PM_PATTERN.search(old_ci)

                    mean_val = float(m_sd.group(2))
                    sd_val = float(m_sd.group(3))
                    ci_val = float(m_ci.group(3))

                    is_bold = "\\mathbf" in c
                    if is_bold:
                        sd_str = f"$\\mathbf{{{mean_val:.3f} \\pm {sd_val:.3f}}}$"
                    else:
                        sd_str = f"${mean_val:.3f} \\pm {sd_val:.3f}$"

                    lower = mean_val - ci_val
                    upper = mean_val + ci_val
                    ci_str = f"$[{lower:.3f}, {upper:.3f}]$"

                    sd_cells.append(sd_str)
                    ci_cells.append(ci_str)

            row_sd = lead + " & " + " & ".join(sd_cells) + r" \\"
            row_ci = " & " + " & ".join(ci_cells) + r" \\"
            row_new = row_sd + "\n" + row_ci
            print(f"{r}|||{row_new}")

    # Footnote updates
    print("\n# footnote updates")
    print(
        "mean $\\pm$ 95\\% confidence interval (CI) across 5 cross-validation folds|||mean $\\pm$ standard deviation (SD) across 5 cross-validation folds; bracketed values show 95\\% confidence intervals computed as $\\bar{x} \\pm t_{0.025,4} \\times s/\\sqrt{5}$"
    )
    print(
        "mean $\\pm$ 95\\% confidence interval (CI)|||mean $\\pm$ standard deviation (SD); bracketed values show 95\\% confidence intervals computed as $\\bar{x} \\pm t_{0.025,4} \\times s/\\sqrt{5}$"
    )


if __name__ == "__main__":
    main()
