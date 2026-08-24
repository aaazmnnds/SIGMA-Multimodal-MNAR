import os
import re

scripts = [
    "evaluate_survival.py",
    "evaluate_direct_survival.py",
    "evaluate_crossmodal.py",
    "evaluate_integration.py",
    "evaluate_pam50.py"
]

methods_str = "['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']"

for s in scripts:
    with open(s, 'r') as f:
        content = f.read()

    # 1. Update RESULTS_DIR and remove old dirs
    content = re.sub(r'RESULTS_DIR = f"\{BASE\}/results_optuna"', r'RESULTS_DIR = f"{BASE}/results_final"', content)
    content = re.sub(r'    BASELINES_DIR = f"\{BASE\}/results"\n', '', content)
    content = re.sub(r'    SIMULATION_DIR = f"\{BASE\}/mnar_simulation"\n', '', content) # just clean up if needed

    # Fix indentation errors if any
    content = content.replace('\n    \n', '\n')
    
    # In evaluate_survival.py, we have `methods = [...]`
    # Replace the whole list
    content = re.sub(r'    methods = \[\n.*?\]', f'    methods = {methods_str}', content, flags=re.DOTALL)
    
    # Replace `fusions = ['EF', 'CMA', 'PoE']`
    content = re.sub(r'    fusions = \[\'EF\', \'CMA\', \'PoE\'\]', f'    methods = {methods_str}', content)
    content = re.sub(r'for fusion in fusions:', r'for method in methods:', content)
    
    # 2. Update the get_method_dir
    content = re.sub(r'    def get_method_dir\(method\):.*?(?=    times_auc =)', '', content, flags=re.DOTALL)
    content = re.sub(r'    def get_method_dir\(method\):.*?(?=    all_summary =)', '', content, flags=re.DOTALL)
    
    # 3. Path construction
    content = re.sub(r'method_dir = f"\{RESULTS_DIR\}/\{fusion\}_\{args.severity\}_\{args.condition\}"', r'method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"', content)
    content = re.sub(r'model_path = f"\{method_dir\}/fold_\{k\}_model.pt"', r'model_path = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}/fold_{k}_model.pt"', content)
    content = re.sub(r'res_path = f"\{method_dir\}/fold_\{k\}_results.pkl"', r'res_path = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}/fold_{k}_results.pkl"', content)
    content = re.sub(r'params_path = f"\{method_dir\}/fold_\{k\}_best_params.json"', r'params_path = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}/fold_{k}_best_params.json"', content)
    content = re.sub(r'res_path = f"\{RESULTS_DIR\}/\{fusion\}_\{args.severity\}_\{args.condition\}/fold_\{k\+1\}_results.pkl"', r'res_path = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}/fold_{k+1}_results.pkl"', content)
    content = re.sub(r'res_path = f"\{RESULTS_DIR\}/\{fusion\}_severe_\{args.condition\}/fold_\{k\+1\}_results.pkl"', r'res_path = f"{RESULTS_DIR}/{method}_severe_{args.condition}/fold_{k+1}_results.pkl"', content)
    
    # evaluate_survival specific
    content = re.sub(r'        method_dir = f"\{RESULTS_DIR\}/\{method\}_\{args.severity\}_\{args.condition\}"\n        if not os.path.exists\(method_dir\):\n            print\(f"Warning: \{method_dir\} does not exist. Skipping \{method\}."\)\n            continue\n\n        # Initialize full arrays using first available fold result\n        full_x_c = None\n        full_x_g = None\n        \n        # Pre-scan to get shapes\n        for k_scan in range\(1, 6\):\n            scan_path = f"\{method_dir\}/fold_\{k_scan\}_results.pkl"\n', r'        method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"\n        if not os.path.exists(method_dir):\n            print(f"Warning: {method_dir} does not exist. Skipping {method}.")\n            continue\n\n        # Initialize full arrays using first available fold result\n        full_x_c = None\n        full_x_g = None\n        \n        # Pre-scan to get shapes\n        for k_scan in range(1, 6):\n            scan_path = f"{method_dir}/fold_{k_scan}_results.pkl"\n', content)
    
    # Actually, evaluate_survival had:
    content = re.sub(r'            scan_path = f"\{method_dir\}/fold_\{k_scan\}_results.pkl"', r'            scan_path = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}/fold_{k_scan}_results.pkl"', content)
    content = re.sub(r'            res_path = f"\{method_dir\}/fold_\{k\}_results.pkl"', r'            res_path = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}/fold_{k}_results.pkl"', content)
    
    # Also replace variable 'fusion' inside evaluate_crossmodal and evaluate_integration
    content = re.sub(r'model = SIGMA\(p_c, p_g, d, fusion\)', r'model = SIGMA(p_c, p_g, d, method)', content)
    
    # For evaluate_direct_survival, summary_data.append changes
    content = re.sub(r'\'Fusion\': fusion,', r'\'Method\': method,', content)
    content = re.sub(r'print\(f"\[.*?\] \{fusion\} \|', r'print(f"[\g<1>] {method} |', content) # This might be tricky, let's just do exact
    content = re.sub(r'\[DIRECT_SURVIVAL\] \{fusion\}', r'[DIRECT_SURVIVAL] {method}', content)
    content = re.sub(r'Summary \| \{fusion\}', r'Summary | {method}', content)
    content = re.sub(r'/\{fusion\}_direct_survival.csv', r'/{method}_direct_survival.csv', content)
    
    content = re.sub(r'\[CROSSMODAL\] \{fusion\}', r'[CROSSMODAL] {method}', content)
    content = re.sub(r'Summary \| \{fusion\}', r'Summary | {method}', content)
    content = re.sub(r'/\{fusion\}_crossmodal.csv', r'/{method}_crossmodal.csv', content)
    
    content = re.sub(r'\[INTEGRATION\] \{fusion\}', r'[INTEGRATION] {method}', content)
    content = re.sub(r'Summary \| \{fusion\}', r'Summary | {method}', content)
    content = re.sub(r'/\{fusion\}_integration.csv', r'/{method}_integration.csv', content)
    
    content = re.sub(r'\[PAM50\] \{fusion\}', r'[PAM50] {method}', content)
    content = re.sub(r'Summary \| \{fusion\}', r'Summary | {method}', content)
    content = re.sub(r'/\{fusion\}_pam50.csv', r'/{method}_pam50.csv', content)

    # evaluate_survival.py uses method instead of fusion already
    content = re.sub(r'\[SURVIVAL\] \{method\}', r'[SURVIVAL] {method}', content)

    with open(s, 'w') as f:
        f.write(content)

