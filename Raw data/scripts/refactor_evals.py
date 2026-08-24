import os
import re

scripts = [
    "evaluate_survival.py",
    "evaluate_direct_survival.py",
    "evaluate_crossmodal.py",
    "evaluate_integration.py",
    "evaluate_pam50.py"
]

base_dir = "/Users/nazu.ds/Documents/Research Collections/MO-SVAE/Raw data/scripts"

for script in scripts:
    path = os.path.join(base_dir, script)
    with open(path, 'r') as f:
        content = f.read()

    # 1. Update RESULTS_DIR
    content = re.sub(r'RESULTS_DIR\s*=\s*f"\{BASE\}/results[^"]*"', r'RESULTS_DIR = f"{BASE}/results_final"', content)
    
    # Remove BASELINES_DIR, OUTPUT_DIR, etc. related to old logic if they exist, but we might break things if we remove too much.
    # Actually, better to just remove BASELINES_DIR.
    content = re.sub(r'BASELINES_DIR\s*=\s*f"\{BASE\}/results"\n', '', content)

    # 2. Update methods list
    methods_pattern = r'methods\s*=\s*\[.*?\]'
    new_methods = "methods = ['EF', 'CMA', 'PoE', 'MyeVAE', 'MNARSVAE', 'StandardVAE', 'MICE', 'missForest', 'GAIN', 'MIDA']"
    content = re.sub(methods_pattern, new_methods, content, flags=re.DOTALL)

    # 3. Update the get_method_dir logic and the scan_path / res_path logic
    # In some scripts, it's a def get_method_dir(method): ...
    content = re.sub(r'    def get_method_dir\(method\):.*?(?=    times_auc =|    all_summary =|    results =|    for method in methods:)', '', content, flags=re.DOTALL)

    # Update loop logic: `method_dir = get_method_dir(method)` -> `method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"`
    content = re.sub(r'        method_dir = get_method_dir\(method\)', r'        method_dir = f"{RESULTS_DIR}/{method}_{args.severity}_{args.condition}"', content)
    
    # Update res_path logic in the pre-scan
    # Replace the whole if-elif block for scan_path
    scan_path_regex = r'            if method == \'GAIN\':.*?scan_path = f"\{method_dir\}/fold_\{k_scan\}_results\.pkl"'
    content = re.sub(scan_path_regex, r'            scan_path = f"{method_dir}/fold_{k_scan}_results.pkl"', content, flags=re.DOTALL)
    
    # Update res_path logic in the actual loop
    res_path_regex = r'            if method == \'GAIN\':.*?res_path = f"\{method_dir\}/fold_\{k_}_results\.pkl"'
    # Note: might be k+1 or k depending on the script, so let's match the inner if-elif-else exactly.
    # Actually, it's safer to just regex replace the specific if-elif-else blocks.
    
    # Let's simplify: replace the if-elif-else for `scan_path` and `res_path` with just the standard path.
    # For scan_path:
    content = re.sub(
        r'( +)if method == \'GAIN\':\n\s+scan_path = .*?\n\s+else:\n\s+scan_path = f"\{method_dir\}/fold_\{([^}]+)\}_results\.pkl"',
        r'\1scan_path = f"{method_dir}/fold_{\2}_results.pkl"',
        content, flags=re.DOTALL
    )
    # For res_path:
    content = re.sub(
        r'( +)if method == \'GAIN\':\n\s+res_path = .*?\n\s+else:\n\s+res_path = f"\{method_dir\}/fold_\{([^}]+)\}_results\.pkl"',
        r'\1res_path = f"{method_dir}/fold_{\2}_results.pkl"',
        content, flags=re.DOTALL
    )

    # Sometimes they don't use method_dir, e.g. evaluate_direct_survival.py
    
    with open(path, 'w') as f:
        f.write(content)

