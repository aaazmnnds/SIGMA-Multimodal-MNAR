import re

with open('main.tex', 'r') as f:
    text = f.read()

with open('references.bib', 'r') as f:
    bib = f.read()

# 1. Abbreviations defined at first use
abbrevs = ['MNAR', 'MAR', 'MCAR', 'MICE', 'VAE', 'RSF', 'SIGMA', 'EF', 'CMA', 'PoE', 'METABRIC', 'TCGA', 'BRCA', 'PAM50', 'GISTIC', 'ER', 'HER2', 'PR', 'IBS', 'CCA', 'RMSE', 'MAE', 'C-index']
for abbrev in abbrevs:
    # Find all occurrences
    matches = list(re.finditer(r'\b' + abbrev + r'\b', text))
    if matches:
        first_idx = matches[0].start()
        # Look at the text before the first occurrence to see if it's defined like (ABBREV) or ABBREV (definition)
        # We can just extract the 50 chars before and after the first occurrence
        context = text[max(0, first_idx-100):min(len(text), first_idx+100)]
        print(f"[{abbrev}] First appearance: {context.strip()}")
    else:
        print(f"[{abbrev}] NOT FOUND in main.tex")

# 2 & 3. Table and Fig refs
table_refs = set(re.findall(r'Table~\\ref\{([^}]+)\}', text))
table_labels = set(re.findall(r'\\label\{tab:([^}]+)\}', text)) # Actually let's just get the full label string
table_labels = set(re.findall(r'\\label\{([^}]+)\}', text))

for ref in table_refs:
    if ref not in table_labels:
        print(f"Table ref missing label: {ref}")

fig_refs = set(re.findall(r'Figure~\\ref\{([^}]+)\}', text))
for ref in fig_refs:
    if ref not in table_labels:
        print(f"Fig ref missing label: {ref}")

# 4. Citations
citations = set()
for match in re.findall(r'\\cite[pt]\{([^}]+)\}', text):
    for cite in match.split(','):
        citations.add(cite.strip())

bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
for cite in citations:
    if cite not in bib_keys:
        print(f"Missing citation in bib: {cite}")

# 6. Em dashes
if '---' in text:
    print("Found '---'")
if '—' in text:
    print("Found '—'")

