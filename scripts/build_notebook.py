"""Build and execute notebooks/replication.ipynb from the saved results.

    python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Hidden unfairness in clinical risk models\n\n"
       "This notebook walks through the replication of Kearns, Neel, Roth & Wu (2018) using the `subgroup_audit` "
       "package in `src/`. The heavy lifting (training ~40 models per dataset) is done by "
       "`scripts/run_experiments.py`; here we load its saved results and re-run the parts that are quick."),
    code("import sys, json\nsys.path.insert(0, '../src')\nimport numpy as np, pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from subgroup_audit import audit as A\n"
         "R = {n: json.load(open(f'../results/{n}.json')) for n in ['meps', 'diabetes', 'communities', 'adult']}"),
    md("## 1. The idea in one toy example\n\nFour cells of 400 true negatives each. Single attributes look "
       "perfectly fair; every intersection is off by 15 points."),
    code("rows = []\nfor sex in ['Male', 'Female']:\n    for race in ['White', 'Black']:\n"
         "        fpr = 0.4 if (sex == 'Male') == (race == 'Black') else 0.1\n"
         "        rows += [{'sex': sex, 'race': race, 'y': 0, 'p': int(i < fpr * 400)} for i in range(400)]\n"
         "toy = pd.DataFrame(rows)\n"
         "res = A.audit(toy.y, toy.p, toy[['sex', 'race']], 'fpr', max_order=2)\n"
         "pd.DataFrame(res['single_attribute'] + res['top_raw_gap'][:4])[['subgroup', 'rate', 'gap', 'weighted_gap']]"),
    md("## 2. Baseline models: single attributes vs. intersections (test set)"),
    code("def headline(name):\n    out = []\n    for k, m in R[name]['models'].items():\n"
         "        out.append({'model': k, 'accuracy': m['overall']['accuracy'],\n"
         "                    'worst single': f\"{m['worst_single']['subgroup']} ({100*m['worst_single']['gap']:+.1f} pp)\",\n"
         "                    'worst intersection': f\"{m['worst_intersection']['subgroup']} ({100*m['worst_intersection']['gap']:+.1f} pp, n={m['worst_intersection']['base_size']})\",\n"
         "                    'max weighted gap': m['max_weighted_gap'], 'train max weighted gap': m['train_max_weighted_gap']})\n"
         "    return pd.DataFrame(out).round(4)\n"
         "for n in R:\n    print(n, R[n]['metric'].upper()); display(headline(n))"),
    md("## 3. The gerrymandering effect\n\nA model trained to be fair on each attribute separately, audited "
       "on intersections."),
    code("for n in R:\n    display(Image(f'../figures/{n}_gerrymandering.png'))\n    display(Image(f'../figures/{n}_worst_gaps.png'))"),
    md("## 4. Accuracy–fairness trade-off"),
    code("for n in R:\n    display(Image(f'../figures/{n}_tradeoff.png'))\n"
         "    display(pd.DataFrame(R[n]['tradeoff']['subgroup_fair']).round(4))"),
    md("## 5. How clinical risk scores are actually used\n\nA care-management programme enrols the top 20% of "
       "patients by risk. Who is missed, are scores calibrated across subgroups, and (MEPS) do patients with the "
       "same score have the same level of illness across races?"),
    code("for n in ['meps', 'diabetes']:\n    c = R[n]['clinical']\n"
         "    print(n, 'worst missed (FNR) at top-20%:', c['capacity_fnr']['worst_intersection']['subgroup'],\n"
         "          round(100 * c['capacity_fnr']['worst_intersection']['gap'], 1), 'pp')\n"
         "    display(Image(f'../figures/{n}_capacity_missed.png'))\n"
         "    display(Image(f'../figures/{n}_calibration.png'))\n"
         "display(Image('../figures/meps_need_by_score.png'))\n"
         "display(Image('../figures/meps_need_missed.png'))\n"
         "pd.DataFrame(R['meps']['clinical']['need_by_score']['mean_conditions'], index=range(1, 11)).round(2)"),
    md("## 6. Paper setting and the reference implementation\n\nLinear-threshold subgroups over the 18 "
       "protected attributes used in the authors' GerryFair repository, and the effect of the indexing issue "
       "in its cost update (our port; see README)."),
    code("display(Image('../figures/communities_paper_tradeoff.png'))\n"
         "display(Image('../figures/communities_convergence.png'))\n"
         "pd.DataFrame({k: {a: b for a, b in v.items() if a != 'history'} for k, v in R['communities']['paper']['bug'].items()}).round(4)"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
nbf.write(nb, ROOT / "notebooks" / "replication.ipynb")
print("wrote notebooks/replication.ipynb")
