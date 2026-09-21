"""Keep committed metrics and README claims synchronized."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / "evaluation_results.json").read_text(encoding="utf-8"))
readme = (ROOT / "README.md").read_text(encoding="utf-8")
metric_k = results["k"]

claims = [
    f"{results['in_scope']} questions in-scope",
    f"Recall@{metric_k} = {results[f'recall@{metric_k}']:.0%}".replace("%", " %"),
    f"MRR@{metric_k} = {results[f'mrr@{metric_k}']:.1%}".replace(".", ",").replace("%", " %"),
]
missing = [claim for claim in claims if claim not in readme]
if missing:
    raise SystemExit("README out of sync: " + ", ".join(missing))
print("README and evaluation_results.json are synchronized.")
