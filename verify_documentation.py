"""Keep committed metrics and README claims synchronized."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / "evaluation_results.json").read_text(encoding="utf-8"))
readme = (ROOT / "README.md").read_text(encoding="utf-8")


def pct(value: float) -> str:
    return f"{value:.1%}".replace(".", ",").replace("%", " %")


claims = [
    f"{results['in_scope']} questions in-scope",
    f"{results['hard_in_scope']} questions difficiles",
]
k5 = results["k"]
for name, key in (("Recall", "recall"), ("MRR", "mrr")):
    cells = " | ".join(pct(results[f"{key}@{k}"]) for k in (1, 2, k5))
    claims.append(f"| {name}@k | {cells} |")
claims.append(f"MRR@1 = {pct(results['mrr@1'])}")
claims.append(f"MRR@1 = {pct(results['hard_mrr@1'])}")
missing = [claim for claim in claims if claim not in readme]
if missing:
    raise SystemExit("README out of sync: " + ", ".join(missing))
print("README and evaluation_results.json are synchronized.")
