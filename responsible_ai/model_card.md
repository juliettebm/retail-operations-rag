# Model Card - Assistant Operations (text-to-SQL)

A model card documenting the text-to-SQL assistant of this project, following the spirit of Mitchell et al. (2019). Written for transparency and governance.

---

## Model details

- **Task**: translate a natural-language operations question into a SQLite SELECT query, execute it, and narrate the result in 1-3 sentences for a non-technical business user (store manager, ops team).
- **Model**: `llama3.2` (3B), run locally via Ollama. OpenAI is a configurable alternative (`LLM_PROVIDER=openai`), not used for the results below.
- **Schema exposed to the model**: 4 tables (`magasins`, `commandes`, `retours`, `remboursements`), defined in `notebook/03_text_to_sql_operations.ipynb` (section 5). Joins between these four tables are explicitly allowed.

## Intended use

- **In scope**: ad-hoc operational questions over a retail dataset (delivery delays, return reasons, refund timing), as a demonstrator of a guarded text-to-SQL pattern. A tool to answer known-shape questions faster than writing SQL by hand, not to replace a BI dashboard for recurring reporting.
- **Out of scope**: any use on real customer or transaction data without re-validation; automated decisions taken from the assistant's answer without a human reading the underlying SQL result; production deployment without a larger or fine-tuned model, given the accuracy figures below.

## Data

- **Source**: entirely synthetic. `notebook/02_generation_donnees.ipynb` generates `data/operations.sqlite` from the business rules written in the operational guides (`data/*.md`): 30-day standard / 14-day sale-item return window, refund delay that depends on payment method, two return reasons reserved for online orders. No real Maison Kurt commande exists; the retailer itself is fictional.
- **Size**: 6,024 commandes (includes ~0.4% duplicate orders from a simulated confirmation-retry defect), 1,080 retours, 830 remboursements, across 8 magasins.
- **Deliberate signal**: the combination sale-item + home-delivery is generated with a markedly higher return rate than any other channel/type combination (near-double). Found and quantified in the notebook's bivariate analysis (section 4), and used as a realistic example in discussion.

## Guardrails

Read-only SQLite connection (`mode=ro`, the engine refuses any write regardless of what the generated query contains), structural validation of the generated SQL by `sqlglot` (single `SELECT`, only the four whitelisted tables, joins between them included; everything else rejected: `sqlite_master`, `ATTACH`, CTEs, multiple statements), a row cap and a query timeout. A regex-based fallback exists for the rare case `sqlglot` is unavailable, with the same table whitelist but no structural guarantee. Section 6 of the notebook tests both validators against 7 queries that must pass (joins included) and 11 that must be rejected, using the exact same functions the assistant runs in section 5, with no separate copy of the logic to drift out of sync.

**What the guardrails do not provide is correctness**, which is measured separately below.

## Execution accuracy

17 business questions, each with a hand-written reference SQL, checked against the actual database. The comparison (`resultats_egaux`) checks the full result, not just its first value: same number of rows and columns, numeric values equal within 0.01, text values compared after normalisation (case, whitespace). An earlier version of this comparison looked only at the first cell of the result, which unfairly failed any question whose correct answer spans several rows (a store ranking, for instance). That bug, not the model, was the main driver of a much lower accuracy figure in an earlier pass of this evaluation.

| | reussies | total | % reussi |
|---|---:|---:|---:|
| **Global** | 15 | 17 | **88.2%** |
| sans jointure | 10 | 11 | 90.9% |
| avec jointure | 5 | 6 | 83.3% |

The evaluation set was built with a deliberate mix of single-table and join questions to isolate this gap. Once the comparison bug above was fixed, the gap turned out to be modest (about 8 points), not the sharp drop a smaller, biased sample of the metric first suggested. With only 6 join questions, this gap is not a number to over-read: one question moving from fail to pass shifts it by roughly 17 points.

Of the 2 failures, neither was blocked by the guardrail (0 `refus_garde_fou`): the model always produced a query on an allowed table with no write and no disallowed construct. One (`ecart`, no join) computed an integer-division percentage that rounded to 0 instead of ~24%. One (`echec_execution`, join) referenced an ambiguous or misattributed column across the joined tables. The detailed table in the notebook (section 8) lists both with the generated SQL.

## Limitations

- **Small local model**: `llama3.2` (3B) was chosen for cost (free) and offline availability, not for accuracy.
- **Evaluation set size**: 17 questions, only 6 with a join, is enough to see that a join penalty exists, not enough to size it precisely: the confidence interval on the join-only accuracy is wide.
- **Guardrail scope**: the guardrail proves what cannot happen (no write, no table outside the whitelist, no multi-statement injection); it says nothing about whether the query answers the question asked. The two must not be conflated: a system can be entirely safe and still wrong, which is what happened on both failures here.
- **Narration step**: the assistant used in the demo (`ask`, section 7) drafts a final sentence from the SQL result; this step is not covered by the execution-accuracy evaluation above, which compares raw SQL results, generated via a separate function (`generate_sql`) that skips narration entirely. The demo notebook shows a concrete case of this gap: the narrated answer states "1 commande" directly below a SQL result of 1080. The SQL was correct, the sentence describing it was not.
- **Synthetic data**: every number in this card describes a generated dataset built to reproduce documented business rules, not a measurement of a real retailer's operations.
