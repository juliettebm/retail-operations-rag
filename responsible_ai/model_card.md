# Model Card - Assistant Operations (text-to-SQL)

A model card documenting the text-to-SQL assistant of this project, following the spirit of Mitchell et al. (2019). Written for transparency and governance.

---

## Model details

- **Task**: translate a natural-language operations question into a SQLite SELECT query, execute it, and narrate the result in 1-3 sentences for a non-technical business user (store manager, ops team).
- **Model**: `llama3.2` (3B), run locally via Ollama. OpenAI is a configurable alternative (`LLM_PROVIDER=openai`), not used for the results below.
- **Schema exposed to the model**: 4 tables (`magasins`, `commandes`, `retours`, `remboursements`), described in `src/02_genai_assistant.py::SCHEMA`. Unlike the sibling project this module is ported from (`pharma-commercial-genai`, single table, no join), joins between these four tables are explicitly allowed.

## Intended use

- **In scope**: ad-hoc operational questions over a retail dataset (delivery delays, return reasons, refund timing), as a demonstrator of a guarded text-to-SQL pattern. A tool to answer known-shape questions faster than writing SQL by hand, not to replace a BI dashboard for recurring reporting.
- **Out of scope**: any use on real customer or transaction data without re-validation; automated decisions taken from the assistant's answer without a human reading the underlying SQL result; production deployment without a larger or fine-tuned model, given the accuracy figures below.

## Data

- **Source**: entirely synthetic. `src/01_prepare_data.py` generates `data/operations.sqlite` from the business rules written in the operational guides (`data/*.md`) — 30-day standard / 14-day sale-item return window, refund delay that depends on payment method, two return reasons reserved for online orders. No real Maison Kurt commande exists; the retailer itself is fictional.
- **Size**: 6,000 commandes, 1,078 retours, 823 remboursements, across 8 magasins.
- **Deliberate signal**: one store (Marseille Prado) is generated with a structurally slower home-delivery process; the combination sale-item + home-delivery is generated with a markedly higher return rate than any other channel/type combination. Both are found and quantified in the notebook's bivariate analysis (section 3), and are used as realistic examples for the assistant's demonstration questions.

## Guardrails

Read-only SQLite connection (`mode=ro`, the engine refuses any write regardless of what the generated query contains), structural validation of the generated SQL by `sqlglot` (single `SELECT`, only the four whitelisted tables — joins between them included, everything else rejected: `sqlite_master`, `ATTACH`, CTEs, multiple statements), a row cap and a query timeout. A regex-based fallback exists for the rare case `sqlglot` is unavailable, with the same table whitelist but no structural guarantee. `test_garde_fous.py` checks both validators against 7 queries that must pass (joins included) and 11 that must be rejected, and the notebook (`notebook/02_text_to_sql_operations.ipynb`) re-runs this file rather than trusting a private copy of the logic. The notebook and `test_garde_fous.py` both import the same module, so what one verifies is exactly what the other runs.

**What the guardrails do not provide is correctness**, which is measured separately below.

## Execution accuracy

18 business questions, each with a hand-written reference SQL, checked against the actual database rather than compared as text (execution accuracy: two syntactically different queries that return the same value both count as a pass).

| | reussies | total | % reussi |
|---|---:|---:|---:|
| **Global** | 12 | 18 | **66.7%** |
| sans jointure | 9 | 11 | 81.8% |
| avec jointure | 3 | 7 | 42.9% |

The join/no-join split was the reason this evaluation set was built with roughly equal numbers of each: `pharma-commercial-genai` measured 57.9% (11/19) on a single-table, no-join schema, and the open question was whether allowing joins would cost accuracy. It does, close to doubling the failure rate.

Of the 6 failures, none were blocked by the guardrail (0 `refus_garde_fou`): the model always produced a query on an allowed table with no write and no disallowed construct. 4 failed to execute (`echec_execution` — a nonexistent or misattributed column, most often a join column read from the wrong table), and 2 executed but returned the wrong value (`ecart`). The detailed table in the notebook (section 6) lists all 6 with the generated SQL. The clearest failure mode: on 3 of the 7 join questions, the model referenced a column that belongs to a different table in the join than the one it queried (for example filtering `retours` on a `canal` column that only exists in `commandes`) — a schema-attribution mistake that a single-table query cannot make by construction.

## Limitations

- **Small local model**: `llama3.2` (3B) was chosen for cost (free) and offline availability, not for accuracy. A larger model would likely narrow the join/no-join gap; this was not tested here.
- **Evaluation set size**: 18 questions is enough to see the join effect (a near-doubling of the failure rate, consistent across the two runs used to build this card), not enough for a tight confidence interval on the exact percentage.
- **Guardrail scope**: the guardrail proves what cannot happen (no write, no table outside the whitelist, no multi-statement injection); it says nothing about whether the query answers the question asked. The two must not be conflated — a system can be entirely safe and still wrong, which is exactly what happened on all 6 failures here.
- **Narration step**: the model also drafts the final sentence from the SQL result. In manual testing outside the formal evaluation, this step occasionally misread its own correct SQL result (for instance answering "no orders in 2025" immediately after printing a result of 6,000) — a failure the execution-accuracy metric above does not capture, because it compares the raw SQL result, not the narrated sentence.
- **Synthetic data**: every number in this card describes a generated dataset built to reproduce documented business rules, not a measurement of a real retailer's operations.
