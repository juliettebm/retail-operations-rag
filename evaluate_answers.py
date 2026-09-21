"""Score saved answers without using an LLM as a judge.

Input is a JSON list of {"id": ..., "answer": ..., "retrieved_passage_ids": [...]}. 
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_rag.core import Passage, REFUSAL, deterministic_answer_check, validate_citations

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path)
    args = parser.parse_args()
    labels = {x["id"]: x for x in json.loads((ROOT / "evaluation.json").read_text(encoding="utf-8"))}
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    rows = []
    for result in answers:
        label = labels[result["id"]]
        answer = result["answer"].strip()
        retrieved = [Passage(pid, pid.split("#")[0], "", "") for pid in result["retrieved_passage_ids"]]
        rows.append({
            "id": result["id"],
            "scope_correct": (answer != REFUSAL) == label["in_scope"],
            "minimum_content_correct": (
                deterministic_answer_check(answer, label["expected_terms"])
                if label["in_scope"] else answer == REFUSAL
            ),
            "citations_valid": validate_citations(answer, retrieved),
        })
    totals = {key: sum(row[key] for row in rows) / len(rows) for key in
              ("scope_correct", "minimum_content_correct", "citations_valid")}
    print(json.dumps({"count": len(rows), "rates": totals, "details": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
