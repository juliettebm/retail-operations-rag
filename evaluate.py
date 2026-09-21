"""Run reproducible retrieval and end-to-end evaluations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from retail_rag.core import load_passages, retrieval_metrics

ROOT = Path(__file__).resolve().parent
APP_K = 2  # number of passages app.py sends to the LLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--check", type=Path, help="Fail if metrics differ from this result file")
    args = parser.parse_args()

    dataset = json.loads((ROOT / "evaluation.json").read_text(encoding="utf-8"))
    passages = load_passages(ROOT / "data")
    model = SentenceTransformer(args.model)
    passage_vectors = model.encode(
        [f"passage: {p.text}" for p in passages], normalize_embeddings=True
    )
    index = faiss.IndexFlatIP(passage_vectors.shape[1])
    index.add(passage_vectors)
    scoped = [item for item in dataset if item["in_scope"]]
    query_vectors = model.encode(
        [f"query: {item['question']}" for item in scoped], normalize_embeddings=True
    )
    _, indices = index.search(query_vectors, args.k)
    rankings = [[passages[i].id for i in row] for row in indices]
    relevant = [set(item["relevant_passage_ids"]) for item in scoped]
    hard = [i for i, item in enumerate(scoped) if item.get("level") == "hard"]
    report = {
        "dataset_size": len(dataset),
        "in_scope": len(scoped),
        "out_of_scope": len(dataset) - len(scoped),
        "hard_in_scope": len(hard),
        "embedding_model": args.model,
        "k": args.k,
    }
    # k=1 is the most discriminating cutoff, k=APP_K is what the app sends to the LLM.
    for cutoff in sorted({1, APP_K, args.k}):
        report.update(retrieval_metrics(rankings, relevant, cutoff))
        report.update({
            f"hard_{key}": value
            for key, value in retrieval_metrics(
                [rankings[i] for i in hard], [relevant[i] for i in hard], cutoff
            ).items()
        })
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        for key, value in report.items():
            if key not in expected:
                raise SystemExit(f"Missing key in {args.check}: {key}")
            if isinstance(value, float):
                if not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12):
                    raise SystemExit(f"Metric drift for {key}: {expected[key]} -> {value}")
            elif value != expected[key]:
                raise SystemExit(f"Result drift for {key}: {expected[key]} -> {value}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
