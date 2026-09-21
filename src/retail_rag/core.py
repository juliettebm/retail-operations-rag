"""Pure, testable building blocks for the Maison Kurt RAG application."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REFUSAL = "Information non trouvée dans les documents."

PROMPT = (
    "Tu es un assistant d'analyse de documents opérationnels retail.\n\n"
    "Réponds à la question UNIQUEMENT à partir du contexte fourni.\n\n"
    "IMPORTANT :\n"
    "- Lis attentivement toutes les informations pertinentes du contexte.\n"
    "- Si le contexte indique qu'une règle ne s'applique PAS, "
    "réponds clairement que la règle ne s'applique pas.\n"
    "- Les formulations négatives comme « ne modifie pas », "
    "« ne signifie pas automatiquement » ou « ne peut pas » "
    "contiennent des informations importantes et doivent être utilisées "
    "pour répondre à la question.\n"
    "- Ne réponds jamais « Information non trouvée » si le contexte "
    "contient explicitement ou directement la réponse.\n\n"
    "Si aucune information permettant de répondre à la question "
    "n'est présente dans le contexte, réponds EXACTEMENT : "
    "« Information non trouvée dans les documents. »\n\n"
    "Sois concis et factuel.\n"
    "Toute réponse factuelle doit se terminer par un ou plusieurs identifiants "
    "de source exactement sous la forme [fichier.md#section-N]. "
    "N'invente jamais un identifiant.\n\n"
    "Contexte :\n{context}\n\n"
    "Question : {question}\n"
    "Réponse :"
)

CITATION_RE = re.compile(r"\[([\w.-]+\.md#section-\d+)\]")


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    heading: str
    text: str


def _slug_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def load_passages(data_dir: Path) -> list[Passage]:
    """Split Markdown on level-2 headings and assign stable passage identifiers."""
    passages: list[Passage] = []
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"(?m)^##\s+", text)[1:]
        for index, section in enumerate(sections, 1):
            heading, _, body = section.partition("\n")
            passages.append(Passage(
                id=f"{path.name}#section-{index}",
                source=path.name,
                heading=heading.strip(),
                text=f"{heading.strip()}\n{body.strip()}",
            ))
    return passages


def format_context(passages: Sequence[Passage]) -> str:
    return "\n\n".join(f"SOURCE [{p.id}]\n{p.text}" for p in passages)


def citations(answer: str) -> set[str]:
    return set(CITATION_RE.findall(answer))


def validate_citations(answer: str, passages: Sequence[Passage]) -> bool:
    """A refusal needs no citation; every factual answer needs a retrieved citation."""
    if answer.strip() == REFUSAL:
        return True
    allowed = {passage.id for passage in passages}
    found = citations(answer)
    return bool(found) and found <= allowed


def should_refuse(relevance_scores: Sequence[float], threshold: float) -> bool:
    """Deterministic scope gate based on the best calibrated retrieval score."""
    return not relevance_scores or max(relevance_scores) < threshold


def retrieval_metrics(
    ranked_ids: Iterable[Sequence[str]], relevant_ids: Iterable[set[str]], k: int
) -> dict[str, float]:
    """Compute macro Recall@k and MRR from explicit question/passage labels."""
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for ranking, relevant in zip(ranked_ids, relevant_ids, strict=True):
        if not relevant:
            continue
        top = list(ranking)[:k]
        recalls.append(len(set(top) & relevant) / len(relevant))
        reciprocal_ranks.append(next((1 / (i + 1) for i, pid in enumerate(top) if pid in relevant), 0.0))
    if not recalls:
        return {f"recall@{k}": 0.0, f"mrr@{k}": 0.0}
    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        f"mrr@{k}": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def deterministic_answer_check(answer: str, expected_terms: Sequence[str]) -> bool:
    """Independent, repeatable minimum-content check (not an LLM judge)."""
    normalized = _slug_text(answer)
    return all(_slug_text(term) in normalized for term in expected_terms)
