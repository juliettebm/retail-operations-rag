from pathlib import Path

from retail_rag.core import (
    REFUSAL,
    Passage,
    deterministic_answer_check,
    load_passages,
    retrieval_metrics,
    should_refuse,
    validate_citations,
)


def test_passage_ids_are_stable_and_unique():
    passages = load_passages(Path("data"))
    ids = [passage.id for passage in passages]
    assert len(passages) == 26
    assert len(ids) == len(set(ids))
    assert "guide_retours_echanges.md#section-1" in ids


def test_retrieval_metrics_use_all_relevant_passages():
    metrics = retrieval_metrics(
        [["a", "b", "c"], ["x", "y", "z"]],
        [{"a", "c"}, {"y"}],
        k=2,
    )
    assert metrics == {"recall@2": 0.75, "mrr@2": 0.75}


def test_citations_must_be_present_and_retrieved():
    passages = [Passage("guide.md#section-1", "guide.md", "Titre", "Texte")]
    assert validate_citations("Réponse [guide.md#section-1]", passages)
    assert not validate_citations("Réponse sans source", passages)
    assert not validate_citations("Réponse [other.md#section-1]", passages)
    assert validate_citations(REFUSAL, passages)


def test_deterministic_check_is_accent_insensitive():
    assert deterministic_answer_check("Délai : 30 jours.", ["delai", "30 jours"])


def test_scope_guardrail_handles_empty_low_and_in_scope_results():
    assert should_refuse([], 0.35)
    assert should_refuse([0.12, 0.34], 0.35)
    assert not should_refuse([0.35, 0.2], 0.35)
