# Retail Operations AI - Maison Kurt

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/juliettebm/retail-operations-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/juliettebm/retail-operations-rag/actions/workflows/ci.yml)
[![Reproductibilité](https://img.shields.io/badge/reproductibilit%C3%A9-v%C3%A9rifi%C3%A9e-success)](evaluation_results.json)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-005571)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

POC d'un assistant RAG pour les opérations retail, sur une documentation **entièrement fictive** (Maison Kurt) : retrouver une procédure dans la documentation interne (retours, échanges, remboursements, commandes en ligne), avec une petite interface Streamlit.

## Contenu

- `app.py` : interface Streamlit avec seuil de pertinence, refus hors périmètre et validation des citations.
- `src/retail_rag/` : découpage reproductible en sections, métriques et validateurs testables sans modèle génératif.
- `data/*.md` : les 4 guides opérationnels fictifs qui composent le corpus.
- `evaluation.json` : 26 questions annotées (18 in-scope, 8 hors périmètre) avec passages pertinents.
- `evaluate.py` : calcul de Recall@k et MRR@k sur les annotations passage-question.
- `evaluate_answers.py` : évaluation déterministe du refus, du contenu minimal et des citations.

Les anciens chiffres de hit-rate et de LLM-as-a-judge ont été retirés : ils reposaient sur trop peu de cas et sur un juge instable.

**Résultats reproductibles du retrieval (20 septembre 2026)** : sur les 18 questions in-scope annotées, `multilingual-e5-small` obtient **Recall@5 = 100 %** et **MRR@5 = 97,2 %**. Les 8 questions hors périmètre sont réservées à l'évaluation du garde-fou. La sortie structurée est conservée dans `evaluation_results.json`.

```bash
python evaluate.py --k 5
python evaluate.py --k 5 --check evaluation_results.json
python evaluate_answers.py answers.json
pytest
```

Le second script attend une liste JSON contenant `id`, `answer` et `retrieved_passage_ids`. Il contrôle des critères explicites et reproductibles. Pour une évaluation qualitative, faire relire les mêmes sorties à l'aveugle par une personne et conserver séparément son verdict ; aucun score humain n'est revendiqué dans ce dépôt.

Le notebook et `questions.json` sont conservés comme trace du POC initial. Ils ne constituent plus la source des métriques actuelles ; l'évaluation de référence est `evaluation.json` avec les scripts ci-dessus.

**Limites** : le seuil de pertinence (`MIN_RELEVANCE_SCORE`, 0,35 par défaut) doit être recalibré sur davantage de données. Les contrôles par termes attendus détectent des omissions, mais ne remplacent pas une revue humaine de la fidélité. Llama 3.2 3B est retenu pour le coût et le fonctionnement local, pas pour une précision de production.

Lancer l'interface :

```bash
streamlit run app.py
```

## Installation

```bash
pip install -r requirements.txt
```

Nécessite [Ollama](https://ollama.com) avec le modèle explicite `llama3.2:3b` (`ollama pull llama3.2:3b`). Les dépendances Python sont figées dans `requirements.txt`; Python 3.11 est recommandé (le notebook a aussi été exécuté avec Python 3.12). Le modèle d'embeddings `intfloat/multilingual-e5-small` est téléchargé automatiquement au premier lancement, ce qui nécessite une connexion internet.

## Structure

```
data/              guides operationnels (.md)
notebook/          01_retail_operations_rag.ipynb
app.py             interface Streamlit
src/retail_rag/    composants réutilisables et métriques
evaluation.json    annotations d'évaluation
evaluate*.py       évaluations reproductibles
tests/             tests unitaires
LICENSE            MIT
```

> Maison Kurt, sa documentation et ses données sont entièrement fictives et servent uniquement à démontrer ce cas d'usage.
