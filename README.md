# Retail Operations AI - Maison Kurt

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/juliettebm/retail-operations-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/juliettebm/retail-operations-rag/actions/workflows/ci.yml)
[![Reproductibilité](https://img.shields.io/badge/reproductibilit%C3%A9-v%C3%A9rifi%C3%A9e-success)](evaluation_results.json)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-005571)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

POC d'un assistant RAG pour les opérations retail, sur une documentation **entièrement fictive** (Maison Kurt) : retrouver une procédure dans la documentation interne (retours, échanges, remboursements, commandes en ligne, caisse, stock, SAV, prévention des fraudes), avec une petite interface Streamlit.

## Contenu

- `app.py` : interface Streamlit avec seuil de pertinence, refus hors périmètre et validation des citations.
- `src/retail_rag/` : découpage reproductible en sections, métriques et validateurs testables sans modèle génératif.
- `data/*.md` : les 8 guides opérationnels fictifs (50 passages) qui composent le corpus.
- `evaluation.json` : 65 questions annotées (52 in-scope dont 26 difficiles, 13 hors périmètre) avec passages pertinents.
- `notebook/01_retail_operations_rag.ipynb` : la démarche pas à pas (corpus, embeddings, Recall@k / MRR@k, étude du seuil de refus, validation des citations), exécuté avec ses sorties.
- `evaluate.py` : calcul de Recall@k et MRR@k sur les annotations passage-question.
- `evaluate_answers.py` : évaluation déterministe du refus, du contenu minimal et des citations.

**Résultats reproductibles du retrieval** : `multilingual-e5-small` (avec les préfixes `query:` / `passage:`) sur 52 questions in-scope annotées, dont 26 questions difficiles (reformulations sans les mots du guide, questions ambiguës, questions à deux passages), sur un corpus de 50 passages.

| | k=1 | k=2 (valeur de l'application) | k=5 |
|---|---|---|---|
| Recall@k | 70,2 % | 74,0 % | 83,7 % |
| MRR@k | 76,9 % | 77,9 % | 79,6 % |

Le chiffre le plus parlant est **MRR@1 = 76,9 %** : le bon passage arrive en première position pour environ trois questions sur quatre. Sur les questions difficiles seules, **MRR@1 = 61,5 %**. Le Recall@1 est plafonné pour les questions qui ont deux passages pertinents. Les 13 questions hors périmètre sont réservées à l'évaluation du garde-fou. La sortie structurée est conservée dans `evaluation_results.json`.

```bash
python evaluate.py --k 5
python evaluate.py --k 5 --check evaluation_results.json
python evaluate_answers.py answers.json
pytest
```

Le second script attend une liste JSON contenant `id`, `answer` et `retrieved_passage_ids`. Il contrôle des critères explicites et reproductibles. Pour une évaluation qualitative, faire relire les mêmes sorties à l'aveugle par une personne et conserver séparément son verdict ; aucun score humain n'est revendiqué dans ce dépôt.

**Limites** :

- Corpus fictif : 8 guides courts (50 passages), aucun document réel (PDF scannés, tableaux, versions contradictoires non testés).
- Jeu d'évaluation réduit : 65 questions dont 52 in-scope, rédigées par la même personne que le corpus (biais d'auteur), y compris les questions difficiles. Les métriques valident la recherche sur ce POC, pas sa tenue à l'échelle.
- Seuil de pertinence (`MIN_RELEVANCE_SCORE`, 0,35 par défaut) inopérant : les 13 questions hors périmètre obtiennent des scores allant jusqu'à 0,83, supérieurs au minimum des questions dans le périmètre (0,72), donc aucun seuil ne les sépare sur ce jeu. Le refus repose surtout sur le prompt et la validation des citations.
- Aucune évaluation humaine de la fidélité : les contrôles par termes attendus détectent des omissions, pas des erreurs subtiles, et la validation des citations prouve qu'un identifiant existe, pas que l'affirmation est vraie.
- Llama 3.2 3B est retenu pour le coût et le fonctionnement local, pas pour une précision de production.
- Aucune précision de génération n'est revendiquée : les métriques mesurent uniquement la recherche.

Lancer l'interface :

```bash
streamlit run app.py
```

## Installation

```bash
pip install -r requirements.txt
```

Nécessite [Ollama](https://ollama.com) avec le modèle explicite `llama3.2:3b` (`ollama pull llama3.2:3b`). Les dépendances Python sont figées dans `requirements.txt` ; Python 3.11 est recommandé. Le modèle d'embeddings `intfloat/multilingual-e5-small` est téléchargé automatiquement au premier lancement, ce qui nécessite une connexion internet.

## Structure

```
data/              guides opérationnels (.md)
notebook/          01_retail_operations_rag.ipynb
app.py             interface Streamlit
src/retail_rag/    composants réutilisables et métriques
evaluation.json    annotations d'évaluation
evaluate*.py       évaluations reproductibles
tests/             tests unitaires
LICENSE            MIT
```

> Maison Kurt, sa documentation et ses données sont entièrement fictives et servent uniquement à démontrer ce cas d'usage.
