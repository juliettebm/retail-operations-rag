# Retail Operations AI — Maison Kurt

POC de deux assistants IA pour les opérations retail, sur une documentation et des données **entièrement fictives** (Maison Kurt) :

1. **RAG** — retrouver une procédure dans la documentation interne (retours, échanges, remboursements, commandes en ligne), avec une petite interface Streamlit.
2. **Text-to-SQL** — interroger les données opérationnelles (commandes, retours, remboursements) en langage naturel, avec garde-fous et évaluation.

## 1. RAG — assistant procédures

- `notebook/01_retail_operations_rag.ipynb` : pipeline complet (chunking, embeddings `multilingual-e5-small`, index FAISS, retrieval, génération avec Llama 3.2 via Ollama, évaluation du hit-rate et LLM-as-a-judge, garde-fou sur les questions hors périmètre).
- `app.py` : interface Streamlit — pose une question, la réponse générée s'affiche avec les passages sources utilisés.
- `data/*.md` : les 4 guides opérationnels fictifs qui composent le corpus.
- `questions.json` : jeu de questions utilisé pour l'évaluation du retrieval et de la génération.

Lancer l'interface :

```bash
streamlit run app.py
```

## 2. Text-to-SQL — assistant données opérationnelles

- `src/01_prepare_data.py` : génère `data/operations.sqlite` (magasins, commandes, retours, remboursements) à partir des règles écrites dans les guides opérationnels.
- `src/02_genai_assistant.py` : traduit une question en SQL (SQLite), l'exécute et rédige la réponse. Garde-fous : connexion en lecture seule, validation structurelle de la requête par arbre syntaxique (`sqlglot`), liste blanche de tables (jointures comprises), plafond de lignes et délai d'exécution.
- `test_garde_fous.py` : vérifie les deux validateurs (structurel et repli par expressions régulières) sur des requêtes à autoriser et à bloquer.
- `notebook/02_text_to_sql_operations.ipynb` : exploration des données, démonstration de l'assistant, évaluation par exécution (question → SQL de référence → comparaison des résultats).
- `responsible_ai/model_card.md` : model card de l'assistant (en anglais) — périmètre, garde-fous, précision mesurée, limites.

Régénérer les données puis interroger l'assistant en ligne de commande :

```bash
python src/01_prepare_data.py
python src/02_genai_assistant.py "Quel magasin a le taux de retard le plus élevé sur la livraison ?"
```

## Installation

```bash
pip install -r requirements.txt
```

Nécessite [Ollama](https://ollama.com) avec le modèle `llama3.2` (`ollama pull llama3.2`), ou `LLM_PROVIDER=openai` avec une clé API pour utiliser OpenAI à la place.

## Structure

```
data/               guides operationnels (.md) + base generee (operations.sqlite)
notebook/            01_retail_operations_rag.ipynb, 02_text_to_sql_operations.ipynb
src/                 01_prepare_data.py, 02_genai_assistant.py
responsible_ai/      model_card.md
app.py               interface Streamlit du RAG
test_garde_fous.py   tests des garde-fous du text-to-SQL
questions.json        jeu de questions d'evaluation du RAG
```

> Maison Kurt, sa documentation et ses données opérationnelles sont entièrement fictives et servent uniquement à démontrer ces deux cas d'usage.
