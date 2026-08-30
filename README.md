# Retail Operations AI — Maison Kurt

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20%7C%20text--to--SQL-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-005571)](https://github.com/facebookresearch/faiss)
[![SQLite](https://img.shields.io/badge/SQLite-text--to--SQL-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

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

- `notebook/02_generation_donnees.ipynb` : génère `data/operations.sqlite` (magasins, commandes, retours, remboursements) à partir des règles écrites dans les guides opérationnels. À exécuter en premier.
- `notebook/03_text_to_sql_operations.ipynb` : exploration des données, garde-fous et assistant text-to-SQL, tests des garde-fous, démonstration, évaluation par exécution (question → SQL de référence → comparaison des résultats). Tout le code est dans les cellules du notebook, sans fichier `.py` séparé.
- `responsible_ai/model_card.md` : model card de l'assistant (en anglais) — périmètre, garde-fous, précision mesurée, limites.

Garde-fous de l'assistant : connexion en lecture seule, validation structurelle de la requête par arbre syntaxique (`sqlglot`), liste blanche de tables (jointures comprises), plafond de lignes et délai d'exécution.

## Installation

```bash
pip install -r requirements.txt
```

Nécessite [Ollama](https://ollama.com) avec le modèle `llama3.2` (`ollama pull llama3.2`), ou `LLM_PROVIDER=openai` avec une clé API pour utiliser OpenAI à la place.

## Structure

```
data/               guides operationnels (.md) + base generee (operations.sqlite)
notebook/            01_retail_operations_rag.ipynb
                     02_generation_donnees.ipynb
                     03_text_to_sql_operations.ipynb
responsible_ai/      model_card.md
app.py               interface Streamlit du RAG
questions.json        jeu de questions d'evaluation du RAG
```

> Maison Kurt, sa documentation et ses données opérationnelles sont entièrement fictives et servent uniquement à démontrer ces deux cas d'usage.
