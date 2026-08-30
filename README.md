# Retail Operations AI - Maison Kurt

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20%7C%20text--to--SQL-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-005571)](https://github.com/facebookresearch/faiss)
[![SQLite](https://img.shields.io/badge/SQLite-text--to--SQL-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

POC de deux assistants IA pour les opérations retail, sur une documentation et des données **entièrement fictives** (Maison Kurt) :

1. **RAG** : retrouver une procédure dans la documentation interne (retours, échanges, remboursements, commandes en ligne), avec une petite interface Streamlit.
2. **Text-to-SQL** : interroger les données opérationnelles (commandes, retours, remboursements) en langage naturel, avec garde-fous et évaluation.

## 1. RAG : assistant procédures

- `notebook/01_retail_operations_rag.ipynb` : pipeline complet (chunking, embeddings `multilingual-e5-small`, index FAISS, retrieval, génération avec Llama 3.2 via Ollama, évaluation du hit-rate et LLM-as-a-judge, garde-fou sur les questions hors périmètre).
- `app.py` : interface Streamlit. Pose une question, la réponse générée s'affiche avec les passages sources utilisés.
- `data/*.md` : les 4 guides opérationnels fictifs qui composent le corpus.
- `questions.json` : jeu de questions utilisé pour l'évaluation du retrieval et de la génération.

**Résultats** : hit-rate du retrieval de 100% à partir de k=2 (83,3% à k=1, sur les 12 questions du périmètre) ; fidélité et exactitude moyennes de 1,73/2 en LLM-as-a-judge (15 questions) ; garde-fou anti-hallucination correct sur 3 des 4 questions hors périmètre testées.

**Limites** : un hit-rate à 100% ne garantit pas une réponse juste, sur une question test le bon chiffre est présent dans le contexte à k=2 et le modèle choisit quand même le mauvais (le notebook le montre explicitement, §6). Le hit-rate lui-même est une approximation par mots-clés, pas un Recall@k annoté à la main. Le juge LLM-as-a-judge se contredit parfois lui-même (note 0/0 avec un commentaire disant la réponse correcte, sur 2 des 15 questions). Llama 3.2 (3B, local) est retenu pour le coût et le hors-ligne, pas pour la précision.

Lancer l'interface :

```bash
streamlit run app.py
```

## 2. Text-to-SQL : assistant données opérationnelles

- `notebook/02_generation_donnees.ipynb` : génère `data/operations.sqlite` (magasins, commandes, retours, remboursements) à partir des règles écrites dans les guides opérationnels. À exécuter en premier.
- `notebook/03_text_to_sql_operations.ipynb` : exploration des données, garde-fous et assistant text-to-SQL, tests des garde-fous, démonstration, évaluation par exécution (question, SQL de référence, comparaison des résultats). Tout le code du notebook est dans ses propres cellules, sans import depuis un fichier `.py` séparé.
- `notebook/questions.py` : le jeu de 17 questions de référence (`EVAL_SET`) importé par le notebook pour l'évaluation.
- `app_sql.py` : interface Streamlit de l'assistant text-to-SQL. Pose une question, le SQL généré et le résultat s'affichent avec la réponse rédigée.
- `responsible_ai/model_card.md` : model card de l'assistant (en anglais), périmètre, garde-fous, précision mesurée, limites.

Garde-fous de l'assistant : connexion en lecture seule, validation structurelle de la requête par arbre syntaxique (`sqlglot`), liste blanche de tables (jointures comprises), plafond de lignes et délai d'exécution.

**Résultats** (17 questions, comparaison du résultat SQL complet, pas juste sa première valeur) : execution accuracy de 82,4% (14/17), 81,8% sans jointure (9/11), 83,3% avec jointure (5/6). Aucun des 3 échecs n'a été bloqué par le garde-fou : les requêtes exécutent, elles répondent juste à côté (division entière tronquée, colonne ambiguë après jointure, filtre non demandé recopié d'un exemple du prompt).

**Limites** : seulement 6 questions à jointure, un échantillon trop petit pour une mesure fiable, un seul cas qui bascule déplace ce taux de 17 points (une exécution précédente donnait d'ailleurs 88,2%, avec un taux de jointure déjà à 83,3%). Le garde-fou prouve ce qui ne peut pas arriver (écriture, table hors périmètre) mais rien sur la justesse de la réponse, mesurée séparément. La narration en langage naturel (fonction `ask`, utilisée par la démo et par `app_sql.py`) n'est pas couverte par cette accuracy : elle peut se tromper même quand le SQL est correct (bug reproduit deux fois : le SQL renvoie 1080, la phrase dit "1 commande"). Llama 3.2 (3B, local) est retenu pour le coût et le hors-ligne, pas pour la précision. Détail complet dans `responsible_ai/model_card.md`.

Lancer l'interface (après avoir exécuté `02_generation_donnees.ipynb` au moins une fois) :

```bash
streamlit run app_sql.py
```

## Installation

```bash
pip install -r requirements.txt
```

Nécessite [Ollama](https://ollama.com) avec le modèle `llama3.2` (`ollama pull llama3.2`), ou `LLM_PROVIDER=openai` avec une clé API pour utiliser OpenAI à la place.

## Structure

```
data/              guides operationnels (.md) + base generee (operations.sqlite)
notebook/          01_retail_operations_rag.ipynb
                   02_generation_donnees.ipynb
                   03_text_to_sql_operations.ipynb
                   questions.py
responsible_ai/    model_card.md
app.py             interface Streamlit du RAG
app_sql.py         interface Streamlit du text-to-SQL
entrainement_sql.py  30 questions SQL d'entrainement, hors evaluation de l'assistant
questions.json     jeu de questions d'evaluation du RAG
LICENSE            MIT
```

> Maison Kurt, sa documentation et ses données opérationnelles sont entièrement fictives et servent uniquement à démontrer ces deux cas d'usage.
