# Retail Operations AI - Maison Kurt

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-005571)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

POC d'un assistant RAG pour les opérations retail, sur une documentation **entièrement fictive** (Maison Kurt) : retrouver une procédure dans la documentation interne (retours, échanges, remboursements, commandes en ligne), avec une petite interface Streamlit.

## Contenu

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

## Installation

```bash
pip install -r requirements.txt
```

Nécessite [Ollama](https://ollama.com) avec le modèle `llama3.2` (`ollama pull llama3.2`), ou `LLM_PROVIDER=openai` avec une clé API pour utiliser OpenAI à la place.

## Structure

```
data/              guides operationnels (.md)
notebook/          01_retail_operations_rag.ipynb
app.py             interface Streamlit
questions.json     jeu de questions d'evaluation
LICENSE            MIT
```

> Maison Kurt, sa documentation et ses données sont entièrement fictives et servent uniquement à démontrer ce cas d'usage.
