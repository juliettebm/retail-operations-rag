"""
Assistant operations Maison Kurt -- interface Streamlit pour le pipeline RAG
demontre dans notebook/01_retail_operations_rag.ipynb.

Usage :
    streamlit run app.py
"""
from __future__ import annotations

import os
from html import escape
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_MODEL", "llama3.2")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

EMBED_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
K = 2

PROMPT = (
    "Tu es un assistant d'analyse de documents operationnels retail.\n\n"
    "Reponds a la question UNIQUEMENT a partir du contexte fourni.\n\n"
    "IMPORTANT :\n"
    "- Lis attentivement toutes les informations pertinentes du contexte.\n"
    "- Si le contexte indique qu'une regle ne s'applique PAS, "
    "reponds clairement que la regle ne s'applique pas.\n"
    "- Les formulations negatives comme « ne modifie pas », "
    "« ne signifie pas automatiquement » ou « ne peut pas » "
    "contiennent des informations importantes et doivent etre utilisees "
    "pour repondre a la question.\n"
    "- Ne reponds jamais « Information non trouvee » si le contexte "
    "contient explicitement ou directement la reponse.\n\n"
    "Si aucune information permettant de repondre a la question "
    "n'est presente dans le contexte, reponds EXACTEMENT : "
    "« Information non trouvee dans les documents. »\n\n"
    "Sois concis et factuel.\n"
    "Lorsque tu utilises une information du contexte, cite entre "
    "guillemets l'extrait correspondant.\n\n"
    "Contexte :\n{context}\n\n"
    "Question : {question}\n"
    "Reponse :"
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    documents = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": path.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.from_documents(chunks, embedding_model)
    retriever = store.as_retriever(search_kwargs={"k": K})

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        temperature=0.0,
    )
    generation_chain = (
        ChatPromptTemplate.from_template(PROMPT)
        | llm
        | StrOutputParser()
    )

    return retriever, generation_chain


def answer_question(question: str, retriever, generation_chain):
    sources = retriever.invoke(question)
    context = "\n\n".join(source.page_content for source in sources)
    answer = generation_chain.invoke({"context": context, "question": question})
    return answer, sources


st.set_page_config(
    page_title="Maison Kurt -- Assistant operations",
    page_icon="\U0001F4E6",
    layout="centered",
)

st.html(
    """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#FBF7F1; --card:#FFFFFF; --ink:#2A1F3D; --body:#4B4059;
  --muted:#8A7F9B; --rule:#E7DFD6;
  --accent:#6B3FA0; --accent-soft:#F1EAFA;
  --blue:#2E5EAA; --blue-soft:#E9F0FB;
  --warn:#BC3C7E; --warn-soft:#FBECF4;
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}

.stApp{background:var(--paper)}
.block-container{max-width:760px;padding-top:3rem;padding-bottom:4rem}

h1,h2,h3{font-family:var(--serif)!important;color:var(--ink)!important;font-weight:400!important}

.eyebrow{font:500 .7rem/1.4 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin:0 0 .6rem}

.rag-header{border-bottom:2px solid var(--accent);padding-bottom:1.4rem;margin-bottom:2rem}
.rag-title{font-family:var(--serif);font-weight:400;font-size:2.1rem;line-height:1.15;
  color:var(--ink);margin:0 0 .6rem}
.rag-standfirst{font-family:var(--serif);font-size:1.05rem;line-height:1.55;
  color:var(--body);margin:0;max-width:60ch}

.stTextInput input, .stTextArea textarea{
  font-family:var(--serif)!important;font-size:1.05rem!important;
  background:var(--card)!important;color:var(--ink)!important;
  border:1px solid var(--rule)!important;border-radius:4px!important;
}
.stTextInput input:focus, .stTextArea textarea:focus{
  border-color:var(--accent)!important;box-shadow:none!important;
}

.stButton button{
  font-family:var(--mono)!important;font-size:.72rem!important;
  letter-spacing:.08em;text-transform:uppercase;font-weight:500!important;
  background:var(--accent)!important;color:#fff!important;
  border:none!important;border-radius:4px!important;padding:.55rem 1.4rem!important;
}
.stButton button:hover{background:#5A3488!important}

.answer-card{background:var(--accent-soft);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:0 4px 4px 0;
  padding:1.4rem 1.6rem;margin:1.5rem 0}
.answer-label{display:block;font:600 .68rem/1.4 var(--mono);letter-spacing:.11em;
  text-transform:uppercase;color:var(--accent);margin:0 0 .7rem}
.answer-text{font-family:var(--serif);font-size:1.12rem;line-height:1.6;color:var(--ink);
  margin:0;max-width:64ch}

.sources-label{font:600 .7rem/1.4 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin:2rem 0 .9rem}

.source-card{background:var(--card);border:1px solid var(--rule);border-radius:4px;
  padding:1rem 1.2rem;margin:0 0 .8rem}
.source-doc{font:500 .68rem/1.4 var(--mono);letter-spacing:.06em;text-transform:uppercase;
  color:var(--blue);margin:0 0 .5rem}
.source-text{font-family:var(--sans);font-size:.88rem;line-height:1.6;color:var(--body);
  margin:0;white-space:pre-wrap}

.error-card{background:var(--warn-soft);border-left:2px solid var(--warn);
  border-radius:0 4px 4px 0;padding:1rem 1.2rem;margin:1.5rem 0}
.error-label{display:block;font:600 .68rem/1.4 var(--mono);letter-spacing:.1em;
  text-transform:uppercase;color:var(--warn);margin:0 0 .4rem}
.error-text{font-family:var(--sans);font-size:.9rem;color:var(--body);margin:0}
</style>
"""
)

st.html(
    """
<div class="rag-header">
  <p class="eyebrow">Maison Kurt &middot; assistant operations</p>
  <p class="rag-title">Retrouver une procedure retail</p>
  <p class="rag-standfirst">Pose une question sur les retours, echanges, remboursements ou commandes
  en ligne. La reponse est generee a partir des passages les plus pertinents de la documentation
  interne (fictive), affiches ci-dessous pour verification.</p>
</div>
"""
)

with st.spinner("Chargement du pipeline (documents, embeddings, index)..."):
    retriever, generation_chain = load_pipeline()

question = st.text_input(
    "Question",
    placeholder="Ex. : Quel est le delai de retour d'un article solde ?",
    label_visibility="collapsed",
)
ask = st.button("Poser la question")

if ask and question.strip():
    try:
        with st.spinner("Recherche du passage pertinent, puis generation (~20 secondes)..."):
            answer, sources = answer_question(question.strip(), retriever, generation_chain)

        st.html(
            f"""
<div class="answer-card">
  <span class="answer-label">Reponse</span>
  <p class="answer-text">{escape(answer)}</p>
</div>
"""
        )

        st.html('<p class="sources-label">Passages utilises</p>')
        for source in sources:
            doc_name = source.metadata.get("source", "inconnue")
            st.html(
                f"""
<div class="source-card">
  <p class="source-doc">{escape(doc_name)}</p>
  <p class="source-text">{escape(source.page_content)}</p>
</div>
"""
            )
    except Exception as exc:
        st.html(
            f"""
<div class="error-card">
  <span class="error-label">Erreur</span>
  <p class="error-text">Ollama semble indisponible ou a manque de memoire. Verifie qu'Ollama tourne
  (<code>ollama list</code>) puis reessaie.<br>Detail : {escape(type(exc).__name__)} -- {escape(str(exc)[:200])}</p>
</div>
"""
        )
elif ask:
    st.warning("Ecris une question avant de valider.")
