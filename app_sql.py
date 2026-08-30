"""
Assistant opérations Maison Kurt : interface Streamlit pour l'assistant
text-to-SQL démontré dans notebook/03_text_to_sql_operations.ipynb.

Nécessite data/operations.sqlite (généré par notebook/02_generation_donnees.ipynb).

Usage :
    streamlit run app_sql.py
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_MODEL", "llama3.2")

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "operations.sqlite"

ALLOWED_TABLES = {"magasins", "commandes", "retours", "remboursements"}
MAX_ROWS = 1000
QUERY_TIMEOUT_S = 10

SCHEMA = """
Table `magasins` (un magasin Maison Kurt) :
- magasin_id (INTEGER, clé primaire)
- nom (TEXT)
- ville (TEXT)

Table `commandes` (une commande, en magasin ou en ligne) :
- commande_id (INTEGER, clé primaire)
- magasin_id (INTEGER, clé étrangère vers magasins.magasin_id)
- canal (TEXT) : 'magasin', 'en ligne - livraison' ou 'en ligne - click&collect'
- date_commande (TEXT, format AAAA-MM-JJ)
- type_article (TEXT) : 'standard' ou 'soldé'
- moyen_paiement (TEXT) : 'carte bancaire' ou 'carte cadeau'
- montant (REAL, euros)
- statut (TEXT) : 'en cours', 'livrée' ou 'retournée'
- delai_traitement_jours (INTEGER ou NULL si la commande est encore en cours) : délai réel de livraison
- delai_sla_jours (INTEGER) : délai promis selon le canal
- en_retard (INTEGER 0/1, ou NULL si en cours) : 1 si delai_traitement_jours > delai_sla_jours

Table `retours` (un retour rattaché à une commande livrée) :
- retour_id (INTEGER, clé primaire)
- commande_id (INTEGER, clé étrangère vers commandes.commande_id)
- date_retour (TEXT, format AAAA-MM-JJ)
- motif (TEXT) : 'ne convient pas', 'taille inadaptée', 'article défectueux', "changement d'avis",
  'colis endommagé à réception' ou 'article incorrect reçu' (ces deux derniers uniquement pour les commandes en ligne)
- delai_traitement_jours (INTEGER) : délai de traitement du retour par le magasin
- dans_delai (INTEGER 0/1) : 1 si le retour respecte le délai contractuel (30 jours standard, 14 jours soldé)

Table `remboursements` (un remboursement rattaché à un retour accepté) :
- remboursement_id (INTEGER, clé primaire)
- retour_id (INTEGER, clé étrangère vers retours.retour_id)
- moyen_paiement (TEXT) : 'carte bancaire' ou 'carte cadeau'
- montant (REAL, euros)
- date_remboursement (TEXT, format AAAA-MM-JJ)
- delai_jours (INTEGER) : délai entre le retour et le remboursement
"""

SQL_PROMPT = (
    "Tu es un expert SQL SQLite spécialisé dans l'analyse de données retail.\n"
    "Ta tâche est de transformer la question métier en UNE requête SQL correcte.\n\n"

    "SCHÉMA DE LA BASE :\n"
    "{schema}\n\n"

    "RELATIONS ENTRE LES TABLES (À RESPECTER STRICTEMENT) :\n"
    "- magasins.magasin_id = commandes.magasin_id\n"
    "- commandes.commande_id = retours.commande_id\n"
    "- retours.retour_id = remboursements.retour_id\n\n"

    "CHEMIN DES JOINTURES :\n"
    "- Pour relier magasins et commandes : magasins -> commandes\n"
    "- Pour relier commandes et retours : commandes -> retours\n"
    "- Pour relier retours et remboursements : retours -> remboursements\n"
    "- Pour relier remboursements et commandes : remboursements -> retours -> commandes\n"
    "- Pour relier remboursements et magasins : remboursements -> retours -> commandes -> magasins\n"
    "- Ne crée JAMAIS de jointure directe entre deux tables qui ne possèdent pas de clé étrangère correspondante.\n\n"

    "RÈGLES MÉTIER IMPORTANTES :\n"
    "- Le canal d'une commande est dans commandes.canal.\n"
    "- Le moyen de paiement d'une commande est dans commandes.moyen_paiement.\n"
    "- Le moyen de paiement d'un remboursement est dans remboursements.moyen_paiement.\n"
    "- Le motif d'un retour est dans retours.motif.\n"
    "- Le délai de traitement d'un retour est dans retours.delai_traitement_jours.\n"
    "- Le délai entre retour et remboursement est dans remboursements.delai_jours.\n"
    "- Le montant d'une commande est dans commandes.montant.\n"
    "- Le montant d'un remboursement est dans remboursements.montant.\n"
    "- Une commande peut être reliée à son retour par commandes.commande_id = retours.commande_id.\n"
    "- Un remboursement peut être relié à sa commande uniquement en passant par retours.\n"
    "- 'en ligne' désigne le canal de commande, pas le moyen de paiement.\n"
    "- 'carte bancaire' désigne un moyen de paiement, pas un canal de commande.\n"
    "- 'soldé' désigne commandes.type_article = 'soldé'.\n"
    "- 'en retard' désigne commandes.en_retard = 1.\n"
    "- 'hors délai' pour un retour désigne retours.dans_delai = 0.\n"
    "- Quand la question mentionne des 'commandes retournées', filtre commandes.statut = 'retournée'.\n"
    "- Quand la question demande des 'magasins', retourne par défaut magasins.nom plutôt que magasins.magasin_id, sauf si l'identifiant est explicitement demandé.\n"
    "- Quand une question demande les magasins classés selon le montant total de leurs commandes, sélectionne magasins.nom, regroupe par magasins.nom et trie selon SUM(commandes.montant).\n\n"

    "RÈGLES POUR LES AGRÉGATIONS :\n"
    "- Pour 'combien', utilise COUNT(*) ou COUNT(colonne).\n"
    "- Pour 'montant total', utilise SUM(montant).\n"
    "- Pour 'montant moyen', utilise AVG(montant).\n"
    "- Pour 'délai moyen', utilise AVG() sur la colonne de délai correspondant à la question.\n"
    "- Pour 'le plus fréquent', utilise GROUP BY, ORDER BY COUNT(*) DESC et LIMIT 1.\n"
    "- Pour 'top 5', utilise ORDER BY ... DESC LIMIT 5.\n"
    "- Pour un pourcentage, calcule le ratio demandé sur le sous-ensemble correspondant à la question.\n"
    "- N'ajoute pas de filtre qui n'est pas demandé dans la question.\n\n"

    "RÈGLES STRICTES DE SQL :\n"
    "- produis UNIQUEMENT une requête SELECT.\n"
    "- jamais INSERT, UPDATE, DELETE, DROP, ALTER, CREATE ou PRAGMA.\n"
    "- une seule instruction, sans point-virgule.\n"
    "- utilise uniquement les tables et colonnes présentes dans le schéma.\n"
    "- utilise les relations de jointure indiquées ci-dessus.\n"
    "- si plusieurs tables sont nécessaires, utilise explicitement JOIN ... ON.\n"
    "- ne crée pas de relation entre remboursements et commandes sans passer par retours.\n"
    "- n'utilise jamais sqlite_master ou une autre table système.\n"
    "- ajoute LIMIT 20 au maximum uniquement lorsqu'une question demande une liste.\n"
    "- si la question demande 'le plus fréquent', utilise LIMIT 1 et non LIMIT 20.\n"
    "- réponds avec la requête SQL SEULE, sans explication et sans balise Markdown.\n\n"

    "EXEMPLES DE RAISONNEMENT ATTENDU :\n\n"

    "Question : Quel est le motif de retour le plus fréquent ?\n"
    "SQL : SELECT motif FROM retours GROUP BY motif ORDER BY COUNT(*) DESC LIMIT 1\n\n"

    "Question : Quel est le délai moyen de traitement des retours pour les commandes passées en magasin ?\n"
    "SQL : SELECT AVG(r.delai_traitement_jours) FROM retours r JOIN commandes c ON r.commande_id = c.commande_id WHERE c.canal = 'magasin'\n\n"

    "Question : Quel est le montant moyen remboursé pour le motif article défectueux ?\n"
    "SQL : SELECT AVG(rb.montant) FROM remboursements rb JOIN retours r ON rb.retour_id = r.retour_id WHERE r.motif = 'article défectueux'\n\n"

    "Question : Quel est le délai moyen entre le retour et le remboursement pour les commandes en ligne ?\n"
    "SQL : SELECT AVG(rb.delai_jours) FROM remboursements rb JOIN retours r ON rb.retour_id = r.retour_id JOIN commandes c ON r.commande_id = c.commande_id WHERE c.canal LIKE 'en ligne%'\n\n"

    "Question : Combien de remboursements par carte bancaire concernent des commandes soldées ?\n"
    "SQL : SELECT COUNT(*) FROM remboursements rb JOIN retours r ON rb.retour_id = r.retour_id JOIN commandes c ON r.commande_id = c.commande_id WHERE rb.moyen_paiement = 'carte bancaire' AND c.type_article = 'soldé'\n\n"

    "Question : {question}\n"
    "SQL :"
)

NARRATE_PROMPT = (
    "Tu es analyste opérations retail. En 1 à 3 phrases claires,\n"
    "réponds à la question à partir du résultat SQL, pour un interlocuteur métier.\n"
    "N'avance aucun chiffre qui ne figure pas dans le résultat. Si le résultat\n"
    "est partiel, dis-le explicitement plutôt que de généraliser.\n\n"
    "Question : {question}\n"
    "Résultat : {n_lignes} ligne(s){note}\n{result}\n\n"
    "Réponse :"
)

SQL_CONTINUATIONS = {
    "select", "from", "where", "group", "order", "having", "limit", "offset",
    "join", "left", "right", "inner", "outer", "cross", "on", "union",
    "and", "or", "with", "as", "case", "when", "else", "end", "(", ")",
}
_TABLES_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_]\w*)", re.IGNORECASE)
_MOTS_INTERDITS = re.compile(
    r"\b(drop|delete|update|insert|alter|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip("` \n")
    m = re.search(r"(?is)\bselect\b.*", text)
    sql = m.group(0) if m else text
    sql = re.split(r";|```", sql)[0]
    blocs = re.split(r"\n\s*\n", sql)
    garde = [blocs[0]]
    for bloc in blocs[1:]:
        premier = bloc.strip().split(None, 1)[0].lower().strip("(,") if bloc.strip() else ""
        if premier in SQL_CONTINUATIONS:
            garde.append(bloc)
        else:
            break
    return "\n".join(garde).strip()


def tables_referencees(sql: str) -> set:
    return {t.lower() for t in _TABLES_RE.findall(sql)}


def _is_safe_sqlglot(sql: str) -> bool:
    import sqlglot
    from sqlglot import exp

    try:
        instructions = sqlglot.parse(sql, read="sqlite")
    except Exception:
        return False
    if len(instructions) != 1:
        return False
    arbre = instructions[0]
    if not isinstance(arbre, exp.Select):
        return False
    tables = list(arbre.find_all(exp.Table))
    if not tables:
        return False
    if any(t.name.lower() not in ALLOWED_TABLES for t in tables):
        return False
    interdits = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)
    return not any(isinstance(n, interdits) for n in arbre.walk())


def _is_safe_regex(sql: str) -> bool:
    nu = sql.strip()
    if not nu.lower().startswith("select") or ";" in nu:
        return False
    hors_chaines = re.sub(r"'[^']*'", "''", nu)
    if _MOTS_INTERDITS.search(hors_chaines):
        return False
    return tables_referencees(hors_chaines).issubset(ALLOWED_TABLES)


def is_safe(sql: str) -> bool:
    try:
        import sqlglot  # noqa: F401
    except ImportError:
        return _is_safe_regex(sql)
    return _is_safe_sqlglot(sql)


def _connexion_lecture_seule() -> sqlite3.Connection:
    con = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    limite = time.monotonic() + QUERY_TIMEOUT_S
    con.set_progress_handler(lambda: time.monotonic() > limite, 100_000)
    return con


def run_sql(sql: str):
    con = _connexion_lecture_seule()
    try:
        lecteur = pd.read_sql(sql, con, chunksize=MAX_ROWS)
        df = next(lecteur, pd.DataFrame())
        tronque = next(lecteur, None) is not None
    finally:
        con.close()
    return df, tronque


@st.cache_resource(show_spinner=False)
def load_chains():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0.0)
    sql_chain = ChatPromptTemplate.from_template(SQL_PROMPT) | llm | StrOutputParser()
    narrate_chain = ChatPromptTemplate.from_template(NARRATE_PROMPT) | llm | StrOutputParser()
    return sql_chain, narrate_chain


def ask(question: str, sql_chain, narrate_chain, max_retries: int = 1):
    error, sql = None, ""
    for _ in range(max_retries + 1):
        q = question if error is None else f"{question}\n(La requête précédente a échoué : {error}. Corrige-la.)"
        sql = extract_sql(sql_chain.invoke({"schema": SCHEMA, "question": q}))
        if not is_safe(sql):
            return "refuse", sql, None, None
        try:
            df, tronque = run_sql(sql)
            break
        except Exception as exc:
            error = str(exc)
            df, tronque = None, False
    else:
        return "echec", sql, None, error

    apercu = df.head(20)
    note = ""
    if tronque:
        note = f" (plafond de lecture de {MAX_ROWS} lignes atteint, le total réel est supérieur)"
    if len(df) > len(apercu):
        note += f" ; seules les {len(apercu)} premières sont reproduites ci-dessous"

    answer = narrate_chain.invoke({
        "question": question,
        "n_lignes": len(df),
        "note": note,
        "result": apercu.to_string(index=False),
    })
    return "ok", sql, df, answer


st.set_page_config(
    page_title="Maison Kurt : assistant données",
    page_icon="\U0001F4CA",
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
.block-container{max-width:780px;padding-top:3rem;padding-bottom:4rem}

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

.sql-label{font:600 .7rem/1.4 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin:2rem 0 .6rem}
.sql-card{background:var(--card);border:1px solid var(--rule);border-radius:4px;
  padding:1rem 1.2rem;margin:0 0 1.4rem}
.sql-text{font-family:var(--mono);font-size:.85rem;line-height:1.6;color:var(--ink);
  margin:0;white-space:pre-wrap;word-break:break-word}

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
  <p class="eyebrow">Maison Kurt &middot; assistant données</p>
  <p class="rag-title">Interroger les opérations en langage naturel</p>
  <p class="rag-standfirst">Pose une question sur les commandes, retours ou remboursements. La question est
  traduite en SQL, exécutée en lecture seule sur la base, puis résumée. SQL et résultat bruts affichés
  ci-dessous pour vérification.</p>
</div>
"""
)

if not DB_PATH.exists():
    st.html(
        f"""
<div class="error-card">
  <span class="error-label">Base introuvable</span>
  <p class="error-text">{escape(str(DB_PATH))} n'existe pas. Lance d'abord
  <code>notebook/02_generation_donnees.ipynb</code> pour la générer.</p>
</div>
"""
    )
    st.stop()

with st.spinner("Chargement du modèle..."):
    sql_chain, narrate_chain = load_chains()

question = st.text_input(
    "Question",
    placeholder="Ex. : Combien de retours sont hors délai ?",
    label_visibility="collapsed",
)
ask_clicked = st.button("Poser la question")

if ask_clicked and question.strip():
    try:
        with st.spinner("Génération du SQL, exécution, puis rédaction de la réponse (1 à 2 minutes, prompt long)..."):
            statut, sql, df, answer = ask(question.strip(), sql_chain, narrate_chain)

        if statut == "refuse":
            st.html(
                f"""
<div class="error-card">
  <span class="error-label">Requête refusée</span>
  <p class="error-text">Le SQL généré sort du périmètre autorisé (tables magasins, commandes, retours,
  remboursements uniquement).</p>
</div>
<p class="sql-label">SQL généré (rejeté)</p>
<div class="sql-card"><p class="sql-text">{escape(sql)}</p></div>
"""
            )
        elif statut == "echec":
            st.html(
                f"""
<div class="error-card">
  <span class="error-label">Échec d'exécution</span>
  <p class="error-text">Le SQL généré n'a pas pu s'exécuter : {escape(str(df))}</p>
</div>
<p class="sql-label">SQL généré</p>
<div class="sql-card"><p class="sql-text">{escape(sql)}</p></div>
"""
            )
        else:
            st.html(
                f"""
<div class="answer-card">
  <span class="answer-label">Réponse</span>
  <p class="answer-text">{escape(answer)}</p>
</div>
"""
            )
            st.html(f'<p class="sql-label">SQL généré</p><div class="sql-card"><p class="sql-text">{escape(sql)}</p></div>')
            st.html('<p class="sql-label">Résultat</p>')
            st.dataframe(df, use_container_width=True)
    except Exception as exc:
        st.html(
            f"""
<div class="error-card">
  <span class="error-label">Erreur</span>
  <p class="error-text">Ollama semble indisponible ou a manqué de mémoire. Vérifie qu'Ollama tourne
  (<code>ollama list</code>) puis réessaie.<br>Détail : {escape(type(exc).__name__)} : {escape(str(exc)[:200])}</p>
</div>
"""
        )
elif ask_clicked:
    st.warning("Écris une question avant de valider.")
