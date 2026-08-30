"""
02 - Assistant GenAI : text-to-SQL sur les donnees operationnelles.

Pose une question en langage naturel : l'assistant genere une requete SQL
(SQLite), l'execute sur `data/operations.sqlite`, puis redige une reponse
claire pour un interlocuteur metier (responsable de magasin, ops retail).

Ce module est le portage direct des garde-fous ecrits pour l'assistant
text-to-SQL de `pharma-commercial-genai/src/03_genai_assistant.py`, avec une
seule difference de perimetre : le schema compte quatre tables au lieu
d'une, et les jointures entre ces quatre tables sont autorisees (le projet
d'origine s'y refusait par choix). Le contrat de securite ne change pas.

Garde-fous, du plus fort au plus faible :

1. Connexion en LECTURE SEULE (`mode=ro`). Aucune requete generee par le
   modele ne peut ecrire, quoi qu'elle contienne : c'est le moteur SQLite
   qui l'interdit, pas une expression reguliere.
2. Validation STRUCTURELLE par `sqlglot` : la requete est analysee sous
   forme d'arbre syntaxique, pas de texte. Un SELECT unique, portant
   uniquement sur les quatre tables autorisees, jointures comprises. Cela
   bloque `sqlite_master` tout en acceptant `LIKE '%update %'` — une chaine
   de caracteres n'est pas une instruction. Repli sur une validation par
   expressions regulieres si `sqlglot` n'est pas installe (repli qui, lui,
   n'a pas de notion de jointure et se limite a verifier que les seuls noms
   de table employes appartiennent a la liste blanche).
3. Plafond de lignes et delai maximum d'execution, pour qu'une jointure
   mal formee ne fasse pas tomber le service.
4. Ces couches sont partiellement redondantes, et c'est voulu : la premiere
   garantit, les suivantes donnent un refus lisible plutot qu'une erreur du
   moteur. `test_garde_fous.py` verifie les deux validateurs sur des
   requetes a autoriser (jointures incluses) et des requetes a bloquer.

Usage :
    python src/02_genai_assistant.py "Quel magasin a le taux de retard le plus eleve sur la livraison ?"
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "operations.sqlite"

ALLOWED_TABLES = {"magasins", "commandes", "retours", "remboursements"}
MAX_ROWS = 1000
QUERY_TIMEOUT_S = 10

SCHEMA = """
Table `magasins` (un magasin Maison Kurt) :
- magasin_id (INTEGER, cle primaire)
- nom (TEXT)
- ville (TEXT)

Table `commandes` (une commande, en magasin ou en ligne) :
- commande_id (INTEGER, cle primaire)
- magasin_id (INTEGER, cle etrangere vers magasins.magasin_id)
- canal (TEXT) : 'magasin', 'en ligne - livraison' ou 'en ligne - click&collect'
- date_commande (TEXT, format AAAA-MM-JJ)
- type_article (TEXT) : 'standard' ou 'soldé'
- moyen_paiement (TEXT) : 'carte bancaire' ou 'carte cadeau'
- montant (REAL, euros)
- statut (TEXT) : 'en cours', 'livrée' ou 'retournée'
- delai_traitement_jours (INTEGER ou NULL si la commande est encore en cours) : delai reel de livraison
- delai_sla_jours (INTEGER) : delai promis selon le canal
- en_retard (INTEGER 0/1, ou NULL si en cours) : 1 si delai_traitement_jours > delai_sla_jours

Table `retours` (un retour rattache a une commande livree) :
- retour_id (INTEGER, cle primaire)
- commande_id (INTEGER, cle etrangere vers commandes.commande_id)
- date_retour (TEXT, format AAAA-MM-JJ)
- motif (TEXT) : 'ne convient pas', 'taille inadaptée', 'article défectueux', "changement d'avis",
  'colis endommagé à réception' ou 'article incorrect reçu' (ces deux derniers uniquement pour les commandes en ligne)
- delai_traitement_jours (INTEGER) : delai de traitement du retour par le magasin
- dans_delai (INTEGER 0/1) : 1 si le retour respecte le delai contractuel (30 jours standard, 14 jours soldé)

Table `remboursements` (un remboursement rattache a un retour accepte) :
- remboursement_id (INTEGER, cle primaire)
- retour_id (INTEGER, cle etrangere vers retours.retour_id)
- moyen_paiement (TEXT) : 'carte bancaire' ou 'carte cadeau'
- montant (REAL, euros)
- date_remboursement (TEXT, format AAAA-MM-JJ)
- delai_jours (INTEGER) : delai entre le retour et le remboursement
"""

SQL_PROMPT = (
    "Tu traduis une question en une requete SQL SQLite.\n"
    "{schema}\n"
    "Regles STRICTES :\n"
    "- produis UNIQUEMENT une requete SELECT (jamais INSERT/UPDATE/DELETE/DROP) ;\n"
    "- une seule instruction, sans point-virgule ;\n"
    "- utilise uniquement les tables magasins, commandes, retours, remboursements et leurs colonnes ;\n"
    "- utilise une jointure (JOIN ... ON) quand la question porte sur plusieurs tables ;\n"
    "- ajoute LIMIT 20 au maximum si la question renvoie une liste ;\n"
    "- reponds avec la requete SQL SEULE, sans texte ni balise Markdown.\n\n"
    "Question : {question}\n"
    "SQL :"
)

NARRATE_PROMPT = (
    "Tu es analyste operations retail. En 1 a 3 phrases claires,\n"
    "reponds a la question a partir du resultat SQL, pour un interlocuteur metier.\n"
    "N'avance aucun chiffre qui ne figure pas dans le resultat. Si le resultat\n"
    "est partiel, dis-le explicitement plutot que de generaliser.\n\n"
    "Question : {question}\n"
    "Resultat : {n_lignes} ligne(s){note}\n{result}\n\n"
    "Reponse :"
)


def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=temperature)


SQL_CONTINUATIONS = {
    "select", "from", "where", "group", "order", "having", "limit", "offset",
    "join", "left", "right", "inner", "outer", "cross", "on", "union",
    "and", "or", "with", "as", "case", "when", "else", "end", "(", ")",
}


def extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip("` \n")
    m = re.search(r"(?is)\bselect\b.*", text)
    sql = m.group(0) if m else text
    sql = re.split(r";|```", sql)[0]
    # Une ligne vide n'est PAS une fin de requete : c'est aussi du SQL aere.
    # On ne coupe que si le bloc suivant ne commence pas par un mot-cle SQL,
    # sinon une requete bien formatee est tronquee en silence et le fragment
    # restant s'execute sans erreur, en renvoyant un resultat faux.
    blocs = re.split(r"\n\s*\n", sql)
    garde = [blocs[0]]
    for bloc in blocs[1:]:
        premier = bloc.strip().split(None, 1)[0].lower().strip("(,") if bloc.strip() else ""
        if premier in SQL_CONTINUATIONS:
            garde.append(bloc)
        else:
            break
    return "\n".join(garde).strip()


_TABLES_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_]\w*)", re.IGNORECASE)
_MOTS_INTERDITS = re.compile(
    r"\b(drop|delete|update|insert|alter|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def tables_referencees(sql: str) -> set[str]:
    return {t.lower() for t in _TABLES_RE.findall(sql)}


def _is_safe_sqlglot(sql: str) -> bool:
    """Validation structurelle : on analyse l'arbre syntaxique, pas le texte.

    Contrairement au projet d'origine, les jointures sont autorisees : c'est
    la table, pas la jointure, qui est le perimetre de securite. Chaque
    table de l'arbre, quel que soit le nombre de JOIN, doit appartenir a
    ALLOWED_TABLES.
    """
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
    """Repli si sqlglot n'est pas installe. Meme contrat, garanties plus faibles.

    Reconnait aussi les tables introduites par JOIN, mais sans comprendre la
    structure de la requete : une jointure malformee ou une sous-requete
    imbriquee de facon inhabituelle peut lui echapper la ou sqlglot la
    verifierait sur l'arbre syntaxique complet.
    """
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
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Base introuvable : {DB_PATH}. Lancer d'abord `python src/01_prepare_data.py`."
        )
    con = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    limite = time.monotonic() + QUERY_TIMEOUT_S
    con.set_progress_handler(lambda: time.monotonic() > limite, 100_000)
    return con


def run_sql(sql: str) -> tuple[pd.DataFrame, bool]:
    con = _connexion_lecture_seule()
    try:
        lecteur = pd.read_sql(sql, con, chunksize=MAX_ROWS)
        df = next(lecteur, pd.DataFrame())
        tronque = next(lecteur, None) is not None
    finally:
        con.close()
    return df, tronque


def ask(question: str, max_retries: int = 1, verbose: bool = True):
    llm = get_llm()
    sql_chain = ChatPromptTemplate.from_template(SQL_PROMPT) | llm | StrOutputParser()

    error, sql, df, tronque = None, "", None, False
    for _ in range(max_retries + 1):
        q = question if error is None else f"{question}\n(La requete precedente a echoue : {error}. Corrige-la.)"
        sql = extract_sql(sql_chain.invoke({"schema": SCHEMA, "question": q}))
        if not is_safe(sql):
            return (
                "Requete refusee : seules les tables magasins, commandes, retours et remboursements sont autorisees.",
                sql,
                None,
            )
        try:
            df, tronque = run_sql(sql)
            error = None
            break
        except Exception as exc:
            error = str(exc)
    if error is not None:
        return f"Echec de la generation SQL : {error}", sql, None

    apercu = df.head(20)
    note = ""
    if tronque:
        note = f" (plafond de lecture de {MAX_ROWS} lignes atteint, le total reel est superieur)"
    if len(df) > len(apercu):
        note += f" ; seules les {len(apercu)} premieres sont reproduites ci-dessous"

    narrate = ChatPromptTemplate.from_template(NARRATE_PROMPT) | llm | StrOutputParser()
    answer = narrate.invoke({
        "question": question,
        "n_lignes": len(df),
        "note": note,
        "result": apercu.to_string(index=False),
    })
    if verbose:
        print("\nSQL genere :\n ", sql)
        print(f"\nResultat : {len(df)} ligne(s){note}")
        print(df.head(10).to_string(index=False))
    return answer, sql, df


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Quel magasin a le taux de retard le plus eleve sur la livraison a domicile ?"
    answer, sql, df = ask(q)
    print("\nREPONSE :\n", answer)
