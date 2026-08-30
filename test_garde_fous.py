"""
Tests des garde-fous de l'assistant text-to-SQL.

Ne necessite ni base de donnees, ni LLM : on teste uniquement les fonctions
pures `is_safe` et `extract_sql`. C'est le minimum pour pouvoir ecrire
"aucune requete destructrice ne peut passer, jointures comprises" et le
demontrer.

Usage :
    python test_garde_fous.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src" / "02_genai_assistant.py"
_spec = importlib.util.spec_from_file_location("assistant", SRC)
assistant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assistant)

is_safe = assistant.is_safe
extract_sql = assistant.extract_sql


# --------------------------------------------------------------------------
# Requetes legitimes : doivent passer, jointures comprises
# --------------------------------------------------------------------------
AUTORISEES = [
    "SELECT COUNT(*) FROM commandes WHERE canal = 'magasin'",
    "SELECT motif, COUNT(*) FROM retours GROUP BY motif ORDER BY 2 DESC LIMIT 5",
    "select statut, avg(montant) from commandes group by statut",
    # Jointure entre deux tables de la liste blanche : c'est precisement ce
    # que ce projet autorise, contrairement au projet d'origine.
    """SELECT m.nom, AVG(c.en_retard) FROM commandes c
       JOIN magasins m ON c.magasin_id = m.magasin_id
       WHERE c.canal = 'en ligne - livraison' GROUP BY m.nom""",
    # Jointure a trois tables.
    """SELECT r.motif, AVG(rb.delai_jours) FROM retours r
       JOIN commandes c ON r.commande_id = c.commande_id
       JOIN remboursements rb ON rb.retour_id = r.retour_id
       GROUP BY r.motif""",
    # Une chaine litterale contenant un mot-cle interdit ne doit pas
    # declencher la liste noire : c'est de la donnee, pas une instruction.
    "SELECT * FROM commandes WHERE motif LIKE '%update %' LIMIT 10",
    # Sous-requete : seule la table reelle compte, la table derivee n'est
    # pas un nom a autoriser.
    "SELECT AVG(m) FROM (SELECT SUM(montant) m FROM commandes GROUP BY magasin_id)",
]

# --------------------------------------------------------------------------
# Requetes a refuser
# --------------------------------------------------------------------------
REFUSEES = [
    # Ecriture directe
    "DROP TABLE commandes",
    "DELETE FROM retours",
    "UPDATE commandes SET montant = 0",
    "INSERT INTO commandes VALUES (1)",
    # Enchainement d'instructions
    "SELECT 1 FROM commandes; DROP TABLE commandes",
    # Exfiltration du schema ou lecture d'une table non autorisee, y compris
    # via une jointure vers une table hors liste blanche
    "SELECT name, sql FROM sqlite_master",
    "SELECT * FROM sqlite_sequence",
    "SELECT c.montant FROM commandes c JOIN sqlite_master s ON 1=1",
    # Rattachement d'une autre base
    "ATTACH DATABASE 'autre.sqlite' AS x",
    "SELECT * FROM commandes WHERE 1=1 ATTACH DATABASE 'x' AS y",
    # CTE : refusee par choix, le filtre n'admet qu'un SELECT simple
    "WITH t AS (SELECT 1 FROM commandes) SELECT * FROM t",
]


def _verifier(valideur, nom: str) -> None:
    for sql in AUTORISEES:
        assert valideur(sql), f"[{nom}] faux refus : {sql}"
    for sql in REFUSEES:
        assert not valideur(sql), f"[{nom}] NON BLOQUE : {sql}"


def test_is_safe() -> None:
    """Le validateur effectivement utilise par l'assistant."""
    _verifier(is_safe, "actif")


def test_is_safe_sqlglot() -> None:
    """Validation par arbre syntaxique (chemin nominal), jointures comprises."""
    try:
        import sqlglot  # noqa: F401
    except ImportError:
        print("    (sqlglot absent : chemin non teste, `pip install sqlglot`)")
        return
    _verifier(assistant._is_safe_sqlglot, "sqlglot")


def test_is_safe_regex() -> None:
    """Repli sans sqlglot : meme liste blanche de tables, y compris apres JOIN."""
    _verifier(assistant._is_safe_regex, "regex")


def test_extract_sql_garde_les_requetes_aerees() -> None:
    aere = "SELECT motif, COUNT(*)\nFROM retours\n\nGROUP BY motif\n\nORDER BY 2 DESC\nLIMIT 5"
    obtenu = extract_sql(aere)
    assert "ORDER BY" in obtenu.upper(), f"requete tronquee : {obtenu!r}"
    assert "LIMIT" in obtenu.upper(), f"requete tronquee : {obtenu!r}"


def test_extract_sql_coupe_la_prose() -> None:
    bavard = (
        "SELECT COUNT(*) FROM commandes\n\n"
        "Cette requete compte le nombre total de commandes."
    )
    obtenu = extract_sql(bavard)
    assert obtenu == "SELECT COUNT(*) FROM commandes", repr(obtenu)


def test_extract_sql_retire_les_balises() -> None:
    assert extract_sql("```sql\nSELECT 1 FROM commandes\n```") == "SELECT 1 FROM commandes"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} tests passes, {len(AUTORISEES)} requetes autorisees, "
          f"{len(REFUSEES)} requetes bloquees.")
