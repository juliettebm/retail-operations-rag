"""
Entraînement SQL - 30 questions sur data/operations.sqlite.

30 questions métier, classées par construction SQL testée, du plus simple
(COUNT, filtre) au plus composé (jointure à 3 tables, plusieurs
contraintes). Chaque question porte son SQL de référence, à écrire
soi-même avant de vérifier.

Usage :
    python entrainement_sql.py            # vérifie que toutes les requêtes s'exécutent
    python entrainement_sql.py --resultats # affiche aussi le résultat de chaque requête
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "operations.sqlite"

# Chaque entrée : (question, SQL de référence, constructions testées).
QUESTIONS = [
    # --- COUNT ---------------------------------------------------------
    ("Combien de magasins Maison Kurt existent au total ?",
     "SELECT COUNT(*) FROM magasins",
     ["COUNT"]),
    ("Combien de commandes ont été passées en magasin ?",
     "SELECT COUNT(*) FROM commandes WHERE canal = 'magasin'",
     ["COUNT", "filtre"]),
    ("Combien de retours sont hors délai ?",
     "SELECT COUNT(*) FROM retours WHERE dans_delai = 0",
     ["COUNT", "filtre"]),

    # --- SUM -------------------------------------------------------------
    ("Quel est le montant total de toutes les commandes ?",
     "SELECT ROUND(SUM(montant),2) FROM commandes",
     ["SUM"]),
    ("Quel est le montant total des commandes soldées ?",
     "SELECT ROUND(SUM(montant),2) FROM commandes WHERE type_article = 'soldé'",
     ["SUM", "filtre"]),

    # --- AVG ---------------------------------------------------------
    ("Quel est le montant moyen d'une commande ?",
     "SELECT ROUND(AVG(montant),2) FROM commandes",
     ["AVG"]),
    ("Quel est le délai moyen de traitement des retours hors délai ?",
     "SELECT ROUND(AVG(delai_traitement_jours),2) FROM retours WHERE dans_delai = 0",
     ["AVG", "filtre"]),

    # --- Filtres ---------------------------------------------------------
    ("Combien de commandes dépassent 120 euros ?",
     "SELECT COUNT(*) FROM commandes WHERE montant > 120",
     ["filtre"]),
    ("Combien de commandes soldées ont été passées en ligne, avec livraison à domicile ?",
     "SELECT COUNT(*) FROM commandes WHERE type_article = 'soldé' AND canal = 'en ligne - livraison'",
     ["filtre", "plusieurs contraintes"]),
    ("Combien de commandes ont été payées par carte cadeau ou sont encore en cours ?",
     "SELECT COUNT(*) FROM commandes WHERE moyen_paiement = 'carte cadeau' OR statut = 'en cours'",
     ["filtre", "plusieurs contraintes"]),

    # --- GROUP BY ---------------------------------------------------------
    ("Combien de commandes par canal ?",
     "SELECT canal, COUNT(*) FROM commandes GROUP BY canal",
     ["GROUP BY", "COUNT"]),
    ("Quel est le montant moyen par type d'article ?",
     "SELECT type_article, ROUND(AVG(montant),2) FROM commandes GROUP BY type_article",
     ["GROUP BY", "AVG"]),
    ("Quel est le montant total remboursé par moyen de paiement ?",
     "SELECT moyen_paiement, ROUND(SUM(montant),2) FROM remboursements GROUP BY moyen_paiement",
     ["GROUP BY", "SUM"]),
    ("Quels motifs de retour comptent plus de 100 occurrences ?",
     "SELECT motif, COUNT(*) FROM retours GROUP BY motif HAVING COUNT(*) > 100",
     ["GROUP BY", "HAVING"]),

    # --- ORDER BY / LIMIT ---------------------------------------------------------
    ("Quelles sont les 10 commandes les plus chères ?",
     "SELECT commande_id, montant FROM commandes ORDER BY montant DESC LIMIT 10",
     ["ORDER BY", "LIMIT"]),
    ("Quel est le retour le plus ancien ?",
     "SELECT retour_id, date_retour FROM retours ORDER BY date_retour ASC LIMIT 1",
     ["ORDER BY", "LIMIT"]),
    ("Quels sont les 3 motifs de retour les plus fréquents ?",
     "SELECT motif, COUNT(*) AS n FROM retours GROUP BY motif ORDER BY n DESC LIMIT 3",
     ["GROUP BY", "ORDER BY", "LIMIT"]),
    ("Donne les 5 premières commandes enregistrées.",
     "SELECT commande_id FROM commandes ORDER BY commande_id ASC LIMIT 5",
     ["ORDER BY", "LIMIT"]),

    # --- JOIN 2 tables ---------------------------------------------------------
    ("Quel est le magasin de chaque commande retournée ? (20 premières)",
     """SELECT c.commande_id, m.nom FROM commandes c JOIN magasins m ON c.magasin_id = m.magasin_id
        WHERE c.statut = 'retournée' LIMIT 20""",
     ["JOIN 2 tables", "filtre", "LIMIT"]),
    ("Combien de commandes par magasin ?",
     "SELECT m.nom, COUNT(*) FROM commandes c JOIN magasins m ON c.magasin_id = m.magasin_id GROUP BY m.nom",
     ["JOIN 2 tables", "GROUP BY", "COUNT"]),
    ("Quel est le montant moyen des commandes par magasin ?",
     """SELECT m.nom, ROUND(AVG(c.montant),2) FROM commandes c JOIN magasins m ON c.magasin_id = m.magasin_id
        GROUP BY m.nom""",
     ["JOIN 2 tables", "GROUP BY", "AVG"]),
    ("Quels sont les 5 magasins avec le plus de commandes en livraison à domicile ?",
     """SELECT m.nom, COUNT(*) AS n FROM commandes c JOIN magasins m ON c.magasin_id = m.magasin_id
        WHERE c.canal = 'en ligne - livraison' GROUP BY m.nom ORDER BY n DESC LIMIT 5""",
     ["JOIN 2 tables", "filtre", "GROUP BY", "ORDER BY", "LIMIT"]),
    ("Quel est le délai moyen de traitement des retours pour les commandes soldées ?",
     """SELECT ROUND(AVG(r.delai_traitement_jours),2) FROM retours r JOIN commandes c ON r.commande_id = c.commande_id
        WHERE c.type_article = 'soldé'""",
     ["JOIN 2 tables", "AVG", "filtre"]),
    ("Quel est le montant moyen remboursé par motif de retour ?",
     """SELECT r.motif, ROUND(AVG(rb.montant),2) FROM remboursements rb JOIN retours r ON rb.retour_id = r.retour_id
        GROUP BY r.motif""",
     ["JOIN 2 tables", "GROUP BY", "AVG"]),

    # --- JOIN 3 tables ---------------------------------------------------------
    ("Combien de remboursements par carte bancaire concernent des commandes soldées ?",
     """SELECT COUNT(*) FROM remboursements rb
        JOIN retours r ON rb.retour_id = r.retour_id
        JOIN commandes c ON r.commande_id = c.commande_id
        WHERE rb.moyen_paiement = 'carte bancaire' AND c.type_article = 'soldé'""",
     ["JOIN 3 tables", "COUNT", "plusieurs contraintes"]),
    ("Quel est le montant moyen remboursé par canal de commande ?",
     """SELECT c.canal, ROUND(AVG(rb.montant),2) FROM remboursements rb
        JOIN retours r ON rb.retour_id = r.retour_id
        JOIN commandes c ON r.commande_id = c.commande_id
        GROUP BY c.canal""",
     ["JOIN 3 tables", "GROUP BY", "AVG"]),
    ("Quels sont les 5 magasins avec le plus de retours ?",
     """SELECT m.nom, COUNT(*) AS n FROM retours r
        JOIN commandes c ON r.commande_id = c.commande_id
        JOIN magasins m ON c.magasin_id = m.magasin_id
        GROUP BY m.nom ORDER BY n DESC LIMIT 5""",
     ["JOIN 3 tables", "GROUP BY", "ORDER BY", "LIMIT"]),

    # --- Plusieurs contraintes ---------------------------------------------------------
    ("Quels sont les 3 magasins avec le taux de retour le plus élevé, sur les commandes standards vendues en ligne ?",
     """SELECT m.nom, ROUND(100.0*SUM(CASE WHEN c.statut='retournée' THEN 1 ELSE 0 END)/COUNT(*),1) AS taux
        FROM commandes c JOIN magasins m ON c.magasin_id = m.magasin_id
        WHERE c.type_article = 'standard' AND c.canal != 'magasin'
        GROUP BY m.nom ORDER BY taux DESC LIMIT 3""",
     ["JOIN 2 tables", "plusieurs contraintes", "GROUP BY", "ORDER BY", "LIMIT"]),
    ("Quel est le délai moyen de remboursement pour les retours dans les délais, motif article défectueux, payés par carte bancaire ?",
     """SELECT ROUND(AVG(rb.delai_jours),2) FROM remboursements rb
        JOIN retours r ON rb.retour_id = r.retour_id
        WHERE r.motif = 'article défectueux' AND r.dans_delai = 1 AND rb.moyen_paiement = 'carte bancaire'""",
     ["JOIN 2 tables", "AVG", "plusieurs contraintes"]),
    ("Quels magasins ont un montant total de commandes soldées supérieur à 2000 euros, du plus élevé au plus faible ?",
     """SELECT m.nom, ROUND(SUM(c.montant),2) AS total FROM commandes c
        JOIN magasins m ON c.magasin_id = m.magasin_id
        WHERE c.type_article = 'soldé'
        GROUP BY m.nom HAVING SUM(c.montant) > 2000 ORDER BY total DESC""",
     ["JOIN 2 tables", "filtre", "GROUP BY", "HAVING", "ORDER BY", "plusieurs contraintes"]),
]


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    afficher_resultats = "--resultats" in sys.argv

    for i, (question, sql, concepts) in enumerate(QUESTIONS, 1):
        try:
            df = pd.read_sql(sql, con)
            statut = "OK"
        except Exception as exc:
            df, statut = None, f"ERREUR : {exc}"
        print(f"{i:2d}. [{', '.join(concepts):45s}] {statut:6s} {question}")
        if afficher_resultats and df is not None:
            print(df.head(5).to_string(index=False))
            print()

    print(f"\n{len(QUESTIONS)} questions.")


if __name__ == "__main__":
    main()
