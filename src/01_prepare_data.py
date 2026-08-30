"""
01 - Generation des donnees operationnelles fictives (Maison Kurt).

Produit `data/operations.sqlite`, quatre tables normalisees : magasins,
commandes, retours, remboursements. Les regles de generation reprennent
celles ecrites noir sur blanc dans les guides operationnels (`data/*.md`) :
delai de retour de 30 jours pour un article standard, 14 jours pour un
article solde, remboursement sur carte bancaire plus lent qu'un credit sur
carte cadeau, deux motifs de retour reserves aux commandes en ligne.

Rien n'est mesure sur de vraies commandes : les distributions ci-dessous
sont choisies pour ressembler a des donnees d'exploitation plausibles, pas
pour reproduire un chiffre reel de Maison Kurt (fictif).

Usage :
    python src/01_prepare_data.py
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "operations.sqlite"

SEED = 42
N_COMMANDES = 6000
DATE_DEBUT = date(2025, 1, 1)
DATE_SNAPSHOT = date(2025, 12, 15)   # "aujourd'hui" du jeu de donnees

MAGASINS = [
    (1, "Maison Kurt Paris Haussmann", "Paris"),
    (2, "Maison Kurt Lyon Part-Dieu", "Lyon"),
    (3, "Maison Kurt Marseille Prado", "Marseille"),
    (4, "Maison Kurt Bordeaux Chartrons", "Bordeaux"),
    (5, "Maison Kurt Lille Centre", "Lille"),
    (6, "Maison Kurt Toulouse Capitole", "Toulouse"),
    (7, "Maison Kurt Nantes Centre", "Nantes"),
    (8, "Maison Kurt Strasbourg Centre", "Strasbourg"),
]
MAGASIN_LENT = 3   # Marseille Prado : delai de livraison structurellement plus long

CANAUX = ["magasin", "en ligne - livraison", "en ligne - click&collect"]
P_CANAL = [0.45, 0.40, 0.15]
SLA_JOURS = {"magasin": 0, "en ligne - click&collect": 2, "en ligne - livraison": 5}

MOTIFS_COMMUNS = ["ne convient pas", "taille inadaptée", "article défectueux", "changement d'avis"]
MOTIFS_EN_LIGNE = ["colis endommagé à réception", "article incorrect reçu"]


def _bornes(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def generer_commandes(rng: np.random.Generator) -> pd.DataFrame:
    magasin_id = rng.integers(1, len(MAGASINS) + 1, N_COMMANDES)
    canal = rng.choice(CANAUX, N_COMMANDES, p=P_CANAL)
    type_article = rng.choice(["standard", "soldé"], N_COMMANDES, p=[0.85, 0.15])
    moyen_paiement = rng.choice(["carte bancaire", "carte cadeau"], N_COMMANDES, p=[0.75, 0.25])

    jours_ecoules = rng.integers(0, (DATE_SNAPSHOT - DATE_DEBUT).days + 1, N_COMMANDES)
    date_commande = np.array([DATE_DEBUT + timedelta(days=int(j)) for j in jours_ecoules])

    montant = np.where(
        type_article == "standard",
        _bornes(rng.normal(65, 25, N_COMMANDES), 10, 250),
        _bornes(rng.normal(35, 15, N_COMMANDES), 8, 150),
    ).round(2)

    delai_sla = np.array([SLA_JOURS[c] for c in canal])

    # Delai reel de livraison : Marseille Prado est structurellement plus lent
    # sur la livraison a domicile (probleme logistique local), les autres
    # magasins tournent autour du SLA affiche.
    delai_reel = np.empty(N_COMMANDES)
    for i, c in enumerate(canal):
        if c == "magasin":
            delai_reel[i] = 0
        elif c == "en ligne - click&collect":
            delai_reel[i] = _bornes(rng.normal(1.8, 1.0), 0, 8)
        else:  # en ligne - livraison
            if magasin_id[i] == MAGASIN_LENT:
                delai_reel[i] = _bornes(rng.normal(7.5, 2.5), 1, 20)
            else:
                delai_reel[i] = _bornes(rng.normal(4.3, 1.8), 1, 16)
    delai_reel = delai_reel.round().astype(int)

    date_livraison_prevue = np.array(
        [dc + timedelta(days=int(d)) for dc, d in zip(date_commande, delai_reel)]
    )
    livree = date_livraison_prevue <= DATE_SNAPSHOT

    statut = np.where(canal == "magasin", "livrée", np.where(livree, "livrée", "en cours"))
    delai_traitement = np.where((canal == "magasin") | livree, delai_reel, np.nan)
    en_retard = np.where(
        (canal != "magasin") & livree,
        delai_traitement > delai_sla,
        np.nan,
    )

    return pd.DataFrame({
        "commande_id": np.arange(1, N_COMMANDES + 1),
        "magasin_id": magasin_id,
        "canal": canal,
        "date_commande": [d.isoformat() for d in date_commande],
        "type_article": type_article,
        "moyen_paiement": moyen_paiement,
        "montant": montant,
        "statut": statut,
        "delai_traitement_jours": delai_traitement,
        "delai_sla_jours": delai_sla,
        "en_retard": en_retard,
    })


def generer_retours(rng: np.random.Generator, commandes: pd.DataFrame) -> pd.DataFrame:
    eligibles = commandes[commandes["statut"] == "livrée"].copy()

    p_retour = pd.Series(0.10, index=eligibles.index)
    p_retour += np.where(eligibles["canal"] == "en ligne - click&collect", 0.05, 0.0)
    p_retour += np.where(eligibles["canal"] == "en ligne - livraison", 0.15, 0.0)
    # Combinaison "soldé + livraison a domicile" : achat impulsif sans essayage,
    # taux de retour nettement plus eleve que les autres combinaisons canal/type.
    p_retour += np.where(
        (eligibles["type_article"] == "soldé") & (eligibles["canal"] == "en ligne - livraison"),
        0.20,
        np.where(eligibles["type_article"] == "soldé", 0.03, 0.0),
    )
    tire = rng.random(len(eligibles))
    retournees = eligibles[tire < p_retour.values].copy()

    n = len(retournees)
    en_ligne_mask = retournees["canal"] != "magasin"

    motifs = []
    for is_en_ligne in en_ligne_mask:
        pool = MOTIFS_COMMUNS + (MOTIFS_EN_LIGNE if is_en_ligne else [])
        poids = [0.32, 0.28, 0.22, 0.18] if not is_en_ligne else [0.24, 0.21, 0.17, 0.13, 0.14, 0.11]
        motifs.append(rng.choice(pool, p=poids))

    fenetre = np.where(retournees["type_article"].values == "standard", 30, 14)
    # Tirage large : ~15% des retours tombent hors delai (cas "depassement du
    # delai" documente dans le guide retours/echanges, section 6).
    delta_jours = rng.integers(1, (fenetre + 10) + 1)
    dates_commande = pd.to_datetime(retournees["date_commande"]).values
    date_retour = dates_commande + delta_jours.astype("timedelta64[D]")
    dans_delai = delta_jours <= fenetre

    delai_traitement = _bornes(rng.normal(2.0, 1.5, n), 0, 10).round().astype(int)

    return pd.DataFrame({
        "retour_id": np.arange(1, n + 1),
        "commande_id": retournees["commande_id"].values,
        "date_retour": pd.to_datetime(date_retour).strftime("%Y-%m-%d"),
        "motif": motifs,
        "delai_traitement_jours": delai_traitement,
        "dans_delai": dans_delai.astype(int),
    })


def generer_remboursements(rng: np.random.Generator, commandes: pd.DataFrame, retours: pd.DataFrame) -> pd.DataFrame:
    fusion = retours.merge(
        commandes[["commande_id", "moyen_paiement", "montant"]], on="commande_id", how="left"
    )

    p_conforme = np.where(
        fusion["motif"] == "article défectueux",
        0.97,
        np.where(fusion["dans_delai"] == 1, 0.95, 0.20),
    )
    tire = rng.random(len(fusion))
    conformes = fusion[tire < p_conforme].copy()

    n = len(conformes)
    est_cb = conformes["moyen_paiement"].values == "carte bancaire"
    delai = np.where(
        est_cb,
        _bornes(rng.normal(4.0, 1.5, n), 1, 12),
        _bornes(rng.normal(1.0, 0.5, n), 0, 4),
    ).round().astype(int)

    date_retour = pd.to_datetime(conformes["date_retour"]).values
    date_remboursement = date_retour + delai.astype("timedelta64[D]")

    return pd.DataFrame({
        "remboursement_id": np.arange(1, n + 1),
        "retour_id": conformes["retour_id"].values,
        "moyen_paiement": conformes["moyen_paiement"].values,
        "montant": conformes["montant"].values,
        "date_remboursement": pd.to_datetime(date_remboursement).strftime("%Y-%m-%d"),
        "delai_jours": delai,
    })


def main() -> None:
    rng = np.random.default_rng(SEED)

    magasins = pd.DataFrame(MAGASINS, columns=["magasin_id", "nom", "ville"])
    commandes = generer_commandes(rng)
    retours = generer_retours(rng, commandes)
    commandes.loc[commandes["commande_id"].isin(retours["commande_id"]), "statut"] = "retournée"
    remboursements = generer_remboursements(rng, commandes, retours)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        magasins.to_sql("magasins", con, index=False)
        commandes.to_sql("commandes", con, index=False)
        retours.to_sql("retours", con, index=False)
        remboursements.to_sql("remboursements", con, index=False)

    print(f"Base ecrite : {DB_PATH}")
    print(f"  magasins        : {len(magasins)}")
    print(f"  commandes       : {len(commandes)}")
    print(f"  retours         : {len(retours)}")
    print(f"  remboursements  : {len(remboursements)}")


if __name__ == "__main__":
    main()
