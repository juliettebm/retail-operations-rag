"""
Jeu d'évaluation du text-to-SQL : 17 questions métier, chacune associée à
un SQL de référence et à un indicateur de jointure. Utilisé par
`03_text_to_sql_operations.ipynb` (section 8, Évaluation).
"""

EVAL_SET = [
    ("Combien de commandes ont le statut livrée ?",
     "SELECT COUNT(*) FROM commandes WHERE statut = 'livrée'", False),
    ("Combien de retours ont été enregistrés au total ?",
     "SELECT COUNT(*) FROM retours", False),
    ("Quel est le montant total des remboursements effectués par carte cadeau ?",
     "SELECT ROUND(SUM(montant),2) FROM remboursements WHERE moyen_paiement = 'carte cadeau'", False),
    ("Quel est le délai moyen de remboursement pour les paiements par carte bancaire ?",
     "SELECT ROUND(AVG(delai_jours),2) FROM remboursements WHERE moyen_paiement = 'carte bancaire'", False),
    ("Combien de commandes ont été passées via le canal en ligne click&collect ?",
     "SELECT COUNT(*) FROM commandes WHERE canal = 'en ligne - click&collect'", False),
    ("Quel est le motif de retour le plus fréquent ?",
     "SELECT motif FROM retours GROUP BY motif ORDER BY COUNT(*) DESC LIMIT 1", False),
    ("Combien de retours sont hors délai ?",
     "SELECT COUNT(*) FROM retours WHERE dans_delai = 0", False),
    ("Quel est le montant moyen d'une commande soldée ?",
     "SELECT ROUND(AVG(montant),2) FROM commandes WHERE type_article = 'soldé'", False),
    ("Combien de magasins Maison Kurt existent au total ?",
     "SELECT COUNT(*) FROM magasins", False),
    ("Combien de commandes sont encore en cours ?",
     "SELECT COUNT(*) FROM commandes WHERE statut = 'en cours'", False),
    ("Quel pourcentage des commandes en ligne - livraison sont en retard ?",
     "SELECT ROUND(100.0*AVG(en_retard),1) FROM commandes WHERE canal='en ligne - livraison' AND en_retard IS NOT NULL", False),
    ("Combien de commandes retournées le magasin de Marseille a-t-il enregistrées ?",
     """SELECT COUNT(*) FROM commandes c JOIN magasins m ON c.magasin_id = m.magasin_id
        WHERE m.ville = 'Marseille' AND c.statut = 'retournée' """, True),
    ("Quel est le délai moyen de traitement des retours pour les commandes passées en magasin ?",
     """SELECT ROUND(AVG(r.delai_traitement_jours),2) FROM retours r
        JOIN commandes c ON r.commande_id = c.commande_id WHERE c.canal = 'magasin' """, True),
    ("Quels sont les 5 magasins avec le montant total de commandes le plus élevé ?",
     """SELECT m.nom FROM commandes c JOIN magasins m ON c.magasin_id = m.magasin_id
        GROUP BY m.nom ORDER BY SUM(c.montant) DESC LIMIT 5""", True),
    ("Quel est le montant moyen remboursé pour le motif article défectueux ?",
     """SELECT ROUND(AVG(rb.montant),2) FROM remboursements rb
        JOIN retours r ON rb.retour_id = r.retour_id WHERE r.motif = 'article défectueux' """, True),
    ("Quel est le délai moyen entre le retour et le remboursement pour les commandes en ligne ?",
     """SELECT ROUND(AVG(rb.delai_jours),2) FROM remboursements rb
        JOIN retours r ON rb.retour_id = r.retour_id
        JOIN commandes c ON r.commande_id = c.commande_id WHERE c.canal != 'magasin' """, True),
    ("Combien de remboursements par carte bancaire concernent des commandes soldées ?",
     """SELECT COUNT(*) FROM remboursements rb
        JOIN retours r ON rb.retour_id = r.retour_id
        JOIN commandes c ON r.commande_id = c.commande_id
        WHERE rb.moyen_paiement = 'carte bancaire' AND c.type_article = 'soldé' """, True),
]
