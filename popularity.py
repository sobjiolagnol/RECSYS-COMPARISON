import numpy as np

##============================================
# Fonction de recommandation basée sur la popularité
##============================================
def recommend(M_train, id_user, new=True):
    # Calcul de la moyenne des notes pour chaque film dans M_train, en ignorant les NaN
    # Cela permet d'obtenir une estimation de la popularité moyenne de chaque film
    scores = np.nanmean(M_train, axis=0)  # Vecteur contenant la moyenne des notes par film
    
    if new:
        # Si "new=True", recommander un film que l'utilisateur n'a pas encore évalué
        # On identifie les indices des films non évalués par l'utilisateur (NaN dans M_train)
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0]  # Indices des films inconnus pour cet utilisateur
        
        # Sélection du film le plus populaire parmi ceux non évalués
        rec_ind_in_unknown = np.nanargmax(scores[inds_unknown])  # Trouve le film le plus populaire parmi ceux non vus
        
        # Retourne l'indice du film recommandé
        return inds_unknown[rec_ind_in_unknown]
    else:
        # Si "new=False", recommander le film le plus populaire globalement
        return np.nanargmax(scores)  # Retourne l'indice du film avec la meilleure moyenne globale

# Lien avec compare.ipynb :
# Cette fonction est utilisée pour comparer les recommandations basées sur la popularité avec d'autres approches.
# Lorsque new=True, elle recommande des films non encore vus, ce qui permet une évaluation plus pertinente.

# Lien avec data.py :
# La matrice M_train est généralement chargée via data.py. Elle contient les évaluations des films par utilisateur,
# servant de base au calcul des scores moyens des films pour la recommandation.

# Lien avec eval.py :
# La qualité des recommandations produites par cette fonction peut être évaluée via eval.py.
# Des métriques comme le RMSE permettent de mesurer l'efficacité des recommandations générées.

##============================================
# Fonction de complétion des évaluations manquantes
##============================================
def complete(M_train):
    # Calcul de la moyenne des notes par film, en ignorant les NaN
    # Cela permet d'estimer la popularité globale de chaque film
    scores = np.nanmean(M_train, axis=0)  # Vecteur contenant la moyenne des notes de chaque film
    
    # Remplacement des NaN dans scores par 0 (si un film n'a aucune évaluation, on lui donne une note de 0)
    scores[np.isnan(scores)] = 0  # Permet d'éviter les erreurs de calcul sur les films non notés
    
    # Création d'une matrice où chaque ligne (utilisateur) reçoit la moyenne des scores des films
    to_complete = np.ones((M_train.shape[0], 1)) @ scores.reshape((1, -1))  
    # Cette matrice contient les scores moyens des films dupliqués pour tous les utilisateurs
    
    # Création d'une copie de M_train pour ne pas modifier les données originales
    M_completed = M_train.copy()  
    
    # Remplacement des NaN dans M_train par les scores moyens des films
    # Cela permet d'estimer les évaluations manquantes avec la moyenne des autres utilisateurs
    M_completed[np.isnan(M_train)] = to_complete[np.isnan(M_train)]  
    
    # Retour de la matrice complétée, où les valeurs manquantes ont été remplacées
    return M_completed

# Lien avec compare.ipynb :
# Cette fonction est utile pour compléter les matrices de notation dans le notebook de comparaison.
# Elle permet d'obtenir une matrice plus exploitable pour tester les différentes méthodes de recommandation.

# Lien avec data.py :
# M_train est chargé depuis data.py. Cette fonction le modifie en complétant les évaluations manquantes,
# créant ainsi une version utilisable pour différentes approches de recommandation.

# Lien avec eval.py :
# Une fois la matrice complétée, elle peut être évaluée via des métriques comme le RMSE.
# Cela permet de mesurer l'impact de cette complétion sur la précision des recommandations.
