import numpy as np  # Importation de la bibliothèque NumPy pour la manipulation des tableaux et calculs mathématiques

#============================================
# Filtrage collaboratif basé sur les utilisateurs (User-Based Collaborative Filtering)
#============================================
# Cet algorithme recommande des films en utilisant les k plus proches voisins d'un utilisateur.
# Il utilise la similarité cosinus pour identifier les utilisateurs les plus similaires.
# 
# Fonctionnement :
# 1. Calcul de la similarité entre utilisateurs avec la fonction cosinus.
# 2. Prédiction des notes manquantes d'un utilisateur en prenant une moyenne pondérée des notes de ses k voisins.
# 3. Recommandation du film ayant la meilleure note prédite parmi ceux non notés.
# 4. Complétion de la matrice des notes pour tous les utilisateurs.

#============================================
# Fonction cosinus(M_train, u1, u2)
#============================================
def cosinus(M_train, u1, u2):
    """
    Calcule la similarité cosinus entre deux utilisateurs u1 et u2.
    """
    # Sélection des films notés par les deux utilisateurs (pas de NaN)
    inds_movie = np.where(np.sum(np.isnan(M_train[[u1, u2], ]), axis=0) == 0)[0]
    
    if len(inds_movie) != 0:  # Vérifie s'il existe des films en commun
        n1 = M_train[u1, inds_movie]  # Notes de u1 pour ces films
        n2 = M_train[u2, inds_movie]  # Notes de u2 pour ces films
        
        # Calcul de la similarité cosinus
        cos = sum(n1 * n2) / (np.sqrt(sum(n1 ** 2)) * np.sqrt(sum(n2 ** 2)))
        return cos
    else:
        return 0  # Retourne 0 si aucun film en commun

#============================================
# Fonction complete_a_user_knn(M_train, id_user, k)
#============================================
def complete_a_user(M_train, id_user, k):
    """
    Complète les notes manquantes d'un utilisateur en utilisant la moyenne pondérée des k plus proches voisins.
    """
    scores = np.zeros(M_train.shape[1])  # Initialisation du tableau des scores
    
    for id_item in range(M_train.shape[1]):  # Parcours de tous les films
        inds_known = np.where(~np.isnan(M_train[:, id_item]))[0]  # Utilisateurs ayant noté ce film
        
        if np.isnan(M_train[id_user, id_item]):  # Vérifie si l'utilisateur n'a pas noté ce film
            if len(inds_known) > 0:
                # Calcul des similarités cosinus avec les utilisateurs ayant noté ce film
                sims = np.array([cosinus(M_train, id_user, u) for u in inds_known])
                
                if len(inds_known) > k:
                    ind = np.argsort(-sims)  # Trie les similarités par ordre décroissant
                    inds_known = inds_known[ind][:k]  # Sélection des k plus proches voisins
                    sims = sims[ind][:k]
                
                if sum(abs(sims)) != 0:  # Vérifie si les similarités ne sont pas nulles
                    rates = M_train[inds_known, id_item]  # Notes des k voisins pour ce film
                    mean_rates = np.nanmean(M_train[inds_known, :], axis=1)  # Moyenne des notes des voisins
                    
                    # Prédiction de la note en utilisant une moyenne pondérée par la similarité
                    scores[id_item] = np.nanmean(M_train[id_user, :]) + np.sum(sims * (rates - mean_rates)) / sum(abs(sims))
                else:
                    scores[id_item] = np.nanmean(M_train[id_user, :])  # Sinon, moyenne des notes de l'utilisateur
            else:
                scores[id_item] = np.nanmean(M_train[id_user, :])  # Si aucun voisin, moyenne des notes connues
        else:
            scores[id_item] = M_train[id_user, id_item]  # Si l'utilisateur a déjà noté le film, on garde la note existante
    
    return scores  # Retourne les scores prédits

#============================================
# Fonction recommend(M_train, id_user, new=True, k=10)
#============================================
def recommend(M_train, id_user, new=True, k=10):
    """
    Recommande un film à un utilisateur en utilisant les k plus proches voisins.
    """
    scores = complete_a_user(M_train, id_user, k)  # Complète les notes de l'utilisateur
    
    if new:
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0]  # Films non notés par l'utilisateur
        rec_ind_in_unknown = np.argmax(scores[inds_unknown])  # Sélection du meilleur score parmi ces films
        return inds_unknown[rec_ind_in_unknown]
    else:
        return np.argmax(scores)  # Retourne le film avec la note prédite la plus haute

#============================================
# Fonction complete(M_train, k)
#============================================
def complete(M_train, k):
    """
    Complète toute la matrice des évaluations en prédisant toutes les notes manquantes.
    """
    M_completed = np.zeros(M_train.shape)  # Initialisation de la matrice complétée
    
    for id_user in range(M_train.shape[0]):  # Parcours de tous les utilisateurs
        M_completed[id_user, :] = complete_a_user(M_train, id_user, k)  # Complète leurs notes
    
    return M_completed  # Retourne la matrice complétée

