import numpy as np 

def replaceNA_with_zeros(M_train):
    """
    Remplace les valeurs manquantes (NaN) par des zéros pour permettre la factorisation.
    """
    if M_train is None or M_train.size == 0:
        raise ValueError("La matrice d'entrée est vide ou None.")
    
    return np.nan_to_num(M_train, nan=0.0)

def replaceNA_with_mean(M_train):
    """
    Remplace les valeurs manquantes (NaN) par la moyenne de chaque colonne.
    Si la colonne est vide (toutes les valeurs sont NaN), elle est remplie avec 0.
    """
    if M_train is None or M_train.size == 0:
        raise ValueError("La matrice d'entrée est vide ou None.")
    
    # Calculer la moyenne de chaque colonne, en ignorant les NaN
    col_mean = np.nanmean(M_train, axis=0)
    
    # Traiter les colonnes vides (toutes les valeurs sont NaN)
    col_mean[np.isnan(col_mean)] = 0  # Remplacer les moyennes NaN par 0
    
    # Remplacer les NaN par la moyenne de la colonne correspondante
    inds = np.isnan(M_train)
    M_train[inds] = np.take(col_mean, np.where(inds)[1])
    
    return M_train





def complete(M_train, k, replaceNA_fn=replaceNA_with_zeros):
    """
    Factorise la matrice M_train avec SVD après remplacement des NaN.
    Retourne une matrice approximative en utilisant les k plus grandes valeurs singulières.
    """
    if k <= 0:
        raise ValueError("k doit être un entier positif.")

    M_filled = replaceNA_fn(M_train)
    
    # Décomposition en valeurs singulières
    U, S, Vt = np.linalg.svd(M_filled, full_matrices=False)
    
    # On tronque les matrices pour garder les k premiers composants
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    # Reconstruction de la matrice approximative
    M_approx = U_k @ S_k @ Vt_k
    return M_approx

def recommend(M_train, id_user, new=True, k=10, replaceNA_fn=replaceNA_with_zeros):
    """
    Recommande des articles à un utilisateur donné en prédisant ses notes.
    
    - `new=True` : Recommande uniquement les articles non notés par l'utilisateur.
    - `new=False` : Recommande les articles les mieux notés (tous confondus).
    """
    if id_user < 0 or id_user >= M_train.shape[0]:
        raise ValueError("id_user est hors des limites de la matrice.")

    M_approx = complete(M_train, k, replaceNA_fn)
    
    # Récupérer les prédictions de notation pour l'utilisateur
    user_ratings = M_approx[id_user]
    
    if new:
        # Identifier les indices des articles non notés
        user_original_ratings = M_train[id_user]
        unseen_items = np.where(np.isnan(user_original_ratings))[0]

        if unseen_items.size == 0:
            return []  # Aucun article à recommander

        # Trier les prédictions par ordre décroissant
        recommendations = unseen_items[np.argsort(user_ratings[unseen_items])[::-1]]
    else:
        # Trier tous les articles selon les scores les plus élevés
        recommendations = np.argsort(user_ratings)[::-1]
    
    return recommendations[0]
