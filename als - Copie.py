import numpy as np

def complete(M_train, k, n_iter=5, lambd=0.1):
    """
    Implémente un système de recommandation basé sur la décomposition de matrice par ALS (Moindres carrés alternés).
    
    Paramètres :
    M_train : matrice d'entraînement (contenant des NaN pour les valeurs manquantes)
    k : nombre de facteurs latents
    n_iter : nombre d'itérations pour la mise à jour des matrices U et V
    lambd : régularisation
    
    Retourne :
    M_pred : matrice prédite (valeurs remplies)
    """
    
    # Initialisation des matrices U et V avec des petites valeurs aléatoires
    n_users, n_items = M_train.shape
    U = np.random.rand(n_users, k)
    V = np.random.rand(n_items, k)
    
    # On effectue n_iter passes pour optimiser les matrices U et V
    for _ in range(n_iter):
        # Mettre à jour U
        for i in range(n_users):
            # Indices des éléments notés pour l'utilisateur i
            rated_items = ~np.isnan(M_train[i, :])
            if rated_items.sum() > 0:
                V_rated = V[rated_items, :]
                M_rated = M_train[i, rated_items]
                # Calculer la mise à jour de U_i
                U[i, :] = np.linalg.solve(V_rated.T @ V_rated + lambd * np.eye(k),
                                          V_rated.T @ M_rated)
        
        # Mettre à jour V
        for j in range(n_items):
            # Indices des utilisateurs ayant noté l'élément j
            rated_users = ~np.isnan(M_train[:, j])
            if rated_users.sum() > 0:
                U_rated = U[rated_users, :]
                M_rated = M_train[rated_users, j]
                # Calculer la mise à jour de V_j
                V[j, :] = np.linalg.solve(U_rated.T @ U_rated + lambd * np.eye(k),
                                          U_rated.T @ M_rated)
    
    # Calculer la matrice prédite M_pred
    M_pred = U @ V.T
    return M_pred


def recommend(M_train, id_user, new=True, k=10, n_iter=5, lambd=0.1):
    """
    Recommande un film pour un utilisateur donné.
    
    Paramètres :
    M_train : matrice d'entraînement (contenant des NaN pour les valeurs manquantes)
    id_user : identifiant de l'utilisateur pour lequel recommander un film
    new : True si le film recommandé ne doit pas avoir été noté par l'utilisateur
    k : nombre de facteurs latents
    n_iter : nombre d'itérations pour la mise à jour des matrices U et V
    lambd : régularisation
    
    Retourne :
    id_movie : identifiant du film recommandé
    """
    
    # Compléter la matrice M_train pour prédire les valeurs manquantes
    M_pred = complete(M_train, k, n_iter, lambd)
    
    # Si new=True, on exclut les films déjà notés par l'utilisateur
    if new:
        # Trouver les films déjà notés (Non-NaN dans M_train)
        rated_movies = ~np.isnan(M_train[id_user, :])  # films notés
        M_pred[id_user, rated_movies] = -np.inf  # On met une valeur très basse pour les films déjà notés
    
    # Trouver le film avec la note la plus élevée dans la prédiction
    id_movie = np.argmax(M_pred[id_user, :])
    
    # Retourner l'id du film recommandé
    return id_movie
