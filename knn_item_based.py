import numpy as np

##============================================
## cosinus_items(M_train, i1, i2)
##============================================
def cosinus_items(M_train, i1, i2):
    # Utilisateurs ayant noté les deux films
    inds_users = np.where(np.sum(np.isnan(M_train[:, [i1, i2]]), axis=1) == 0)[0]
    
    if len(inds_users) != 0:
        n1 = M_train[inds_users, i1]
        n2 = M_train[inds_users, i2]
        cos = sum(n1 * n2) / (np.sqrt(sum(n1**2)) * np.sqrt(sum(n2**2)))
        return cos
    else:
        return 0

##============================================
## complete_a_user_item_based(M_train, id_user, k)
##============================================
def complete_a_user_item_based(M_train, id_user, k):
    scores = np.zeros(M_train.shape[1])
    for id_item in range(M_train.shape[1]):
        if np.isnan(M_train[id_user, id_item]):
            # Films notés par l'utilisateur
            inds_known = np.where(~np.isnan(M_train[id_user, :]))[0]
            if len(inds_known) > 0:
                # Similarité entre le film cible et les films notés
                sims = np.array([cosinus_items(M_train, id_item, i) for i in inds_known])
                
                if len(inds_known) > k:
                    ind = np.argsort(-sims)  # Tri décroissant
                    inds_known = inds_known[ind][:k]
                    sims = sims[ind][:k]
                
                if sum(abs(sims)) != 0:
                    rates = M_train[id_user, inds_known]
                    scores[id_item] = np.sum(sims * rates) / sum(abs(sims))
                else:
                    scores[id_item] = np.nanmean(M_train[id_user, :])  # Moyenne si pas de similarité
            else:
                scores[id_item] = np.nanmean(M_train[id_user, :])  # Moyenne si rien n'est noté
        else:
            scores[id_item] = M_train[id_user, id_item]  # Garder la note existante
    return scores

##============================================
## recommend_item_based(M_train, id_user, new=True, k=10)
##============================================
def recommend_item_based(M_train, id_user, new=True, k=10):
    scores = complete_a_user_item_based(M_train, id_user, k)
    
    if new:
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0]
        rec_ind_in_unknown = np.argmax(scores[inds_unknown])
        return inds_unknown[rec_ind_in_unknown]
    else:
        return np.argmax(scores)

##============================================
## complete_item_based(M_train, k)
##============================================
def complete_item_based(M_train, k):
    M_completed = np.zeros(M_train.shape)
    for id_user in range(M_train.shape[0]):
        M_completed[id_user, :] = complete_a_user_item_based(M_train, id_user, k)
    return M_completed