##============================================
##============================================
## fonctions utiles à l'évaluation des algorithmes
##============================================
##============================================

# * get_train_val(M, prop=0_8)
# * RMSE(M_completed, M_star)
# * quantitative_comparison(scoring_fn, M_star, recommmenders, prop=0_8, nrep=10)



import numpy as np
from time import time
import pandas as pd
from scipy.stats import spearmanr

##============================================
## get_train_val(M, prop=0_8)
##============================================
def get_train_val(M, prop=0.8):
  n, m = M.shape
  M_train = np.nan * np.ones((n, m), dtype=float)
  M_validation = M.copy()
  
  for id_user in range(n):
    inds_star = np.where(~np.isnan(M[id_user, :]))[0]
    if len(inds_star)==1:
      inds = inds_star
    else:
      inds = np.random.choice(inds_star, max(1, int(prop*len(inds_star))), replace=False)
    M_train[id_user, inds] = M[id_user, inds]
    M_validation[id_user, inds] = np.nan
  
  return (M_train, M_validation)




##============================================
## RMSE(M_completed, M_star)
##============================================
def RMSE(M_completed, M_star):
  inds = ~np.isnan(M_star)
  return np.sqrt(np.mean((M_completed[inds] - M_star[inds])**2))



##============================================
## MAE(M_completed, M_star)
##============================================
def MAE(M_completed, M_star):
    inds = ~np.isnan(M_star)
    return np.mean(np.abs(M_completed[inds] - M_star[inds]))


##============================================
## Precision at k
##============================================
def precision_at_k(M_completed, M_star, k=10):
    """
    Precision at k: proportion des recommandations pertinentes parmi les k premiers éléments recommandés.
    """
    n, m = M_star.shape
    precision_scores = []
    for i in range(n):
        # Top-K recommandations basées sur les prédictions
        top_k_indices = np.argsort(M_completed[i, :])[-k:]
        relevant_items = M_star[i, top_k_indices] > 0  # suppose que la valeur positive indique un item pertinent
        precision_scores.append(np.mean(relevant_items))
    return np.mean(precision_scores)


##============================================
## Recall at k
##============================================
def recall_at_k(M_completed, M_star, k=10):
    """
    Recall at k: proportion des éléments pertinents qui ont été recommandés parmi les k premiers.
    """
    n, m = M_star.shape
    recall_scores = []
    for i in range(n):
        # Top-K recommandations basées sur les prédictions
        top_k_indices = np.argsort(M_completed[i, :])[-k:]
        relevant_items = M_star[i, :] > 0  # suppose que la valeur positive indique un item pertinent
        recall_scores.append(np.mean(relevant_items[top_k_indices]))
    return np.mean(recall_scores)


##============================================
## User-space Coverage
##============================================
def user_space_coverage(M_completed, M_star):
    n, m = M_star.shape
    recommended_users = np.sum(~np.isnan(M_completed), axis=1) > 0
    return np.mean(recommended_users)


##============================================
## Item-space Coverage
##============================================
def item_space_coverage(M_completed, M_star):
    n, m = M_star.shape
    recommended_items = np.sum(~np.isnan(M_completed), axis=0) > 0
    return np.mean(recommended_items)


##============================================
## Ranking based on predicted vs actual ratings
##============================================
def ranking_based_on_ratings(M_completed, M_star):
    """
    Measure la corrélation entre les classements prédit et réels en utilisant le coefficient de Spearman.
    """
    n, m = M_star.shape
    correlation_scores = []
    for i in range(n):
        # Classement des éléments en fonction des évaluations prédites et réelles
        pred_ranks = np.argsort(M_completed[i, :])
        true_ranks = np.argsort(M_star[i, :])
        correlation, _ = spearmanr(pred_ranks, true_ranks)
        correlation_scores.append(correlation)
    return np.mean(correlation_scores)
##============================================
## quantitative_comparison(scoring_fn, M_star, recommmenders, prop=0_8, nrep=10)
##============================================

def quantitative_comparison(scoring_fn, M_star, recommenders, prop=0.8, nrep=10):
  scores = np.zeros((len(recommenders), nrep))
  scores_train = np.zeros((len(recommenders), nrep))
  computation_time = np.zeros((len(recommenders), nrep))
  for id_rep in range(nrep):
    M_train, M_validation = get_train_val(M_star, prop)
    for id_rec in range(len(recommenders)):
      ptm = time()
      M_completed = recommenders[id_rec]['fn'](M_train)
      computation_time[id_rec, id_rep] = (time() - ptm)
      scores[id_rec, id_rep] = scoring_fn(M_completed, M_validation)
      scores_train[id_rec, id_rep] = scoring_fn(M_completed, M_train)


  return pd.DataFrame({
          'recommender': [rec['label'] for rec in recommenders],
          'validation score': np.mean(scores, axis=1),
          'training score': np.mean(scores_train, axis=1),
          'computation time': np.mean(computation_time, axis=1)
          })



