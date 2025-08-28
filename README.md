
---

#  Système de recommandation – Comparaison d’algorithmes

##  Objectif

Ce projet vise à **comparer différents algorithmes de recommandation** sur un jeu de données de films, à la fois **quantitativement** (métriques de performance) et **qualitativement** (qualité des recommandations proposées aux utilisateurs).

Les algorithmes étudiés incluent :

* **Popularité** (baseline simple)
* **KNN user-based**
* **KNN item-based**
* **SVD** (factorisation en valeurs singulières)
* **SVD avec imputation par la moyenne**
* **ALS** (Alternating Least Squares)

---

##  Organisation du projet

* `compare.ipynb` → **Évaluation quantitative**

  * Recherche des meilleurs hyperparamètres (`k`) pour KNN, SVD, ALS
  * Comparaison selon RMSE, MAE, précision\@k, rappel\@k, coverage, etc.
  * Analyse de l’overfitting sur SVD

* `qualitative_compare.ipynb` → **Évaluation qualitative**

  * Exemples de recommandations pour un utilisateur donné
  * Comparaison de la pertinence perçue des résultats

* `test.ipynb` → **Tests unitaires**

  * Vérifie le bon fonctionnement des modules (`data`, `popularity`, `knn`, `svd`, `als`, `eval`)
  * Contrôle des erreurs de reconstruction (RMSE → tend vers 0 quand k est grand)

* `data.py` → Chargement du dataset et fonctions utilitaires (titres de films, split train/validation).

* `eval.py` → Métriques d’évaluation : RMSE, MAE, précision\@k, rappel\@k, coverage, etc.

* `popularity.py`, `knn.py`, `knn_item_based.py`, `svd.py`, `als.py` → Implémentations des algorithmes de recommandation.

---

##  Installation

### 1. Cloner le projet

```bash
git clone [https://github.com/username/recommandation.git](https://github.com/sobjiolagnol/RECSYS-COMPARISON.git)
cd recommandation
```

### 2. Créer un environnement Python

```bash
python -m venv venv
source venv/bin/activate   # sous Linux/Mac
venv\Scripts\activate      # sous Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## ▶️ Utilisation

### 1. Lancer Jupyter Notebook

```bash
jupyter notebook
```

### 2. Exécuter les notebooks

* `compare.ipynb` → pour la comparaison quantitative
* `qualitative_compare.ipynb` → pour les recommandations sur un utilisateur donné
* `test.ipynb` → pour valider les implémentations

---

##  Résultats principaux

* **KNN** : sensible au choix de `k` ; bon compromis entre précision et rappel.
* **SVD** : plus puissant, mais risque de sur-apprentissage pour des `k` élevés.
* **SVD avec imputation moyenne** : améliore les résultats dans certains cas.
* **ALS** : stable mais dépend fortement des hyperparamètres (`k`, `λ`, `n_iter`).
* **Popularité** : baseline utile mais limité.

---

##  Améliorations possibles

* Tester sur des datasets plus larges (**MovieLens 1M / 20M**).
* Implémenter d’autres algorithmes (Content-based, Hybrid, Deep Learning).
* Optimisation automatique des hyperparamètres (GridSearch, Bayesian Optimization).

---
