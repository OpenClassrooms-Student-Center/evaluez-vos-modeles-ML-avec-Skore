# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # À vous de jouer !
#
# Votre manager vient de recevoir votre premier tableau de bord. Il est ravi,
# votre régression logistique affiche une exactitude proche de 90 %. Il vous
# écrit dans la foulée pour savoir si ce modèle peut partir en production dès
# lundi, afin de détecter automatiquement les défauts de paiement des
# nouveaux clients.
#
# Vous venez de voir qu'une exactitude flatteuse peut cacher un modèle qui
# n'a rien appris. Avant de répondre à votre manager, vérifiez si c'est le
# cas ici, et si oui, pourquoi.
#
# Dans cet exercice, vous allez :
# - construire un `EstimatorReport` pour dépasser la seule lecture de
#   l'exactitude,
# - repérer, grâce au rappel (Recall), si le modèle détecte réellement les
#   défauts de paiement,
# - lancer les checks automatisés de `skore` pour en identifier la cause,
# - traduire ce diagnostic en une recommandation claire pour votre manager.

# %%
import pandas as pd

home_loans = pd.read_csv("home_loans.csv")
target = home_loans["defaut"]
data = home_loans.drop(columns="defaut")
home_loans

# %% [markdown]
# ## Consigne
#
# 1. Entraînez une régression logistique à échelle à l’aide d’une pipeline
#    scikit-learn (`make_pipeline(StandardScaler(), LogisticRegression())`) et
#    évaluez-la avec `skore.evaluate`, en réservant 20 % des données pour le
#    test grâce au paramètre `splitter`.

# %%
# Écrivez votre code ici.

# %% [markdown]
# 2. Affichez le tableau de métriques du rapport et repérez le rappel
#    (Recall) associé au label `1` (un défaut de paiement).

# %%
# Écrivez votre code ici.

# %% [markdown]
# 3. Lancez les checks automatisés en mode rapide (`fast_mode=True`) et
#    notez les alertes classées dans la catégorie Issues.

# %%
# Écrivez votre code ici.

# %% [markdown]
# 4. Rédigez, dans une cellule Markdown, une réponse de 3 à 5 lignes à votre
#    manager en vous appuyant sur ces résultats pour statuer sur la mise en
#    production.

# %% [markdown]
# _Rédigez votre réponse au manager ici._
