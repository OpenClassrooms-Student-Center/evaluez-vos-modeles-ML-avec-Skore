# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # À vous de jouer ! (Corrigé)
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from skore import evaluate

model = make_pipeline(StandardScaler(), LogisticRegression())
report = evaluate(model, data, target, splitter=0.2)

# %% [markdown]
# 2. Affichez le tableau de métriques du rapport et repérez le rappel
#    (Recall) associé au label `1` (un défaut de paiement).

# %%
report.metrics.summarize()

# %% [markdown]
# Le tableau de métriques confirme l'exactitude annoncée par le manager,
# proche de 90 %. Mais le rappel de la classe `defaut` est extrêmement bas,
# proche de 0.1. Le modèle ne détecte donc presque aucun client qui fait
# réellement défaut. Il classe la quasi-totalité des clients en "pas de
# défaut", ce qui suffit à obtenir une exactitude élevée puisque cette
# classe domine largement le jeu de données.

# %% [markdown]
# 3. Lancez les checks automatisés en mode rapide (`fast_mode=True`) et
#    notez les alertes classées dans la catégorie Issues.

# %%
report.checks.summarize(fast_mode=True)

# %% [markdown]
# La section Issues confirme ce que nous savions déjà. La variable `defaut`
# reste fortement déséquilibrée (SKD004). Elle révèle aussi une information
# nouvelle. Deux variables sont fortement corrélées (SKD008), ce qui peut
# déstabiliser les coefficients d'un modèle linéaire.
#
# La section Tips ajoute deux pistes d'amélioration. Le check SKD006 rappelle
# que les coefficients d'une régression logistique ne sont comparables comme
# mesure d'importance des variables que si celles-ci partagent la même
# échelle. Or nos variables ont, à l'origine, des échelles très différentes,
# par exemple le montant du prêt en milliers de dollars contre un ratio
# d'endettement proche de 1, ce qui invite à la prudence dans
# l'interprétation des coefficients. Le check SKD016 rappelle que
# l'hyperparamètre `C` de la régression logistique est resté à sa valeur par
# défaut, alors qu'il mériterait d'être ajusté.
#
# La section Passed répond à notre énigme. Ni le sous-apprentissage ni le
# surapprentissage ne sont détectés. La complexité du modèle n'est donc pas
# en cause. La feuille de route est plutôt de traiter le déséquilibre des
# classes, d'examiner les variables corrélées, et de régler l'hyperparamètre
# `C` par une recherche d'hyperparamètres.

# %% [markdown]
# 4. Rédigez, dans une cellule Markdown, une réponse de 3 à 5 lignes à votre
#    manager en vous appuyant sur ces résultats pour statuer sur la mise en
#    production.

# %% [markdown]
# ```{note}
# Bonjour,
#
# Bien que notre modèle affiche une exactitude globale de ~90 %, il n'est
# pas prêt pour la mise en production. Les checks de skore écartent une
# piste, la complexité du modèle n'est pas en cause. En revanche, ils
# confirment que la variable `defaut` est fortement déséquilibrée, ce qui
# explique le rappel quasi nul, et signalent des variables corrélées qui
# fragilisent l'interprétation du modèle. Avant tout déploiement, nous
# devons rééquilibrer les données, traiter les variables corrélées, et
# régler l'hyperparamètre `C` par une recherche dédiée.
# ```

# %% [markdown]
# ## En résumé
#
# - L'exactitude est trompeuse sur des données déséquilibrées, le rappel
#   révèle la vraie capacité de détection d'un modèle.
# - `report.checks.summarize` va au-delà de l'évaluation, il diagnostique
#   les causes des mauvaises performances, y compris quand ce n'est pas la
#   complexité du modèle qui pose problème.
# - Les alertes Issues, ici le déséquilibre des classes (SKD004) et des
#   variables corrélées (SKD008), tracent une feuille de route concrète
#   avant tout déploiement.
#
# Vous savez désormais évaluer la santé algorithmique d'un modèle et
# déjouer les pièges des métriques standards. Rendez-vous dans le prochain
# chapitre pour apprendre à intégrer vos propres contraintes métiers grâce à
# des métriques personnalisées.



