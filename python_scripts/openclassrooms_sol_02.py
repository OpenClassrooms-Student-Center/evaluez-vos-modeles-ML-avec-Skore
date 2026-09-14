# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # À vous de jouer ! (Corrigé)
#
# Votre métrique métier "Credit Gain" a fait mouche. Le comité de direction
# l'a présentée en réunion, et personne n'a raté l'idée que chaque décision
# du modèle a un prix, et que ce prix se calcule. Mais le contexte
# macro-économique se tend, et la banque révise sa politique de risque en
# conséquence. Un défaut de paiement n'est plus jugé 5 fois plus coûteux
# qu'un bon client refusé, mais 10 fois plus coûteux. La perte de capital
# sur un crédit impayé est désormais considérée comme critique.
#
# On vous demande d'adapter votre métrique à ce nouveau ratio de 1 pour 10,
# et de présenter la nouvelle évaluation à la direction.
#
# Dans cet exercice, vous allez :
# - adapter une fonction de coût métier à une nouvelle contrainte business,
# - construire le `scorer` scikit-learn correspondant avec `make_scorer`,
# - ajouter plusieurs métriques métier à un même rapport `skore` et
#   comparer leur impact.

# %% [markdown]
# ## Point de départ
#
# Dans la vidéo précédente, nous avons repris le modèle et le rapport
# `skore` de l'exercice précédent, puis construit un premier indicateur
# métier, `credit_gain`, qui valorise un bon client accepté à +1, pénalise
# un bon client refusé à -1, et pénalise un défaut manqué à -5. Un client
# accepté qui rembourse ne rapporte ni ne coûte rien dans ce calcul. On
# reproduit cet état ci-dessous avant de répondre à la nouvelle consigne.

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import evaluate

home_loans = pd.read_csv("home_loans.csv")
target = home_loans["defaut"]
data = home_loans.drop(columns="defaut")

scaled_model = make_pipeline(StandardScaler(), LogisticRegression())
report = evaluate(scaled_model, data, target, splitter=0.2)


def credit_gain(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn * 1 + fp * (-1) + fn * (-5) + tp * 0


credit_gain_scorer = make_scorer(credit_gain, greater_is_better=True)
report.metrics.add(credit_gain_scorer, name="Credit Gain")
report.metrics.summarize()

# %% [markdown]
# ## Consigne
#
# À partir de la métrique métier ci-dessus, vous allez :
# 1. Adapter les coûts de la fonction de calcul métier pour refléter la
#    nouvelle politique de risque (ratio 1 pour 10).

# %%
def strict_business_gain(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn * 1 + fp * (-1) + fn * (-10) + tp * 0

# %% [markdown]
# 2. Créer le `scorer` scikit-learn correspondant avec `make_scorer`.

# %%
strict_gain_scorer = make_scorer(strict_business_gain, greater_is_better=True)

# %% [markdown]
# 3. Ajouter cette nouvelle métrique sous le nom "Strict Credit Gain" au
#    rapport `skore`, via `report.metrics.add`.

# %%
report.metrics.add(strict_gain_scorer, name="Strict Credit Gain")

# %% [markdown]
# 4. Résumer les métriques et interpréter le résultat.

# %%
report.metrics.summarize()

# %% [markdown]
# Le tableau affiche désormais deux métriques métier, `credit_gain` et
# `strict_credit_gain`, côte à côte avec les métriques statistiques habituelles.
# Le constat est sans appel : le score passe de 383 à 138 unités financières. La
# matrice de confusion de notre modèle ne change pas, il commet toujours 49 faux
# négatifs sur le jeu de test. Mais chacune de ces erreurs coûte désormais deux
# fois plus cher, ce qui suffit à faire fondre le gain total.
#
# Ce même modèle, jugé acceptable sous l'ancienne politique de risque, devient
# beaucoup moins rentable sous la nouvelle. Rien n'a changé dans ses
# prédictions, seule la grille de coûts métier a bougé, et cela suffit à
# renverser le diagnostic. Cela confirme à la direction que le monitoring du
# modèle reflète instantanément un changement de politique de risque, et que
# réduire ces 49 défauts manqués, par exemple en ajustant le seuil de décision,
# doit devenir une priorité.

# %% [markdown]
# ## En résumé
#
# - Une fonction de coût métier n'est pas figée, elle doit évoluer avec les
#   contraintes de l'entreprise, ici un ratio de risque révisé.
# - `report.metrics.add` accepte plusieurs métriques personnalisées à la
#   suite, ce qui permet de comparer plusieurs politiques de coûts sur un
#   même modèle sans le réentraîner.
# - Un modèle dont les prédictions n'ont pas changé peut devenir beaucoup
#   moins rentable si le contexte métier, lui, a changé.
#
# Vous savez désormais faire évoluer vos métriques métier au rythme des
# décisions de votre entreprise. Rendez-vous dans le prochain chapitre pour
# apprendre à intégrer vos contraintes métiers avec des vérifications
# personnalisées !
