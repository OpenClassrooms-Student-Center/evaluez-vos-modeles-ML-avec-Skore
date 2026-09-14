# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # À vous de jouer !
#
# Dans la vidéo précédente, vous avez comparé votre régression logistique à
# un Gradient Boosting grâce à une validation croisée et `skore.compare`. Au
# seuil de décision par défaut, la régression logistique l'a emporté sur le
# Credit Gain. Plutôt que de complexifier le modèle, l'équipe Data
# Science a donc poussé l'optimisation plus loin en ajustant directement le
# seuil de décision de la régression logistique.
#
# Le comité de direction se réunit cet après-midi. Le directeur des risques
# et la directrice de la conformité vous ont demandé un verdict final
# clair : doivent-ils déployer cette version ajustée à la place du modèle
# actuel ? Vous devez comparer les deux rapports et vérifier si
# l'ajustement du seuil améliore réellement les performances métiers de la
# banque.
#
# Dans cet exercice, vous allez :
# - construire un rapport `skore` robuste grâce à la validation croisée, en
#   donnant un entier au paramètre `splitter`,
# - ajuster automatiquement le seuil de décision de votre régression
#   logistique avec `TunedThresholdClassifierCV`, pour maximiser une
#   métrique métier plutôt que l'exactitude,
# - regrouper deux rapports `skore` dans un `ComparisonReport` grâce à
#   `skore.compare`,
# - traduire ce comparatif en une recommandation chiffrée pour le comité de
#   direction.

# %% [markdown]
# ## Point de départ
#
# On reprend le modèle du chapitre précédent, la régression logistique,
# ainsi que `strict_business_gain`, la fonction de coût qui reflète la
# politique de risque actuelle de la banque (un défaut manqué coûte 10 fois
# plus qu'un bon client refusé). Pour gagner en robustesse avant cette
# décision finale, on remplace le simple partage train/test par une
# validation croisée à 5 plis. Il suffit de donner un entier au paramètre
# `splitter` de `evaluate` pour obtenir un `CrossValidationReport`.

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


def strict_business_gain(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn * 1 + fp * (-1) + fn * (-10) + tp * 0


strict_gain_scorer = make_scorer(strict_business_gain, greater_is_better=True)

scaled_model = make_pipeline(StandardScaler(), LogisticRegression())
logistic_report = evaluate(scaled_model, data, target, splitter=5, pos_label=1)
logistic_report.metrics.add(strict_gain_scorer, name="Strict Credit Gain")
logistic_report.metrics.summarize()

# %% [markdown]
# Le rapport affiche désormais une moyenne et un écart-type par métrique
# sur les 5 plis, un Strict Credit Gain de 161 en moyenne, avec un
# écart-type de 41. La performance financière varie donc sensiblement d'un
# pli à l'autre, une variabilité qu'une simple séparation train/test
# aurait pu masquer.

# %% [markdown]
# ## Consigne
#
# 1. Ajustez automatiquement le seuil de décision de `scaled_model` avec
#    `TunedThresholdClassifierCV`, en lui passant `strict_gain_scorer`
#    comme `scoring`. Évaluez ce modèle ajusté avec `evaluate` (les mêmes
#    paramètres que `logistic_report`, `splitter=5, pos_label=1`) dans un
#    rapport `tuned_model_report`, ajoutez-lui la métrique "Strict Credit
#    Gain", et affichez son résumé.

# %%
# Écrivez votre code ici.

# %% [markdown]
# 2. Regroupez `logistic_report` et `tuned_model_report` dans un rapport de
#    comparaison avec `skore.compare`, puis affichez
#    `comparatif.metrics.summarize()`.

# %%
# Écrivez votre code ici.

# %% [markdown]
# 3. Appuyez-vous sur la métrique "Strict Credit Gain" pour recommander
#    formellement, en 3 à 5 lignes, le modèle à conserver pour le comité de
#    direction.

# %% [markdown]
# _Rédigez votre recommandation ici._
