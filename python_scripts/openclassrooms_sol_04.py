# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # À vous de jouer ! (Corrigé)
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
from sklearn.model_selection import TunedThresholdClassifierCV

tuned_model = TunedThresholdClassifierCV(
    scaled_model, scoring=strict_gain_scorer, random_state=0
)
tuned_model_report = evaluate(tuned_model, data, target, splitter=5, pos_label=1)
tuned_model_report.metrics.add(strict_gain_scorer, name="Strict Credit Gain")
tuned_model_report.metrics.summarize()

# %% [markdown]
# Le Strict Credit Gain moyen grimpe de 161 à 236, une hausse de 46 %. En
# contrepartie, l'exactitude recule légèrement, de 92,6 % à 91,0 %, pendant que
# le rappel des défauts passe de 24,0 % à 43,6 %, presque le double. L'aire sous
# la courbe ROC, elle, ne bouge pas (0,77 dans les deux cas), signe que seul le
# seuil de décision a changé, pas le classement des clients par probabilité de
# défaut.

# %% [markdown]
# 2. Regroupez `logistic_report` et `tuned_model_report` dans un rapport de
#    comparaison avec `skore.compare`, puis affichez
#    `comparatif.metrics.summarize()`.

# %%
import skore

comparatif = skore.compare(
    {"regression logistique": logistic_report, "tuned model": tuned_model_report}
)
comparatif.metrics.summarize()

# %% [markdown]
# Le tableau comparatif place les deux rapports côte à côte, moyenne et
# écart-type sur 5 plis compris. L'exactitude recule à peine, mais le
# Strict Credit Gain, notre indicateur financier, passe de 161 à 236 en
# moyenne. C'est ce dernier chiffre qui doit trancher, il traduit
# directement l'objectif de la banque.
#
# L'écart-type du Strict Credit Gain augmente lui aussi, passant de 41
# avec le seuil par défaut à 53 avec le seuil ajusté. Le modèle ajusté est
# donc un peu moins stable d'un pli à l'autre, mais ce surcroît de
# variabilité reste modeste au regard du gain moyen : même sur son pli le
# plus défavorable, le modèle ajusté reste compétitif face à la régression
# logistique par défaut.

# %% [markdown]
# 3. Appuyez-vous sur la métrique "Strict Credit Gain" pour recommander
#    formellement, en 3 à 5 lignes, le modèle à conserver pour le comité de
#    direction.

# %% [markdown]
# ```{note}
# Bonjour,
#
# Après comparaison des deux rapports skore en validation croisée, je recommande
# formellement le déploiement de la régression logistique à seuil ajusté
# (`tuned_model_report`), en remplacement de la version actuelle. Ajuster le
# seuil de décision permet de détecter près de deux fois plus de défauts de
# paiement, sans changer de modèle ni retravailler nos variables. Le Strict
# Credit Gain moyen progresse de 161 à 236 sur nos 5 plis de validation, soit un
# gain de 46 %. Cette version ajustée est donc objectivement plus rentable pour
# la banque.
# ```

# %% [markdown]
# ## En résumé
#
# - Une évaluation robuste passe par la validation croisée
#   (`CrossValidationReport`), que skore génère très facilement en donnant
#   un nombre entier au paramètre `splitter` de la fonction `evaluate`.
# - Le seuil de probabilité par défaut (50 %) de scikit-learn est rarement
#   optimal pour des problèmes métiers où les coûts des faux positifs et
#   faux négatifs sont asymétriques.
# - Ajuster le seuil de décision, que ce soit à l'œil avec les outils
#   visuels (`.metrics.roc().plot()` et `.metrics.precision_recall().plot()`)
#   ou automatiquement avec `TunedThresholdClassifierCV`, fait souvent
#   baisser l'exactitude tout en augmentant radicalement le gain financier.
# - La fonction `skore.compare` centralise vos résultats dans un
#   `ComparisonReport` interactif, idéal pour justifier objectivement le
#   choix du modèle final auprès des parties prenantes, même sans changer
#   d'algorithme.
#
# Félicitations, vous êtes désormais capable d'évaluer, d'optimiser et de
# partager sereinement les performances de vos modèles de Machine Learning. Pour
# valider toutes ces compétences, il ne vous reste plus qu'à franchir une
# dernière étape !
