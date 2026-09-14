# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # À vous de jouer ! (Corrigé)
#
# Votre métrique métier a convaincu la direction, et votre régression
# logistique reste sous surveillance permanente. Mais un tableau de bord ne
# sert à rien si quelqu'un doit encore le lire à l'œil nu avant chaque mise
# en production. L'équipe Risque vous demande d'aller plus loin, en gravant
# une règle financière simple dans le processus de validation lui-même :
# aucun modèle ne doit être recommandé s'il fait perdre de l'argent à la
# banque.
#
# Dans cet exercice, vous allez :
# - compléter une classe métier qui hérite du protocole `Check` de `skore`,
# - y encoder une règle financière non négociable à partir d'une métrique
#   déjà construite,
# - greffer cette vérification à un rapport `skore` via `report.checks.add`,
# - lire le verdict dans `report.checks.summarize()` et conclure sur la mise
#   en production.

# %% [markdown]
# ## Point de départ
#
# On reprend le modèle et le rapport du chapitre précédent, ainsi que
# `strict_business_gain`, la fonction de coût qui reflète la politique de
# risque actuelle de la banque (un défaut manqué coûte 10 fois plus qu'un bon
# client refusé).

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


def strict_business_gain(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn * 1 + fp * (-1) + fn * (-10) + tp * 0


strict_gain_scorer = make_scorer(strict_business_gain, greater_is_better=True)
report.metrics.add(strict_gain_scorer, name="Strict Credit Gain")
report.metrics.summarize()

# %% [markdown]
# ## Consigne
#
# La classe ci-dessous encode le protocole `Check` de `skore`, celui-là même
# utilisé dans la vidéo précédente pour détecter un écart de performance
# entre groupes sensibles. Elle est incomplète, chaque `...` marque une
# information manquante.
#
# 1. Renseignez les métadonnées manquantes, un code unique et un niveau de
#    sévérité, puis terminez la méthode `check_function` pour qu'elle
#    renvoie un message d'alerte explicite quand le Credit Gain est négatif,
#    et `None` sinon.

# %%
from skore import Check, CheckNotApplicable


class CheckCreditGainMinimum(Check):
    code = "RISK002"
    title = "Contrôle de la rentabilité financière (Credit Gain)"
    report_types = ["estimator"]
    docs_url = None
    severity = "issue"

    def check_function(self, report):
        if report.y_test is None:
            raise CheckNotApplicable("Les données de test sont indisponibles.")

        y_test = report.y_test
        y_pred = report.estimator_.predict(report.X_test)
        gain = strict_business_gain(y_test, y_pred)
        if gain < 0:
            return (
                f"Le modèle génère une perte financière de {abs(gain)} unités, "
                "en violation de la politique de risque qui interdit tout "
                "Credit Gain négatif."
            )
        return None

# %% [markdown]
# La sévérité est fixée à `"issue"`, car un modèle qui fait perdre de
# l'argent à la banque n'est pas une simple piste d'amélioration, c'est un
# motif de rejet. Le code `"RISK002"` poursuit la numérotation des règles de
# l'équipe Risque, entamée avec `"RISK001"` dans la vidéo précédente.

# %% [markdown]
# 2. Instanciez cette vérification et ajoutez-la au rapport avec
#    `report.checks.add`.

# %%
report.checks.add([CheckCreditGainMinimum()])

# %% [markdown]
# 3. Affichez `report.checks.summarize()` et repérez dans quelle section,
#    Issues, Tips ou Passed, apparaît votre nouveau check.

# %%
report.checks.summarize(fast_mode=True)

# %% [markdown]
# 4. Concluez, en 2 ou 3 phrases, sur la mise en production de ce modèle.

# %% [markdown]
# `RISK002` apparaît dans la section Passed, pas dans les Issues. Avec un
# Strict Credit Gain de 138, largement positif, notre garde-fou financier ne
# trouve rien à redire à ce modèle précis. Les problèmes déjà identifiés au
# premier chapitre, eux, n'ont pas bougé, la variable `defaut` reste
# déséquilibrée (SKD004) et deux variables restent corrélées (SKD008).
#
# Ce modèle n'est donc pas rejeté pour cause de perte financière, mais il
# reste fragile sur le plan statistique, et rien ne garantit qu'une future
# itération continuera de passer ce contrôle. En particulier, ajuster le
# seuil de décision, ce que nous ferons au prochain chapitre, déplace
# l'équilibre entre faux positifs et faux négatifs, et donc le Credit Gain.
# `RISK002` est désormais un filet de sécurité permanent, prêt à sonner
# l'alarme dès qu'une future version basculerait dans le rouge.

# %% [markdown]
# ## En résumé
#
# - Le protocole `Check` de `skore` encapsule une règle métier, ici
#   financière, dans une classe Python standardisée, au même titre qu'une
#   règle statistique native.
# - Chaque vérification personnalisée requiert des métadonnées obligatoires,
#   un code unique, un titre, et un niveau de sévérité, `"issue"` pour un
#   motif de rejet ou `"tip"` pour un simple conseil.
# - `report.checks.add` greffe durablement ces règles à un rapport, elles
#   sont réévaluées à chaque nouveau diagnostic, aux côtés des checks
#   natifs de `skore`.
# - Qu'un check personnalisé passe ou échoue, il documente une contrainte
#   métier de façon reproductible, ce qui vaut mieux qu'une relecture
#   manuelle avant chaque mise en production.
#
# Vous disposez maintenant d'un filet de sécurité financier automatisé.
# Rendez-vous dans le chapitre suivant pour apprendre à riposter : vous
# allez entraîner un modèle plus performant, ajuster son seuil de décision,
# et comparer vos rapports pour repousser encore la rentabilité de votre
# modèle.
