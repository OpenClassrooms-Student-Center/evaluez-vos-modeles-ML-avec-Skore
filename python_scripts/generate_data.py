# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%

# ## Create dataset
#
# Predict clients who default on their loan (see
# https://www.openml.org/search?type=data&status=active&id=46431)

import pandas as pd
from sklearn.datasets import fetch_openml

credit_card = fetch_openml(data_id=46431, as_frame=True, parser="pandas")
X, y = credit_card.data, credit_card.target
y.value_counts()


df = pd.concat([X, y], axis=1)
df = df.dropna(axis=0, how="any")
df = df.dropna(axis=1, how="any")

french_column_names = {
    "bad": "defaut",
    "loan": "montant_pret",
    "mortdue": "solde_hypotheque",
    "value": "valeur_bien",
    "yoj": "anciennete_emploi",
    "derog": "nb_rapports_defavorables",
    "delinq": "nb_credits_impayes",
    "clage": "anciennete_credit",
    "ninq": "nb_demandes_credit",
    "clno": "nb_lignes_credit",
    "debtinc": "ratio_endettement",
    "job_Mgr": "emploi_cadre",
    "job_Office": "emploi_bureau",
    "job_Other": "emploi_autre",
    "job_ProfExe": "emploi_cadre_superieur",
    "job_Sales": "emploi_vente",
    "job_Self": "emploi_independant",
    "job_nan": "emploi_inconnu",
    "reason_DebtCon": "motif_consolidation_dette",
    "reason_HomeImp": "motif_amelioration_habitat",
    "reason_nan": "motif_inconnu",
}

df = df.rename(columns=french_column_names)
df.to_csv("home_loans.csv", index=False)