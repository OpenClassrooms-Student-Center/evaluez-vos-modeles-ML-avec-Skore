# evaluez-vos-modeles-ML-avec-Skore

Support de cours du cours **skore**, co-développé par Probabl et OpenClassrooms.

Ce repo contient :
- les fichiers sources Python, dont l'étudiant n'a pas besoin ;
- le jeu de données `home_loans.csv` ;
- les notebooks Jupyter utilisés tout au long du cours.

Les fichiers se terminant par `ex_*` correspondent aux notebooks « À vous de
jouer ! ». Les fichiers se terminant par `sol_*` correspondent aux notebooks 
« Corrigés » portant la même numérotation.

## Workflow de modification

**La source de vérité se trouve dans `python_scripts/`.** Toute modification de
contenu (explications, code, exercices, etc.) doit être effectuée dans les
fichiers `.py` du dossier `python_scripts`. Les notebooks `.ipynb`
correspondants sont générés automatiquement à partir de ces fichiers grâce à
[Jupytext](https://jupytext.readthedocs.io/) et ne doivent **pas** être modifiés
à la main. Toute modification manuelle d'un notebook serait perdue lors de sa
prochaine régénération.

## Génération des notebooks avec Jupytext

Les notebooks sont générés à partir des fichiers Python à l'aide de Jupytext.

### Prérequis

- [Jupytext](https://jupytext.readthedocs.io/) (testé avec la version `1.17.0`)

Installez-le avec :

```bash
pip install jupytext
```

### Générer un seul notebook

Depuis la racine du dépôt, exécutez par exemple :

```bash
jupytext --to notebook python_scripts/openclassrooms_create_dataset.py && \
mv python_scripts/openclassrooms_create_dataset.ipynb notebooks/
```

Cette commande crée (ou met à jour) le fichier `.ipynb` correspondant et le
déplace dans le dossier `notebooks`.

Répétez cette commande pour chaque fichier Python à convertir, en adaptant le
nom de fichier.

### Générer tous les notebooks en une seule fois

Pour régénérer tous les notebooks à partir de l'ensemble des fichiers de
`python_scripts/` :

```bash
jupytext --to notebook python_scripts/*.py && mv python_scripts/*.ipynb notebooks/
```
