# Documentation du package satellite-rl

## Sommaire

- [Documentation du package satellite-rl](#documentation-du-package-satellite-rl)
  - [Sommaire](#sommaire)
  - [Description des modules](#description-des-modules)
  - [Modélisation](#modélisation)
  - [Algorithmes utilisés](#algorithmes-utilisés)

## Description des modules

Le package `satellite-rl` s'articule autour de 3 sous-packages et de plusieurs sous-modules :
- `comparison` permet de comparer les performances des algorithmes mis en oeuvre
- `reinforcment_learning` implémente les algorithmes d'apprentissage par renforcement
- `report` génère des rapports html et json sur les performances des algorithmes
- `config.py` contient les valeurs de constantes configurables par des variables d'environnement
- `generate_instances.py` est un script qui génère des instances aléatoires du problème
- `__main__.py` contient une interface en ligne de commandes pour exécuter les algorithmes

## Modélisation

Voir le document [modelisation.md](modelisation.md)

## Algorithmes utilisés

Voir le document [algorithmes.md](algorithmes.md)


