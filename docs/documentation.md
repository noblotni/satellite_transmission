# Documentation du package satellite-rl

## Sommaire

- [Documentation du package satellite-rl](#documentation-du-package-satellite-rl)
  - [Sommaire](#sommaire)
  - [Description des modules](#description-des-modules)
  - [Modélisation](#modélisation)
  - [Algorithmes utilisés](#algorithmes-utilisés)
  - [Références](#références)

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

## Références

Voici les sources utilisées pour écrire cette documentation : 

- [Understanding Actor-Critic Methods and A2C, Chris Yoon](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
  
- [Actor-critic using deep-RL: continuous mountain car in TensorFlow, Andy Steinbach](https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c)
  
- [Proximal Policy Optimization Explained, Wouter van Heeswijk, PhD
Wouter van Heeswijk, PhD](https://towardsdatascience.com/proximal-policy-optimization-ppo-explained-abed1952457b)
  
- [Simple PPO implementation, Alvaro Durán Tovar](https://medium.com/deeplearningmadeeasy/simple-ppo-implementation-e398ca0f2e7c)
 
- [Proximal Policy Optimization Algorithms, John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov](https://arxiv.org/abs/1707.06347)