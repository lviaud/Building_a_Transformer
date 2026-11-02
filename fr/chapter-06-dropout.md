# Chapitre 6 : Le dropout (le filet de sécurité de l'entraînement)

> Traduction française non officielle du cours de Rimom Costa.
> [Texte original](https://github.com/rimomcosta/Building_a_Transformer/blob/d8e275a2c41426889e116bfa6d64090805b3bd9a/chapter-06-dropout.md).
> Version source : `d8e275a2c41426889e116bfa6d64090805b3bd9a` ; traduction du 2 novembre 2026.
>
> Les exemples numériques sont conservés, y compris le vecteur illustratif annoncé à la fin du chapitre 5. Les [notes de traduction](notes-traduction.md) précisent la différence entre une probabilité de suppression et une proportion effectivement supprimée.

### Le problème du surapprentissage

**Imaginez la situation suivante.**

Vous révisez un examen de mathématiques. Vous travaillez sur 20 exercices et devenez EXCELLENT sur ces 20 exercices précis. Vous mémorisez :

- « La réponse à l'exercice 5 est 42. »
- « La réponse à l'exercice 12 est 17. »
- « La réponse à l'exercice 18 est 99. »

Mais le jour de l'examen, les exercices sont DIFFÉRENTS ! Ils évaluent les mêmes notions, mais les nombres ont changé. **Vous paniquez** parce que vous avez mémorisé les réponses au lieu d'apprendre les principes mathématiques !

**C'est le surapprentissage, ou overfitting :** mémoriser les exemples d'entraînement sans suffisamment apprendre les régularités qui permettent de généraliser.

**Les réseaux de neurones rencontrent le même problème !** À force de revoir les mêmes données, ils peuvent mémoriser des associations telles que :

- « Quand je vois exactement “The cat slept”, prédire “because”. »
- « Quand je vois exactement “I love”, prédire “pizza”. »

Ils risquent alors d'échouer sur des phrases légèrement différentes. Ils ont mémorisé au lieu de bien généraliser.

### Qu'est-ce que le dropout ?

**Hyperparamètre :** taux de dropout $p = 0.1$, soit 10 %.

**La solution :** pendant **l'entraînement uniquement**, nous désactivons aléatoirement certaines activations en les mettant à zéro. Le réseau ne peut plus dépendre du fait qu'une composante particulière sera toujours présente : cela encourage des représentations plus robustes !

**L'analogie des révisions :**

Au lieu de toujours travailler avec les mêmes amis :

- Lundi : vous révisez avec A, B et C.
- Mardi : B est absent ; vous travaillez avec A et C.
- Mercredi : A est absent ; vous travaillez avec B et C.
- Jeudi : A et B sont absents ; vous vous débrouillez avec C !

**Résultat :** vous apprenez à aborder les notions de plusieurs façons. Vous ne pouvez pas simplement compter sur « B explique toujours l'exercice 5 », puisque B n'est pas toujours là !

**Dans un réseau :** la suppression aléatoire d'activations encourage l'apprentissage par plusieurs voies. Cela favorise la **redondance** et la **robustesse** !

### Le dropout en action

### L'analogie de l'équipe de football

**Imaginez l'entraînement d'une équipe.**

**Sans dropout :**

```text
Entraînement 1 : l'équipe complète joue (11 joueurs)
Entraînement 2 : l'équipe complète joue
Entraînement 3 : l'équipe complète joue
...
Tout le monde s'habitue à : « Si je passe à gauche, Alex est TOUJOURS là ! »
```

**Le jour du match :** Alex est blessé ! 😱

```text
Un joueur pense : « Je passe à gauche comme toujours… Mais Alex n'est pas là ! »
L'équipe s'adapte mal : elle dépendait trop d'Alex !
```

**Avec dropout :**

```text
Entraînement 1 : l'équipe complète joue
Entraînement 2 : Alex et Bob restent de côté (10% retirés au hasard, selon l'analogie)
Entraînement 3 : Charlie et Dana restent de côté
Entraînement 4 : Alex, Emma et Frank restent de côté
...
Tout le monde apprend : « Alex est parfois là, parfois non. Je dois m'adapter ! »
```

Le pourcentage de l'analogie n'est pas un décompte exact des joueurs ; le masque numérique ci-dessous précise le fonctionnement réel.

**Le jour du match :** Alex est blessé ! ✓

```text
Un joueur pense : « Alex est absent, mais je me suis entraîné à cela ! Je passe à Bob. »
L'équipe s'adapte : elle a appris à jouer malgré des absences !
```

**Le dropout applique cette idée aux activations !** En en supprimant aléatoirement certaines à l'entraînement, il réduit la dépendance à une composante toujours active.

### Les mathématiques du dropout

Reprenons le vecteur de sortie illustratif du chapitre précédent : $[0.25, -0.31, 0.42, 0.18, -0.09, 0.33]$.

Avec $p = 0.1$, chaque composante a une probabilité de suppression de 10 %.

**Étape 1 : Générer un masque aléatoire**

Nous tirons un nombre entre 0 et 1 par dimension :

```text
Nombres aléatoires : [0.89, 0.23, 0.95, 0.77, 0.12, 0.88]
```

**Construire le masque :** conserver si le tirage est supérieur à 0.1 ; supprimer s'il est inférieur ou égal à 0.1.

```text
0.89 > 0.1 ? OUI → Conserver (1)
0.23 > 0.1 ? OUI → Conserver (1)
0.95 > 0.1 ? OUI → Conserver (1)
0.77 > 0.1 ? OUI → Conserver (1)
0.12 > 0.1 ? OUI → Conserver (1)
0.88 > 0.1 ? OUI → Conserver (1)

Masque : [1, 1, 1, 1, 1, 1] ← Aucun élément supprimé cette fois !
```

**Avec d'autres tirages :**

```text
Nombres aléatoires : [0.89, 0.03, 0.95, 0.77, 0.12, 0.88]
                         ↑ (0.03 ≤ 0.1, donc SUPPRIMER !)

Masque : [1, 0, 1, 1, 1, 1] ← La dimension 1 est supprimée !
```

**Étape 2 : Appliquer le masque** par une multiplication composante par composante.

$$[0.25, -0.31, 0.42, 0.18, -0.09, 0.33] \times [1, 0, 1, 1, 1, 1] = [0.25, 0, 0.42, 0.18, -0.09, 0.33]$$

La dimension 1 est désactivée, mise à zéro, comme Alex qui ne participe pas à l'entraînement.

**Étape 3 : Multiplier par $1/(1-p)$ pour préserver l'espérance**

**Pourquoi cette mise à l'échelle ?** Sans correction, chaque activation est conservée avec une probabilité de 90 %, ce qui réduit sa valeur moyenne sur les tirages. Nous augmentons donc les activations conservées :

$$[0.25, 0, 0.42, 0.18, -0.09, 0.33] / 0.9 = [0.278, 0, 0.467, 0.2, -0.1, 0.367]$$

Les valeurs restantes augmentent d'environ 11 %, grâce au facteur $1/0.9$. Cela préserve l'espérance de chaque composante, pas nécessairement la somme du vecteur pour chaque masque particulier.

**Pendant l'inférence, après l'entraînement :** le dropout est **désactivé**. Toutes les composantes sont utilisées sans suppression aléatoire ! Dans cette convention, la correction d'échelle est déjà appliquée pendant l'entraînement et n'est pas répétée à l'inférence.

**Où appliquer le dropout ?** Le chapitre retient deux emplacements :

1. **Après la couche d'attention**, avant l'ajout de la connexion résiduelle.
2. **Après le réseau feed-forward**, avant l'ajout de la connexion résiduelle.

La source évoque des taux entre 0.1, soit 10 %, et 0.3, soit 30 %. Ce sont des choix possibles, pas une règle universelle. Le bon réglage dépend de l'architecture, des données et de l'entraînement ; il faut l'évaluer sur des données de validation.

**L'analogie de l'apprentissage en ensemble :** chaque masque active un sous-réseau différent. On peut donc voir le dropout comme l'entraînement de nombreux sous-réseaux qui partagent leurs poids. Cette image aide à comprendre son effet régularisant, mais l'inférence sans dropout n'est pas, en général, la moyenne exacte des prédictions de tous ces sous-réseaux non linéaires.

---

**Navigation dans le cours :** [Précédent : chapitre 5 — Auto-attention multi-têtes](chapter-05-multi-head-self-attention.md) | [Suivant : chapitre 7 — Réseau feed-forward (anglais)](https://github.com/lviaud/Building_a_Transformer/blob/main/chapter-07-feed-forward-network.md)
