# Chapitre 2 : La tokenisation (découper le texte en morceaux)

> Traduction française non officielle du cours de Rimom Costa.
> [Texte original](https://github.com/rimomcosta/Building_a_Transformer/blob/d8e275a2c41426889e116bfa6d64090805b3bd9a/chapter-02-tokenization.md).
> Version source : `d8e275a2c41426889e116bfa6d64090805b3bd9a` ; traduction du 2 novembre 2026.
>
> Les exemples anglais et les identifiants de tokens sont conservés pour assurer la continuité du cours. L'explication de BPE est volontairement simplifiée ; voir les [notes de traduction](notes-traduction.md).

### Comprendre le problème

Nous disposons d'un vocabulaire de 50 000 tokens. Commençons par un cas simple.

Texte d'entrée : `"I love pizza"` (« J'aime la pizza »).

**Étape 1 :** chercher chaque mot dans le vocabulaire.

- « I » → 123 ✓ Trouvé !
- « love » → 567 ✓ Trouvé !
- « pizza » → 999 ✓ Trouvé !

Résultat : `[123, 567, 999]`.

Facile ! Mais que se passe-t-il avec un mot absent du dictionnaire ?

**Le problème :** prenons « cryptocurrency ». Ce mot était peu courant lorsque nous avons construit notre vocabulaire ; supposons donc qu'il n'y figure pas. Nous ne pouvons pas l'ignorer : il nous faut représenter chaque mot d'une manière ou d'une autre.

**Mauvaise solution 1 :** lui attribuer un token spécial « inconnu ».

- Problème : tous les mots inconnus deviennent identiques pour le modèle ! « cryptocurrency » et « philosophy » auraient la même représentation.

**Mauvaise solution 2 :** ajouter tous les mots possibles au vocabulaire.

- Problème : il existe énormément de mots en anglais, et de nouveaux apparaissent chaque jour, comme « selfie », « podcast » ou « ChatGPT ». Le vocabulaire devrait croître sans limite !

**La bonne solution : décomposer les mots inconnus en morceaux connus !**

### La solution Byte-Pair Encoding (BPE)

BPE ressemble à un assembleur de puzzle astucieux. Lorsqu'il rencontre « cryptocurrency », on peut se représenter son fonctionnement ainsi :

1. Vérifier : « cryptocurrency » est-il dans le vocabulaire ? Non.
2. Essayer : peut-on le décomposer en deux morceaux connus, « crypto » + « currency » ? Oui ! Tous deux sont assez courants pour figurer dans notre vocabulaire.
3. Résultat : « cryptocurrency » → [token de « crypto », token de « currency »].

Si un morceau ne figure toujours pas dans le vocabulaire, on le décompose davantage :

- « cryptocurrency » → [« crypto », « currency »] → [« crypt », « o », « cur », « ren », « cy »].

**Simplification importante :** ce tutoriel utilise une **représentation mentale fondée sur les caractères**, plus facile à suivre sur papier. Dans de nombreux systèmes de type GPT, les tokeniseurs fonctionnent en réalité au **niveau des octets**, et non des lettres. L'idée générale reste la même : si le mot entier n'est pas disponible, on utilise des morceaux plus petits présents dans le vocabulaire.

Dans notre exemple, le cas extrême consiste donc à « descendre jusqu'aux caractères ». Pour de nombreux tokeniseurs de production, il s'agit plutôt de « descendre jusqu'aux octets ».

Imaginez que vous expliquiez quelque chose avec des mots déjà connus de votre interlocuteur. Si un enfant ne connaît pas « automobile », vous pourriez expliquer « auto » et « mobile », c'est-à-dire « qui se déplace par soi-même ». S'il ne connaît pas « auto », vous décomposez encore l'explication en notions plus simples.

### Comment BPE construit le vocabulaire (l'entraînement)

Avant d'utiliser BPE, nous devons créer notre vocabulaire de 50 000 tokens. Voici le principe.

**Étape 1 : Partir des éléments de base**

Dans cette version simplifiée, nous commençons par les caractères individuels : a, b, c, …, z, A, B, C, …, Z, ainsi que l'espace et la ponctuation.

Cela représente environ 100 tokens de base dans notre exemple.

**Étape 2 : Trouver la paire la plus fréquente**

Nous examinons des millions de textes d'exemple. Quels sont les deux caractères qui se suivent le plus souvent ?

Peut-être que « t » suivi de « h » apparaît 150 000 fois dans le corpus.

**Étape 3 : Fusionner la paire**

Nous créons un nouveau token « th » et l'ajoutons au vocabulaire, qui contient désormais 101 tokens.

Nous remplaçons les occurrences de « t » + « h » par le token unique « th ».

**Étape 4 : Recommencer**

Nous cherchons la paire suivante la plus fréquente. Peut-être « e » + « r », avec 120 000 occurrences.

Nous créons le token « er ». Le vocabulaire contient maintenant 102 tokens.

**Étape 5 : Continuer**

Après de nombreuses fusions, nous obtenons des sous-mots fréquents :

- « ing », issu de « i » + « n » + « g » fusionnés en plusieurs étapes.
- « tion », une terminaison courante.
- « the », un mot très fréquent.
- Puis des mots entiers courants : « pizza », « computer », « love ».

**Étape 6 : S'arrêter à 50 000**

Dans notre exemple, avec environ 100 éléments de base au départ, il faudrait **approximativement 49 900 fusions** pour atteindre près de 50 000 tokens.

**En production :** les tokeniseurs peuvent disposer de tokens réservés supplémentaires, d'alphabets ou d'ensembles d'octets initiaux différents, et d'autres particularités qui compliquent le décompte exact. L'important n'est PAS le calcul précis, mais le principe : fusionner les paires fréquentes jusqu'à la taille de vocabulaire souhaitée.

**Le résultat :** les mots fréquents peuvent devenir des tokens uniques, ce qui est efficace, tandis que les mots rares sont divisés en morceaux, ce qui apporte de la souplesse !

Exemple de hiérarchie dans le vocabulaire final :

```text
Caractères : "a", "b", "c"
Sous-mots :  "th", "ing", "er"
Mots :       "the", "pizza", "love"
```

**Pourquoi cela fonctionne :** la langue présente des régularités. La paire « th » est fréquente en anglais et mérite donc son propre token dans notre exemple. À l'inverse, « qz » est très rare : inutile de lui consacrer une entrée. Le même principe s'applique en production, même lorsque les plus petites unités sont des octets.

**Déterminisme et réversibilité :** une fois le vocabulaire construit et les règles fixées, le même texte produit les mêmes tokens dans le cadre décrit ici. On peut aussi reconvertir les tokens en texte, contrairement à une compression avec perte. Les conditions de cette réversibilité sont précisées dans les notes de traduction.

---

**Navigation dans le cours :** [Précédent : chapitre 1 — Construire notre vocabulaire](chapter-01-building-our-vocabulary.md) | [Suivant : chapitre 3 — Embeddings](chapter-03-embeddings.md)
