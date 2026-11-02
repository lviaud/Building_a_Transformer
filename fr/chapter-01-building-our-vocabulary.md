# Chapitre 1 : Construire notre vocabulaire (le dictionnaire)

> Traduction française non officielle du cours de Rimom Costa.
> [Texte original](https://github.com/rimomcosta/Building_a_Transformer/blob/d8e275a2c41426889e116bfa6d64090805b3bd9a/chapter-01-building-our-vocabulary.md).
> Version source : `d8e275a2c41426889e116bfa6d64090805b3bd9a` ; traduction du 2 novembre 2026.
>
> Les mots anglais utilisés dans les exemples sont conservés pour garder les mêmes tokens et identifiants dans les chapitres suivants. Voir les [notes de traduction](notes-traduction.md).

Les ordinateurs ne parlent qu'en nombres : nous créons donc un grand dictionnaire qui associe les mots à des identifiants. C'est comme attribuer un numéro unique à chaque élève d'une immense école : au lieu de dire « Sarah », on pourrait dire « élève n°567 ».

### Notre vocabulaire

Construisons un vocabulaire de **50 000 tokens**. Pourquoi précisément 50 000 ?

D'abord, un « token » peut être un mot entier, une partie d'un mot ou même une seule lettre. Le nombre 50 000 constitue ici un compromis :

- **Trop peu de tokens, par exemple 10 000 :** beaucoup de mots n'auraient pas leur propre entrée et devraient être découpés en petits morceaux. « Unbelievable » pourrait devenir « un », « believ », « able », ce qui compliquerait le travail du modèle.
- **Trop de tokens, par exemple 500 000 :** nous aurions une entrée pour énormément de mots, mais une table de correspondance IMMENSE, coûteuse à stocker et à exploiter.
- **50 000, un bon compromis pour notre exemple :** cela couvre une grande partie des mots anglais courants et des sous-mots fréquents, sans trop augmenter la taille du vocabulaire.

Des systèmes réels utilisent des tailles comparables : GPT-3 emploie 50 257 tokens et certains systèmes de pointe utilisent des vocabulaires de l'ordre de 100 000 tokens.

Voici à quoi ressemble notre vocabulaire :

```text
ID du token Token
----------- -----
0           <pad>     (spécial : remplissage)
1           <start>   (spécial : début du texte)
2           <end>     (spécial : fin du texte)
...         ...
123         "I"
567         "love"
999         "pizza"
1234        "the"
2001        "cat"
5678        "running"
...         ... (jusqu'à 50,000)
```

Imaginez une immense bibliothèque. On pourrait classer les livres par ordre alphabétique de titre, mais leur recherche prendrait du temps. On attribue plutôt un numéro unique à chaque livre : le livre n°42 occupe toujours le même emplacement. Si quelqu'un cherche « Orgueil et Préjugés », on retrouve son numéro, par exemple n°15 234, puis on y accède directement. Même principe ici : « pizza » correspond toujours au token n°999.

**Le compromis :** un petit vocabulaire demande moins de mémoire, car il comporte moins d'entrées, mais les mots sont découpés en davantage de morceaux. Un grand vocabulaire conserve plus de mots entiers, mais agrandit la table. La matrice d'embeddings comporte une valeur par entrée du vocabulaire et par dimension : 50 000 × 12 288 dimensions représentent environ 614 millions de nombres pour cette seule matrice ! Voilà pourquoi la taille compte.

---

**Navigation dans le cours :** [Précédent : chapitre 0 — La vision d'ensemble (anglais)](https://github.com/lviaud/Building_a_Transformer/blob/main/chapter-00-grand-vision.md) | [Suivant : chapitre 2 — Tokenisation](chapter-02-tokenization.md)
