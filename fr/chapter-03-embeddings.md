# Chapitre 3 : Les embeddings (donner du sens aux nombres)

> Traduction française non officielle du cours de Rimom Costa.
> [Texte original](https://github.com/rimomcosta/Building_a_Transformer/blob/d8e275a2c41426889e116bfa6d64090805b3bd9a/chapter-03-embeddings.md).
> Version source : `d8e275a2c41426889e116bfa6d64090805b3bd9a` ; traduction du 2 novembre 2026.
>
> Les mots anglais, les identifiants de tokens et les valeurs numériques sont conservés pour assurer la continuité avec les chapitres précédents et le chapitre 4. Les propriétés attribuées aux dimensions sont des analogies pédagogiques ; voir les [notes de traduction](notes-traduction.md).

### Le problème : un seul nombre ne suffit pas

Notre phrase est maintenant une suite d'identifiants de tokens : `[123, 567, 999]`, qui représente « I love pizza ».

Mais il reste un **problème majeur** ! Chaque token n'est qu'un nombre, un identifiant comparable à un numéro d'élève :

- Token 123 = « I ».
- Token 567 = « love ».
- Token 999 = « pizza ».

Ces nombres indiquent DE QUEL mot il s'agit, mais ne disent RIEN de son SENS. C'est comme une bibliothèque dont les livres portent les numéros 1, 2, 3 : ces numéros ne disent pas s'ils parlent de cuisine, de science ou d'histoire !

**La question fondamentale : comment représenter le SENS d'un mot avec des nombres qu'un ordinateur peut traiter ?**

### Décrire les choses par plusieurs propriétés

Comment décririez-vous une pizza à quelqu'un qui n'en a jamais vu ?

- « C'est de la **nourriture** », ni une personne ni un outil.
- « Elle est **circulaire** », elle a une forme.
- « Elle est **savoureuse** », une qualité positive.
- « Elle est **italienne** », une origine culturelle.
- « Le mot est un **nom** », une catégorie grammaticale.
- « Elle est **solide** », ni liquide ni abstraite.

Il faut PLUSIEURS informations pour décrire complètement une pizza. Une seule ne suffit pas !

**Autre exemple : vous décrire vous-même.**

- Taille : 5.7 pieds, dans l'exemple original.
- Âge : 25 ans.
- Niveau de bonheur : 8/10.
- Niveau d'énergie : 6/10.
- Créativité : 7/10.
- Aptitudes sportives : 4/10.

Chacune est une **propriété**, ou **caractéristique**, qui vous décrit. Ensemble, elles donnent une image plus complète !

### D'un nombre à plusieurs : les dimensions

C'est exactement ce qu'il nous faut pour les mots ! Au lieu de représenter chaque mot par UN nombre, son identifiant de token, nous utiliserons PLUSIEURS nombres pour rendre compte de différents aspects de sa signification.

**L'idée essentielle :** chaque token dispose de plusieurs « emplacements » où stocker ces informations. Chaque emplacement est une **dimension**.

**Un exemple simple à trois dimensions :**

Imaginons qu'après l'entraînement, trois dimensions représentent :

- Dimension 1 : « À quel point ce mot est-il positif ? », entre -1 et +1.
- Dimension 2 : « Est-il lié à la nourriture ? », entre -1 et +1.
- Dimension 3 : « Est-ce un nom ou un verbe ? », entre -1 et +1.

Nos mots pourraient alors ressembler à ceci :

```text
"pizza":  [0.2,   0.9,   0.8]
          ↑      ↑      ↑
       un peu  aliment  nom
       positif

"love":   [0.9,  -0.3,  -0.7]
          ↑      ↑       ↑
       très    pas un   verbe
       positif aliment

"hate":   [-0.8, -0.2,  -0.6]
           ↑      ↑       ↑
        négatif pas un   verbe
                aliment
```

Voyez-vous les régularités ?

- « love » (0.9) et « hate » (-0.8) sont OPPOSÉS dans la dimension 1 : leurs sentiments sont opposés.
- « pizza » (0.9) a une valeur ÉLEVÉE dans la dimension 2 : c'est bien de la nourriture !
- « love » et « hate » sont tous deux négatifs dans la dimension 3 : ce sont des verbes !

**L'intérêt :** ces nombres permettent de représenter à la fois la ressemblance entre « love » et « hate », deux verbes exprimant des émotions, et leur opposition entre sentiment positif et négatif !

### Davantage de dimensions pour enrichir la représentation

Dans ce tutoriel, chaque mot aura **6 dimensions**. La source évoque **12 288 dimensions** pour de grands modèles ; il s'agit notamment de la taille retenue pour GPT-3 à 175 milliards de paramètres, et non d'une taille universelle des modèles de conversation.

**Pourquoi autant ?** La langue est riche et nuancée ! Un mot peut avoir de nombreuses propriétés :

- Est-il positif ou négatif ? Le sentiment.
- Est-il concret ou abstrait ? Le caractère concret.
- Est-ce un nom, un verbe ou un adjectif ? La grammaire.
- Est-il soutenu ou familier ? Le registre.
- Parle-t-il de personnes, d'objets ou d'idées ? La catégorie.
- Est-il courant ou rare ? La fréquence.
- Est-il moderne ou vieilli ? L'époque.
- Est-il lié à la vue, à l'ouïe, au goût, au toucher ou à l'odorat ? Les perceptions sensorielles.
- Est-il technique ou quotidien ? Le domaine.
- Est-il littéral ou métaphorique ? L'usage.

Dans cette analogie, six dimensions ne décriraient que quelques grandes propriétés, tandis que 12 288 permettraient de représenter des nuances bien plus fines.

**Précision essentielle :** nous ne dirons PAS explicitement au modèle « dimension 1 = sentiment, dimension 2 = nourriture ». Il **apprendra** sa représentation pendant l'entraînement, en découvrant les informations utiles pour prédire le token suivant !

Après l'entraînement :

- Nous ne pouvons PAS regarder la dimension 3 et conclure « c'est la dimension nom/verbe ».
- Les dimensions n'ont PAS d'étiquettes.
- Le modèle les utilise pourtant pour représenter des informations utiles.
- Les chercheurs peuvent analyser les régularités et tenter de les interpréter, mais ce n'est pas simple.

C'est un peu comme le cerveau : vous savez que « chien » et « chiot » sont proches, sans pouvoir désigner un neurone précis et dire « le neurone n°4829 stocke la notion de chien ». Les connaissances sont distribuées entre de nombreux neurones qui fonctionnent ensemble. Même idée ici !

### Petit rappel mathématique : vecteurs et matrices

Avant de poursuivre, présentons les outils mathématiques nécessaires.

**Vecteur :** une liste de nombres représentant plusieurs propriétés.

- Vecteur ligne : $[1, 2, 3]$, par exemple trois propriétés d'un objet.
- Vecteur colonne :

$$
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

**Pourquoi le nom « vecteur » ?** En mathématiques et en physique, un vecteur possède plusieurs composantes ; la vitesse vectorielle représente par exemple une amplitude et une direction. Nos vecteurs de mots possèdent plusieurs composantes de représentation !

**Exemple :** le mot « pizza » avec six dimensions.

$$\text{pizza} = [0.90, -0.10, -0.20, -0.80, 0.50, 0.75]$$

Ce vecteur contient six nombres qui, après apprentissage, contribuent ensemble à représenter « pizza ».

**Matrice :** un tableau de vecteurs empilés.

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
$$

Cette matrice est de taille 2×3 : deux lignes, par exemple deux mots, et trois colonnes, par exemple trois dimensions par mot.

**Lire ses dimensions :**

- Forme : 2 lignes × 3 colonnes. On écrit d'abord les lignes, puis les colonnes.
- Ligne 1 : $[1, 2, 3]$, les propriétés du mot n°1.
- Ligne 2 : $[4, 5, 6]$, les propriétés du mot n°2.

**Multiplication matricielle :** elle permet de transformer les données ! Nous l'utiliserons beaucoup dans le Transformer.

Multiplions la matrice $\mathbf{A}$ de taille 2×3 par la matrice $\mathbf{B}$ de taille 3×2 :

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix}
7 & 8 \\
9 & 10 \\
11 & 12
\end{bmatrix}
$$

Le résultat est une matrice 2×2, donnée par les dimensions extérieures :

$$
\mathbf{C} = \mathbf{A} \times \mathbf{B} = \begin{bmatrix}
c_{11} & c_{12} \\
c_{21} & c_{22}
\end{bmatrix}
$$

**Calcul de chaque élément :**

$c_{11}$, ligne 1 de A et colonne 1 de B :

$$c_{11} = (1)(7) + (2)(9) + (3)(11) = 7 + 18 + 33 = 58$$

$c_{12}$, ligne 1 de A et colonne 2 de B :

$$c_{12} = (1)(8) + (2)(10) + (3)(12) = 8 + 20 + 36 = 64$$

$c_{21}$, ligne 2 de A et colonne 1 de B :

$$c_{21} = (4)(7) + (5)(9) + (6)(11) = 28 + 45 + 66 = 139$$

$c_{22}$, ligne 2 de A et colonne 2 de B :

$$c_{22} = (4)(8) + (5)(10) + (6)(12) = 32 + 50 + 72 = 154$$

**Résultat final :**

$$
\mathbf{C} = \begin{bmatrix}
58 & 64 \\
139 & 154
\end{bmatrix}
$$

**La règle :** pour obtenir l'élément à la ligne i et à la colonne j, prendre la ligne i de la première matrice et la colonne j de la seconde, multiplier les éléments correspondants, puis additionner les produits.

**Règle importante :** pour calculer A×B, le nombre de colonnes de A doit être égal au nombre de lignes de B !

- (2×3) × (3×2) ✓ Possible : les dimensions intérieures correspondent, 3 = 3.
- (2×3) × (2×4) ✗ Impossible : 3 ≠ 2.

Utilisons maintenant ces outils pour construire notre système d'embeddings !

### La table d'embeddings : un grand dictionnaire de correspondance

Nous avons 50 000 tokens à représenter chacun par six dimensions.

**Vocabulaire essentiel :**

- **Identifiant de token :** le nombre qui identifie un token, par exemple 999 pour « pizza ».
- **Embedding :** le vecteur de six nombres représentant ce token.
- **Dimension :** une composante individuelle du vecteur ; nous en avons six.
- **$d_{\text{model}}$ :** la notation pour « le nombre de dimensions utilisées », ici six.

**Pourquoi « $d_{\text{model}}$ » ?** Cela signifie « dimension du modèle » : la taille des vecteurs qui circulent dans le Transformer. Chaque position possède une représentation de $d_{\text{model}}$ nombres dans les états de cette taille.

**Notre choix :** $d_{\text{model}} = 6$, une petite taille pour calculer à la main !

**Le choix cité pour GPT-3 :** $d_{\text{model}} = 12,288$, une représentation beaucoup plus grande.

### Créer la table de correspondance des embeddings

Nous créons un grand tableau, une matrice, de dimensions `[vocab_size × d_model]` = `[50,000 × 6]`.

Comme dans un tableur :

- **50 000 lignes :** une ligne par token du vocabulaire.
- **6 colonnes :** une colonne par dimension.

```text
Matrice d'embeddings E (seuls 3 mots sur 50,000 sont détaillés) :

ID token  Mot       dim₀    dim₁    dim₂    dim₃    dim₄    dim₅
--------  ----      ----    ----    ----    ----    ----    ----
   123    "I"       0.01   -0.20    0.30    0.15   -0.05    0.11
   567    "love"   -0.40    0.60    0.00    0.25    0.90   -0.30
   999    "pizza"   0.90   -0.10   -0.20   -0.80    0.50    0.75
   ...    ...       ...     ...     ...     ...     ...     ...
 49999    (dernier) 0.23    0.45   -0.67    0.12    0.89   -0.34
```

**Comment lire ce tableau :**

- Ligne 123, token « I » : `[0.01, -0.20, 0.30, 0.15, -0.05, 0.11]`.
- Ligne 999, token « pizza » : `[0.90, -0.10, -0.20, -0.80, 0.50, 0.75]`.

**D'où viennent ces nombres au départ ?** Ils sont **complètement aléatoires** ! Ce sont de petits nombres proches de zéro ; la stratégie d'initialisation sera expliquée plus loin. Le modèle n'a initialement AUCUNE connaissance du sens des mots.

**Comment acquièrent-ils du sens ?** Grâce à l'entraînement ! Le modèle les ajuste au fil des exemples, en apprenant progressivement une représentation utile. Dans l'intuition proposée par la source :

- Les mots similaires finissent avec des vecteurs proches dans l'espace à six dimensions.
- Les mots opposés seraient éloignés ou orientés dans des directions opposées ; ce point est à nuancer dans les notes de traduction.
- Les dimensions s'organisent pour représenter des régularités utiles.

### Retrouver un embedding : de l'identifiant au vecteur

Pour le token 123, « I », nous prenons simplement la ligne 123 du tableau :

$$\text{embedding}_{123} = \mathbf{E}[123, :] = [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]$$

**Que signifie cette notation ?**

- $\mathbf{E}$ : la matrice d'embeddings, notre grand tableau.
- $[123, :]$ : « ligne 123, toutes les colonnes ».
- Résultat : un vecteur à six dimensions représentant « I ».

**Pour la phrase entière « I love pizza » :**

Identifiants : `[123, 567, 999]`.

Nous récupérons chaque ligne :

- Token 123 → `[0.01, -0.20, 0.30, 0.15, -0.05, 0.11]`.
- Token 567 → `[-0.40, 0.60, 0.00, 0.25, 0.90, -0.30]`.
- Token 999 → `[0.90, -0.10, -0.20, -0.80, 0.50, 0.75]`.

Puis nous les empilons en matrice :

$$
\mathbf{X} = \begin{bmatrix}
0.01 & -0.20 & 0.30 & 0.15 & -0.05 & 0.11 \\
-0.40 & 0.60 & 0.00 & 0.25 & 0.90 & -0.30 \\
0.90 & -0.10 & -0.20 & -0.80 & 0.50 & 0.75
\end{bmatrix}
$$

**Forme :** $3 \times 6$, trois mots avec six dimensions chacun.

**Lecture de la matrice :**

- Ligne 1 : embedding de « I ».
- Ligne 2 : embedding de « love ».
- Ligne 3 : embedding de « pizza ».
- Colonne 1, dim₀ : première composante de la représentation pour les trois mots.
- Colonne 2, dim₁ : deuxième composante.
- Et ainsi de suite.

C'est cette matrice qui entre dans la suite du Transformer ! Chaque ligne représente un token par un vecteur à six dimensions.

### Ce que chaque dimension apprend (après l'entraînement)

Nous ne disons **pas** au modèle ce que chaque dimension doit signifier. Mais après l'entraînement sur de très nombreux exemples, des régularités apparaissent ! Voici ce qui pourrait se produire dans une représentation imaginaire ; c'est le modèle qui construit la représentation réelle.

**⚠️ PRÉCISION IMPORTANTE — À lire d'abord !**

Les exemples suivants sont des **expériences de pensée pour développer l'intuition**. En réalité, les concepts ne sont PAS soigneusement rangés dans des dimensions individuelles !

**En pratique :** un concept comme le « caractère concret » serait représenté par un motif complexe distribué sur PLUSIEURS dimensions à la fois. Les embeddings réels sont plus complexes et difficiles à interpréter.

**Ces étiquettes simplifiées servent uniquement à enseigner l'idée de représentation.** Ne vous attendez pas à trouver une « dimension du sentiment » clairement identifiée dans un modèle entraîné !

**Voici donc des exemples hypothétiques.**

**Dimension 0 :** pourrait représenter le « caractère concret », dans cette expérience de pensée.

- « pizza » = 0.90 : très concret, un objet tangible.
- « love » = -0.40 : un concept abstrait.
- « I » = 0.01 : neutre, un pronom.

**Dimension 1 :** pourrait représenter le « sentiment ou l'émotion ».

- « love » = 0.60 : une émotion positive !
- « pizza » = -0.10 : un objet plutôt neutre ; la source imagine une valeur légèrement négative parce que tous les aliments ne sont pas appréciés.
- « I » = -0.20 : un pronom neutre.

**Dimension 2 :** pourrait représenter « l'usage fréquent dans des contextes positifs ».

- « I » = 0.30 : dans cette hypothèse, il commence souvent des phrases positives.
- « love » = 0.00 : il apparaît dans des contextes positifs comme négatifs.
- « pizza » = -0.20 : un aliment, neutre.

**Dimension 3 :** pourrait représenter « la relation aux objets physiques ou aux actions ».

- « I » = 0.15 : lié à une personne.
- « love » = 0.25 : une action ou un sentiment.
- « pizza » = -0.80 : un objet physique !

**Et ainsi de suite…**

Le modèle découvre les informations utiles pour prédire le token suivant : caractère concret ou abstrait, contexte positif ou négatif, rôle grammatical, etc. Il apprend une représentation utile à cette tâche !

**L'émergence de régularités :** après l'entraînement, des mots de sens proches peuvent avoir des embeddings proches. Exemple illustratif :

- « pizza » = `[0.90, -0.10, -0.20, -0.80, 0.50, 0.75]`.
- « pasta » ≈ `[0.88, -0.12, -0.18, -0.79, 0.48, 0.73]` ← Très proche !
- « love » = `[-0.40, 0.60, 0.00, 0.25, 0.90, -0.30]`.
- « adore » ≈ `[-0.38, 0.63, 0.02, 0.27, 0.88, -0.28]` ← Très proche !

Des régularités se forment parce que les mots de sens proches apparaissent souvent dans des contextes similaires, et que le modèle apprend à partir de ces contextes.

### Pourquoi davantage de dimensions ?

**Avec deux dimensions :** notre analogie pourrait représenter « positif ou négatif » et « nom ou verbe ».

- La représentation serait très limitée.
- Elle distinguerait mal « happy » et « ecstatic », deux adjectifs positifs.

**Avec six dimensions**, comme dans ce tutoriel : plusieurs grandes propriétés peuvent contribuer à la représentation.

- Assez pour illustrer des régularités simples.
- Assez peu pour calculer à la main !

**Avec 12 288 dimensions**, comme dans la configuration GPT-3 citée : la représentation dispose d'une capacité bien plus grande pour contribuer au traitement de nuances telles que :

- Les différences entre « happy », « joyful », « content » et « ecstatic ».
- Les sens dépendant du contexte, comme « bank », banque ou rive.
- Les emplois métaphoriques, les indices de sarcasme, le registre et les connotations culturelles.
- Tout ce qui fait la richesse du langage !

Ces sens contextuels sont construits par les couches du modèle ; ils ne sont pas tous distingués par la seule ligne initiale de la table d'embeddings.

**Analogie : décrire une peinture.**

- Deux couleurs : une esquisse très sommaire.
- Six couleurs : la scène dans ses grandes lignes.
- 12 288 couleurs : davantage de nuances et de textures.

Même intuition pour les représentations des mots : une plus grande dimension peut permettre une représentation plus riche et plus nuancée.

### Comment les nombres aléatoires acquièrent-ils du sens ? (aperçu)

**Vous vous demandez peut-être :** « Si les embeddings sont aléatoires au départ, comment apprennent-ils à représenter le sens ? »

**Réponse courte :** par l'entraînement ! Le modèle fait des prédictions, mesure ses erreurs et ajuste ses nombres pour s'améliorer. Sur de très nombreux exemples, des régularités utiles émergent.

**Réponse un peu plus longue :** imaginez que vous apprenez à faire un gâteau. Votre première tentative est mauvaise : vous avez choisi les proportions au hasard. Quelqu'un le goûte : « Trop sucré, il faut davantage de farine. » Vous ajustez. La tentative suivante est meilleure. Après des centaines d'essais, la recette s'améliore. Les valeurs des embeddings sont ajustées de la même manière à partir du retour fourni par les erreurs de prédiction.

**Ne vous inquiétez pas encore des détails !** Nous verrons précisément COMMENT fonctionne cet apprentissage au **[chapitre 11 : entraîner le Transformer (anglais)](https://github.com/lviaud/Building_a_Transformer/blob/main/chapter-11-training-the-transformer.md)**. Pour l'instant, retenez :

- Les embeddings COMMENCENT avec des valeurs aléatoires, sans signification apprise.
- L'entraînement les AJUSTE en fonction des erreurs de prédiction.
- Ils finissent par REPRÉSENTER des régularités utiles : des mots proches peuvent avoir des vecteurs proches.

Une structure utile émerge ainsi d'une initialisation aléatoire grâce à l'entraînement !

**À retenir pour ce chapitre :**

- Retrouver un embedding est rapide : il suffit de prendre une ligne du tableau.
- Chaque token est représenté par $d_{\text{model}}$ nombres, six dans notre exemple.
- Ces nombres contribuent ensemble à sa représentation.
- Le modèle apprend cette représentation pendant l'entraînement.

---

**Navigation dans le cours :** [Précédent : chapitre 2 — Tokenisation](chapter-02-tokenization.md) | [Suivant : chapitre 4 — Encodage positionnel](chapter-04-positional-encoding.md)
