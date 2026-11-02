# Notes de traduction et précisions techniques

Cette page distingue la traduction des chapitres 1 à 6 des précisions apportées lors de leur préparation. Le texte original est celui de **Rimom Costa**, dans le dépôt [Building_a_Transformer](https://github.com/rimomcosta/Building_a_Transformer), au commit `d8e275a2c41426889e116bfa6d64090805b3bd9a`.

## Choix de traduction

- Les sections des chapitres 1 à 6 sont conservées, ainsi que les tableaux, formules et exemples numériques. Les blocs mathématiques sont adaptés à l'affichage MathJax sans changer les expressions.
- Les phrases anglaises associées aux vecteurs restent en anglais pour conserver leurs positions. Par exemple, « I love pizza » comporte trois mots ; sa traduction française ne conserverait pas nécessairement ce découpage.
- Les légendes et explications sont traduites. Les identifiants symboliques du pseudo-code restent inchangés.
- Les valeurs arrondies proviennent de la source. Des égalités numériques sont parfois approximatives : lire « ≈ » lorsque les valeurs affichées sont arrondies.
- Les chapitres 1 à 6 sont reliés en français. Les liens vers les chapitres 0, 7, 11 et 12 conduisent aux versions anglaises du fork.

## Précisions sur les chapitres 1 à 3

### Tokens et mots ne sont pas équivalents

Un token peut être un mot, un sous-mot ou une unité plus petite. Le cours utilise des mots entiers dans son exemple pour simplifier. Les identifiants 123, 567 et 999 sont ceux du vocabulaire fictif du tutoriel, pas les identifiants garantis d'un tokeniseur existant. Les tokens spéciaux `<pad>`, `<start>` et `<end>` sont également ceux de cet exemple.

### Vocabulaire et accès aux embeddings

La taille du vocabulaire constitue un compromis entre la longueur des séquences de tokens et la taille des matrices du modèle. Elle ne signifie pas que chaque recherche d'embedding exige de parcourir toutes les entrées : l'identifiant permet d'accéder à une ligne. Avec 50 000 entrées et 12 288 dimensions, le calcul donne 614 400 000 valeurs ; la valeur de 614 millions citée est arrondie.

### BPE : une intuition de découpage, des règles de fusion

Le chapitre 2 présente le découpage en morceaux connus comme une intuition. Dans BPE, l'entraînement apprend des fusions de paires fréquentes. L'encodage applique ensuite les règles et leur priorité à la séquence d'unités initiales ; ce n'est pas une recherche arbitraire de deux morceaux dans un dictionnaire. Voir la [présentation de BPE par Hugging Face](https://huggingface.co/docs/course/en/chapter6/5).

Le déterminisme suppose un tokeniseur et des réglages fixes, sans stratégie d'échantillonnage. La réversibilité exacte dépend aussi de la normalisation : par exemple, convertir les majuscules en minuscules peut faire perdre la casse d'origine. Les [composants d'un tokeniseur](https://huggingface.co/docs/tokenizers/components) précisent le rôle de la normalisation et du décodage.

### La dimension d'un modèle n'est pas universelle

La valeur 12 288 citée dans le chapitre 3 correspond à GPT-3 à 175 milliards de paramètres. Elle ne décrit pas tous les modèles de conversation. La traduction précise ce point au lieu de présenter cette taille comme universelle. Voir le tableau 2.1 de [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).

### Sens distribué, antonymes et contexte

Les étiquettes « nourriture », « sentiment », « nom ou verbe » servent d'analogies. Une dimension ne correspond pas nécessairement à une propriété humaine identifiable, et augmenter le nombre de dimensions ne garantit pas à lui seul une meilleure compréhension.

La proximité entre vecteurs ne constitue pas une simple échelle d'opposition sémantique. Des antonymes peuvent apparaître dans des contextes semblables et avoir des représentations proches ; ils ne sont pas obligatoirement éloignés ou orientés dans des directions opposées.

Enfin, l'embedding initial d'un token est une ligne de la table. Les représentations qui distinguent ses sens selon le contexte sont construites dans les couches du Transformer. Les nombres du tableau sont des valeurs d'illustration : on ne doit pas y lire une signification sémantique déjà apprise avant l'entraînement.

## Précisions sur le chapitre 4

### Longueurs et indices de certains exemples

La phrase « I really love eating delicious homemade pizza with my best friends » contient **11 mots**, indexés de 0 à 10, bien que la source parle de 10 mots et divise par 10. Pour une normalisation par la longueur, on diviserait par 11 ; pour faire correspondre les deux extrémités à 0 et 1, on peut diviser par 10, soit le dernier indice. Ces conventions sont différentes.

Dans « I love delicious homemade pizza », « delicious » est à l'indice 2 et « homemade » à l'indice 3. Le bloc source associe la position normalisée 0.6 à « delicious », alors qu'elle correspond à « homemade » pour un diviseur de 5. Les valeurs sont conservées dans la traduction pour permettre la comparaison.

### Le sinus seul ne détermine pas un angle

Le sinus prend plusieurs fois une même valeur dans un cycle. Par exemple, $\sin(\pi/6)=\sin(5\pi/6)=1/2$. C'est le **couple** sinus/cosinus qui détermine l'angle modulo $2\pi$. La traduction le précise à l'endroit de l'affirmation ambiguë.

### La base 10 000 n'est pas le dernier diviseur

Pour une dimension paire $d$, le dernier indice de paire est $i=d/2-1$. Le dernier diviseur est donc $10000^{1-2/d}$, inférieur à 10 000. Avec $d=6$, il vaut environ 464.159 et sa période est d'environ $2\pi\times464.159\approx2916.4$ positions. La période de 62 800 positions évoquée dans le texte correspondrait à un diviseur de 10 000, et non au cas à six dimensions calculé ensuite.

### Gradients et précision numérique

Pour $x+p$ avec $p$ constant, la dérivée par rapport à $x$ est 1. Le seul fait d'ajouter une grande constante ne rend donc pas automatiquement les mises à jour de $x$ plus petites. L'exemple du taux d'apprentissage exprime une intuition d'échelle, sans démontrer à lui seul un ralentissement.

En float32, autour de 1000, le pas entre valeurs représentables vaut $2^{-14}\approx0.000061$. Une variation de 0.001 n'est donc pas systématiquement perdue. Les erreurs dépendent de l'amplitude et du format utilisé. La fonction [NumPy spacing](https://numpy.org/doc/stable/reference/generated/numpy.spacing.html) permet de mesurer cet espacement.

### Calculer une position ne garantit pas l'extrapolation

La formule sinusoïdale peut être évaluée à des positions absentes de l'entraînement. Cela ne garantit ni la qualité sur de longues séquences ni une distinction parfaite de toutes les positions dans une précision numérique finie. Les affirmations sur des longueurs « infinies » décrivent la formule, et non une garantie de performance du modèle.

### Quels paramètres sont entraînés ?

La comparaison du chapitre oppose les embeddings aux encodages sinusoïdaux fixes. Les autres paramètres du Transformer, notamment les projections d'attention et les couches de calcul, sont également entraînés. Dire que « seuls les embeddings apprennent » doit se lire dans le cadre de cette comparaison.

### RoPE et transformations relatives

RoPE applique les rotations aux vecteurs de requêtes et de clés de l'attention ; la description comme une simple rotation des embeddings de mots est un raccourci. Voir l'article [RoFormer](https://arxiv.org/abs/2104.09864).

La matrice de la note avancée correspond à la fréquence 1 et à l'ordre de coordonnées sinus/cosinus. Pour chaque autre paire, remplacer $k$ par $k/10000^{2i/d_{\text{model}}}$. La transformation complète est formée de ces blocs. Son existence ne signifie pas que le Transformer apprend explicitement une matrice nommée $M_k$.

Enfin, les sinus et cosinus ne sont pas les seules fonctions ayant une relation de décalage linéaire : les exponentielles et une base polynomiale adaptée en possèdent aussi. Leur intérêt ici réside notamment dans la combinaison de valeurs bornées et de plusieurs fréquences.

La section 3.5 de [Attention Is All You Need](https://arxiv.org/html/1706.03762v7#S3.SS5) présente la construction sinusoïdale et l'hypothèse selon laquelle ces relations facilitent l'apprentissage des positions relatives. Les précisions et calculs ci-dessus sont des notes de relecture, distinctes du cours original.

## Précisions sur le chapitre 5

### Des projections par tête, pas nécessairement un découpage de l'entrée

La source commence par diviser le vecteur de dimension 6 en deux morceaux de dimension 3, puis applique des matrices 3×3. Cela constitue un petit modèle pédagogique différent de l'attention multi-têtes standard.

Dans la construction standard, chaque tête reçoit la représentation complète et dispose de projections apprises vers son propre espace. Les dimensions des matrices sont décrites dans la section 3.2.2 de [Attention Is All You Need](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2).

Pour une entrée de dimension 6 et deux têtes de dimension 3, chaque projection de tête est donc 6×3. Sans biais, Q, K et V totalisent $3\times2\times6\times3=108$ paramètres. La projection de sortie 6×6 ajoute 36 paramètres, soit **144**, autant que trois projections 6×6 et une projection de sortie 6×6 pour une tête de dimension 6. Les nombres 54 et 90 de la source décrivent son découpage simplifié, pas ce décompte standard.

Les spécialisations linguistiques attribuées aux têtes sont des exemples hypothétiques. Elles ne sont ni imposées ni garanties. De même, le coût en temps et en mémoire dépend de l'implémentation : les poids d'attention comportent notamment une matrice par tête.

### Les pourcentages des analogies ne sont pas les softmax calculés

Les exemples de rencontres et l'aperçu initial utilisent des répartitions illustratives. Un calcul direct donne :

| Scores et ordre des profils | Résultat présenté par la source | Softmax recalculé |
| --- | --- | --- |
| [0.15, 0.42, 0.89] | [19 %, 31 %, 50 %] | Environ [22.70 %, 29.73 %, 47.57 %] |
| [95, 55, 15], Alex, Casey, Blake | [80 %, 18 %, 2 %] | Alex reçoit presque 100 % ; les autres poids valent environ $4.25\times10^{-18}$ et $1.80\times10^{-35}$ |
| [220, 180, 140], Alex, Casey, Blake | [85 %, 13 %, 2 %] | Même répartition que la ligne précédente, puisque les écarts sont identiques |

Les deux dernières valeurs sont des poids entre 0 et 1, et non des pourcentages. On peut stabiliser le calcul en soustrayant le maximum des scores avant l'exponentielle : cela ne modifie pas le résultat mathématique.

Softmax conserve l'ordre, mais ne donne pas toujours une distribution très concentrée : celle-ci dépend des écarts entre scores. Pour des scores finis, les poids sont strictement positifs en arithmétique exacte ; des masques ou la précision numérique peuvent néanmoins produire des zéros.

### Le produit scalaire dépend aussi de la norme

Le produit scalaire est un score de compatibilité qui dépend de l'angle **et de la norme** des vecteurs. « Produit élevé » ne signifie donc pas automatiquement « mêmes significations », et « produit négatif » ne signifie pas « antonymes ». Les questions et les clés sont des projections apprises : leurs interprétations dans les analogies ne doivent pas être prises comme des étiquettes littérales.

### Relire la matrice de l'exemple Beyoncé

Les colonnes affichées sont « Beyoncé, Vous ». Dans cet ordre, la seconde ligne [0.001 %, 99.999 %] donnerait presque tout le poids à « Vous », contrairement au commentaire. Pour représenter l'intention de la source en gardant cet ordre de colonnes, les deux valeurs de cette ligne devraient être permutées. Le tableau original est conservé et signalé dans la traduction.

### Vérifications numériques de la tête 1

Les formules et nombres de la source sont conservés, mais un calcul avec ses matrices révèle notamment une erreur dans la troisième composante de $V_{\text{love}}$ :

$$0.44\times0.2+1.14\times0.1+0.05\times0.3=0.217$$

La source donne -0.055. En recalculant depuis les vecteurs d'entrée et les matrices affichées, on obtient pour « I » :

- Scores non mis à l'échelle : environ [-0.042727, -0.024617, 0.284086].
- Poids après division par $\sqrt{3}$ et softmax : environ [0.310735, 0.314001, 0.375263].
- Sortie recalculée de la tête 1 : environ **[0.170630, 0.198709, 0.225533]**, au lieu de [0.1706, 0.1987, 0.1401].

Les matrices de la tête 2 ne sont pas indiquées. Sa sortie [0.245, -0.089, 0.156] est donc traitée comme une donnée illustrative, pas comme un résultat reproductible à partir des seules informations du chapitre.

Enfin, le produit du vecteur concaténé **tel qu'affiché par la source** par sa matrice $W^O$ vaut :

$$[0.07619, 0.14538, 0.09446, 0.04753, 0.16343, 0.06277]$$

Il ne correspond pas à [0.25, -0.31, 0.42, 0.18, -0.09, 0.33]. Ce dernier vecteur est conservé comme illustration dans les chapitres 5 et 6 pour maintenir leur continuité, sans le présenter comme le résultat vérifié de cette multiplication.

## Précisions sur le chapitre 6

### Probabilité de suppression et nombre effectivement supprimé

Un taux de 10 % est une probabilité par activation, pas la garantie de supprimer exactement 10 % des composantes à chaque passage. Avec six composantes indépendantes, le nombre moyen supprimé est 0.6, et la probabilité de n'en supprimer aucune vaut $0.9^6=0.531441$, soit environ 53.14 %. Le premier masque, composé uniquement de 1, est donc tout à fait possible.

### Préserver l'espérance avec le dropout inversé

Dans la convention du chapitre, chaque activation conservée est divisée par $1-p$ pendant l'entraînement. Si $m$ vaut 1 avec probabilité $1-p$ et 0 sinon, la sortie est $mx/(1-p)$ et son espérance vaut $x$. Cela ne garantit pas de préserver la somme pour chaque masque, notamment avec des composantes de signes différents. À l'évaluation, cette opération devient l'identité. Voir la [documentation de Dropout dans PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).

Le terme « désactiver un neurone » est une image : ici, on met des activations à zéro pour un passage, sans supprimer définitivement de paramètres du réseau.

### Emplacements, taux et analogie d'ensemble

Les deux emplacements retenus par le chapitre ne forment pas une liste exhaustive. D'autres emplacements existent selon l'architecture, et les taux 0.1 à 0.3 cités ne sont pas une règle générale. Leur pertinence doit être évaluée dans le contexte de l'entraînement.

L'image de sous-réseaux partageant leurs poids est utile pour comprendre le dropout. Pour un réseau non linéaire, utiliser toutes les activations à l'inférence n'équivaut toutefois pas, en général, à calculer exactement la moyenne des sorties de tous les sous-réseaux.


