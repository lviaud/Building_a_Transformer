# Chapitre 5 : L'auto-attention multi-têtes (le cœur du Transformer)

> Traduction française non officielle du cours de Rimom Costa.
> [Texte original](https://github.com/rimomcosta/Building_a_Transformer/blob/d8e275a2c41426889e116bfa6d64090805b3bd9a/chapter-05-multi-head-self-attention.md).
> Version source : `d8e275a2c41426889e116bfa6d64090805b3bd9a` ; traduction du 2 novembre 2026.
>
> Les phrases anglaises et les valeurs des exemples sont conservées pour assurer la continuité du cours. Le découpage des dimensions par tête est une simplification de l'auteur. Plusieurs résultats numériques de la source sont illustratifs ou incohérents ; les [notes de traduction](notes-traduction.md) expliquent ces limites et donnent des vérifications séparées.

Nous voici au cœur du mécanisme ! Cette innovation a joué un rôle essentiel dans le succès des Transformers. Ce chapitre explique comment les mots « communiquent » et déterminent quelles informations prendre en compte.

## Partie 1 : Comprendre le concept fondamental

### Le problème de communication

Nous disposons maintenant de trois vecteurs tenant compte des positions pour « I love pizza » :

```
"I":     [0.01, 0.80, 0.30, 1.15, -0.05, 1.11]
"love":  [0.44, 1.14, 0.05, 1.25,  0.90, 0.70]
"pizza": [1.81, -0.52, -0.11, 0.20,  0.50, 1.75]
```

**Mais un problème subsiste !** Chaque mot ne dispose encore que d'informations sur LUI-MÊME. Le vecteur de « I » représente ce mot et sa position, mais ne contient pas encore d'information issue de « love » ou de « pizza ». Comme des personnes côte à côte qui ne se parlent pas !

Pour exploiter le contexte, les mots doivent se REGARDER et échanger des informations :

- « I » doit identifier l'action qu'il accomplit : « love ».
- « love » doit identifier qui aime, « I », et ce qui est aimé, « pizza ».
- « pizza » doit identifier son rôle d'objet de cette action.

**C'est le rôle de l'attention :** permettre aux tokens d'échanger de l'information et d'actualiser leurs représentations en fonction du contexte !

### Pourquoi cela compte : des exemples concrets

**Exemple 1 : Trouver l'antécédent d'un pronom**

```
"The cat slept because it was tired."
```

Dans cette phrase, « Le chat dormait parce qu'il était fatigué », « it » doit déterminer son antécédent. En examinant les autres mots :

- « The » : pas un antécédent.
- « cat » : OUI ! Ce nom est un antécédent probable de « it ».
- « slept » : un verbe, pas un antécédent.
- « because » : une conjonction, pas un antécédent.
- « was » : un verbe, pas un antécédent.
- « tired » : un adjectif, pas un antécédent.

L'attention peut aider la représentation de « it » à exploiter celle de « cat » !

**Exemple 2 : Comprendre les relations**

```
"The chef cooked the pasta."
```

Dans « Le chef a cuisiné les pâtes » :

- « chef » doit identifier son rôle d'**agent**, le sujet.
- « cooked » doit identifier qui cuisine, « chef », et ce qui est cuisiné, « pasta ».
- « pasta » doit identifier son rôle d'**objet de l'action**.

L'attention permet à chaque token de recueillir ces informations contextuelles !

**Exemple 3 : Les relations entre un nom et ses qualificatifs**

```
"The delicious homemade pizza"
```

Dans « La délicieuse pizza maison » :

- « pizza » doit tenir compte de « delicious » et « homemade », qui la décrivent.
- « delicious » et « homemade » doivent être reliés à « pizza », qu'ils qualifient.

Sans échange d'information, ces mots restent isolés. L'attention relie leurs représentations !

> **Note de l'auteur :** dans ce chapitre, nous laissons temporairement les tokens examiner toute la phrase pour comprendre les calculs. Au **[chapitre 12 (anglais)](https://github.com/lviaud/Building_a_Transformer/blob/main/chapter-12-causal-masking.md)**, nous ajouterons le **masquage causal**, qui empêche un token d'accéder aux positions futures dans un modèle de type décodeur autorégressif.

### L'analogie de l'application de rencontres (comprendre la pondération)

Avant les mathématiques, prenons une analogie : une application de rencontres !

**Vous recherchez une personne compatible.** Chaque profil possède :

- Une **accroche ou biographie**, la **clé (Key)** : ce que la personne met en avant.
- Un **profil complet**, la **valeur (Value)** : les informations disponibles sur cette personne.
- Vos **préférences**, la **requête (Query)** : ce que vous recherchez.

**Supposons que vous cherchiez quelqu'un qui :**

- Aime la pizza : très important !
- Souhaite se marier un jour : EXTRÊMEMENT important, un critère décisif !
- Aime la randonnée : un avantage appréciable.

**Vous voyez trois profils.**

**Profil A : Alex**

- Accroche, la clé : « Aime la pizza et la randonnée, souhaite se marier bientôt ».
- Score de compatibilité avec l'accroche : 95 %, presque tous les critères !
- Profil complet, la valeur : [informations sur sa vie, ses intérêts et sa personnalité].

**Profil B : Blake**

- Accroche : « Passionné de pizza et de sports d'aventure, ne souhaite pas se marier ».
- Score de compatibilité : 70 %, certains critères correspondent.
- Profil complet : [informations sur sa vie, ses intérêts et sa personnalité].

**Profil C : Casey**

- Accroche : « Préfère les sushis, souhaite se poser et se marier ».
- Score de compatibilité : 45 %, le mariage correspond, mais pas la nourriture.
- Profil complet : [informations sur sa vie, ses intérêts et sa personnalité].

**Voici l'idée essentielle de l'attention pondérée.**

**Étape 1 : Calculer la compatibilité** entre les clés des profils et votre requête.

```text
Alex :  95% - aime la pizza ✓, la randonnée ✓, souhaite se marier ✓
Blake : 70% - aime la pizza ✓, l'aventure ✓, pas de mariage ✗
Casey : 45% - préfère d'autres aliments ✗, souhaite se marier ✓
```

**Étape 2 : Toutes vos préférences n'ont pas la même importance !**

Vous leur attribuez des poids :

- Aimer la pizza : poids = 1.0, important.
- Souhaiter se marier : poids = 10.0, un critère décisif !
- Aimer la randonnée : poids = 0.5, un avantage.

**Étape 3 : Recalculer avec ces préférences pondérées**

**Alex :**

```text
Pizza : OUI (1.0 × score élevé)
Mariage : OUI (10.0 × score élevé) ← Forte contribution !
Randonnée : OUI (0.5 × score élevé)
Score pondéré : 95 (très élevé !)
```

**Blake :**

```text
Pizza : OUI (1.0 × score élevé)
Mariage : NON (10.0 × zéro !) ← Forte pénalisation dans l'analogie !
Aventure : OUI (0.5 × score élevé)
Score pondéré : 15 (bien plus faible !)
```

**Casey :**

```text
Pizza : NON (1.0 × score faible)
Mariage : OUI (10.0 × score élevé) ← Forte contribution !
Randonnée : NON (0.5 × score faible)
Score pondéré : 55 (intermédiaire)
```

**Répartition finale de l'attention**, présentée dans la source comme le résultat de softmax :

```text
Alex :  80% ← L'essentiel du temps de lecture !
Casey : 18% ← Une partie de l'attention
Blake : 2%  ← Très peu d'attention
```

Ces pourcentages illustrent l'analogie ; ils ne sont pas le softmax exact des scores précédents. Voir les notes de traduction.

**Étape 4 : Recueillir l'information par une combinaison pondérée des valeurs**

Vous ne lisez pas un seul profil, mais TOUS, en leur consacrant un temps proportionnel à l'attention :

```text
Information recueillie =
  80% × profil complet d'Alex +
  18% × profil complet de Casey +
  2% × profil complet de Blake
```

Vous apprenez SURTOUT des choses sur Alex, à 80 %, un peu sur Casey, à 18 %, et très peu sur Blake, à 2 %.

**C'est le principe de l'attention !**

- **Requête (Query) :** quelle information ce token recherche-t-il ?
- **Clé (Key) :** que propose chaque token ?
- **Scores :** compatibilité entre la requête et chaque clé.
- **Poids :** toutes les caractéristiques n'ont pas la même influence !
- **Valeurs (Values) :** informations à combiner.
- **Sortie (Output) :** combinaison pondérée, davantage influencée par les tokens recevant le plus d'attention.

### L'attention asymétrique : la relation entre une célébrité et son public

**Autre idée essentielle : l'attention est orientée et n'est pas nécessairement symétrique !**

**Exemple :** vous êtes une grande admiratrice ou un grand admirateur de Beyoncé.

**De votre point de vue :**

- Votre requête : « Je veux tout savoir sur Beyoncé ! »
- Sa clé : « Je suis Beyoncé, chanteuse, artiste, icône ».
- Votre attention vers elle : 99 % !
- Vous recueillez les informations de sa valeur : [sa musique, son style, ses réalisations].

**Vous lui consacrez 99 % de votre attention !**

**De son point de vue :**

- Sa requête : « Qui sont les personnes importantes de mon entourage professionnel ? »
- Votre clé : « Je suis une personne parmi vos fans ».
- Son attention vers vous : 0.001 %, puisqu'elle a des millions de fans !
- Elle remarque à peine votre présence.

**Elle ne vous consacre que 0.001 % de son attention !**

**La matrice est ASYMÉTRIQUE :**

```text
         Beyoncé    Vous
Vous     [99%,      1%    ] ← Vous regardez surtout Beyoncé
Beyoncé  [0.001%,   99.999%] ← Elle regarde surtout elle-même ou ses proches
```

Les étiquettes de ce tableau sont reprises de la source ; leur ordre présente une incohérence signalée dans les notes.

**Le même principe s'applique aux Transformers !**

Dans « The famous chef cooked pasta » :

- « chef » pourrait prêter attention à « famous », qui le décrit.
- Mais « famous » pourrait accorder davantage d'attention à « chef » que l'inverse.
- Les poids des deux relations ne sont pas obligatoirement égaux !

Chaque token possède sa propre requête : l'attention a donc une direction !

### L'analogie de la bibliothèque (un autre point de vue)

Pour consolider l'idée, imaginez que vous cherchez des informations sur la cuisine italienne dans une bibliothèque.

**Vous, la requête :** « Je cherche des informations sur la cuisine italienne. »

Vous passez devant les rayonnages. Le titre visible sur chaque dos de livre est sa clé :

- Livre 1 : « Recettes de pâtes italiennes » → **très pertinent !** La requête correspond bien à cette clé.
- Livre 2 : « Guide des vins français » → un peu pertinent : gastronomie et Europe.
- Livre 3 : « Physique quantique » → aucune pertinence ici.
- Livre 4 : « Guide de voyage en Italie » → moyennement pertinent : l'Italie, mais pas la cuisine.
- Livre 5 : « Cuisine mexicaine » → moyennement pertinent : la cuisine, mais pas italienne.

**Vous vous concentrez naturellement sur le livre 1**, dont le titre correspond le mieux à votre recherche !

Vous le prenez pour lire son contenu, la valeur. Vous pouvez aussi consulter les livres 2, 4 et 5, tout en consacrant l'essentiel du temps au premier.

**L'idée :** l'information recueillie, la valeur, provient des livres auxquels vous avez accordé de l'attention, selon la correspondance entre leur titre, la clé, et votre recherche, la requête.

**Et chaque token de la phrase effectue cette opération simultanément !**

Dans « I love pizza » :

- « I » demande : « À qui ou à quoi suis-je relié dans cette phrase ? »
- « love » demande : « Quel sujet accomplit cette action ? Quel objet la reçoit ? »
- « pizza » demande : « Quelle action me concerne ? Qui l'accomplit ? »

Chaque token examine TOUS les tokens, y compris lui-même, et détermine le poids à attribuer à chacun !

### Les trois composantes : requête, clé et valeur

Examinons ces trois termes.

**Requête (Q) : « Que suis-je en train de chercher ? »**

- C'est la question associée à chaque token dans l'analogie.
- Comme une recherche saisie dans un moteur de recherche.
- Exemple : « it », dans « The cat slept because it was tired », pourrait rechercher le nom auquel il renvoie.

**Clé (K) : « Quelle information est-ce que je propose ? »**

- C'est la manière dont le token se présente aux requêtes.
- Comme un titre de livre ou une accroche.
- Exemple : « cat » pourrait proposer des informations utiles à un token recherchant un antécédent nominal ou un animal.

**Valeur (V) : « Quel est le contenu à transmettre ? »**

- C'est l'information qui entre dans la combinaison pondérée.
- Comme le contenu du livre.
- Exemple : le vecteur de valeur de « cat » transporte des informations issues de sa représentation.

**L'idée essentielle :** Q, K et V proviennent de la même représentation du token, mais de transformations DIFFÉRENTES, à l'aide de matrices de poids distinctes. Comme trois points de vue sur le même mot :

- Q : « De quoi a-t-il besoin ? »
- K : « Que met-il en avant ? »
- V : « Quelle information transmet-il ? »

### Le processus d'attention (vue d'ensemble)

Voici ce qui se passe pour CHAQUE token, avec « I » comme exemple.

**Étape 1 : Créer Q, K et V pour tous les tokens**

```text
"I" crée :     Q_I, K_I, V_I
"love" crée :  Q_love, K_love, V_love
"pizza" crée : Q_pizza, K_pizza, V_pizza
```

**Étape 2 : « I » calcule les scores d'attention**

```text
"I" compare sa requête aux clés de tous les tokens :
- Compatibilité de Q_I avec K_I ? (Quel poids pour moi-même ?)
- Compatibilité de Q_I avec K_love ? (Quel poids pour "love" ?)
- Compatibilité de Q_I avec K_pizza ? (Quel poids pour "pizza" ?)
```

**Étape 3 : Convertir les scores en probabilités**

```text
Scores bruts : [0.15, 0.42, 0.89]
Après softmax, selon la source : [19%, 31%, 50%] ← Somme : 100% !
```

Les pourcentages de cet aperçu sont illustratifs, comme le précise la note sur softmax.

**Étape 4 : Former une combinaison pondérée des valeurs**

```text
Nouvelle représentation de "I" =
  19% × V_I + 31% × V_love + 50% × V_pizza
```

**Résultat :** la représentation de « I » est ACTUALISÉE avec des informations provenant de tous les tokens, surtout de « pizza » dans cet exemple, qui reçoit 50 % de l'attention !

**Et cette opération concerne TOUS les tokens SIMULTANÉMENT !**

- « I » actualise sa représentation en tenant compte de tous.
- « love » fait de même.
- « pizza » fait de même.

Le tout en parallèle ! Cette organisation contribue à l'efficacité des Transformers.

---

## Partie 2 : L'attention multi-têtes (pourquoi plusieurs points de vue ?)

Avant les calculs, comprenons un choix d'architecture essentiel : **pourquoi plusieurs têtes d'attention ?**

### Les limites d'une tête unique

Avec UN seul mécanisme d'attention, il faudrait apprendre à prendre en compte :

- Les relations grammaticales : sujet, verbe, objet.
- Les relations sémantiques : synonymes, antonymes.
- Les dépendances à longue distance : antécédents des pronoms.
- Les liens entre adjectifs et noms.
- Les relations temporelles : avant, après.
- Les relations causales : parce que, donc.
- Et bien d'autres !

Cela fait beaucoup pour une seule tête ! Comme confier à une même personne les ventes, le marketing, l'ingénierie, le service client et la comptabilité.

### La solution : plusieurs têtes d'attention

**L'idée essentielle : des têtes différentes peuvent apprendre des relations différentes !**

**Dans notre exemple**, avec $d_{\text{model}} = 6$ et deux têtes :

- **Tête 1 :** pourrait relier les sujets, leurs verbes et les objets, une structure grammaticale.
- **Tête 2 :** pourrait relier les qualificatifs aux mots qu'ils décrivent.

**Dans la configuration GPT-3 citée, avec 96 têtes :**

- Tête 1 : retrouver les antécédents des pronoms.
- Tête 2 : relier adjectifs et noms.
- Tête 3 : relier verbes et sujets.
- Tête 4 : tenir compte de la négation.
- Tête 5 : représenter des relations numériques.
- Et 91 autres motifs possibles !

Ce sont des spécialisations hypothétiques pour l'intuition, pas des rôles imposés à ces têtes.

**Analogie : une équipe de reportage.**

- Caméra 1 : un plan large de la scène.
- Caméra 2 : un gros plan de la personne qui parle.
- Caméra 3 : les réactions du public.
- Caméra 4 : le contexte à l'arrière-plan.

Chaque caméra fournit un point de vue. Le reportage final les combine !

### Comment fonctionne l'attention multi-têtes

**Hyperparamètres de notre exemple :**

- **Nombre de têtes :** $h = 2$, contre 96 dans la configuration GPT-3 citée.
- **Dimension d'une tête :** $d_k = d_{\text{model}} / h = 6 / 2 = 3$.

**Dans le modèle simplifié de ce chapitre, chaque tête travaille sur un sous-ensemble de dimensions :**

- **Tête 1 :** dimensions [0, 1, 2] de chaque token.
- **Tête 2 :** dimensions [3, 4, 5].

**Pourquoi ce découpage dans l'exemple ?** Il donne aux deux têtes des informations distinctes : les dimensions 0–2 pour la première, 3–5 pour la seconde. Cette construction simplifie les calculs à la main.

**Dans l'architecture standard, chaque tête projette au contraire la représentation complète vers son propre espace.** Le découpage des entrées ci-dessous est un choix pédagogique de la source ; voir les notes de traduction.

**Découpons nos vecteurs :**

```
"I":     [0.01, 0.80, 0.30, | 1.15, -0.05, 1.11]
          ︸─────────────────︷  ︸──────────────────︷
              Tête 1                 Tête 2

"love":  [0.44, 1.14, 0.05, | 1.25,  0.90, 0.70]
          ︸─────────────────︷  ︸──────────────────︷
              Tête 1                 Tête 2

"pizza": [1.81, -0.52, -0.11, | 0.20,  0.50, 1.75]
          ︸──────────────────︷  ︸──────────────────︷
              Tête 1                 Tête 2
```

**La procédure complète de l'exemple :**

1. La tête 1 calcule l'attention sur les dimensions [0,1,2] et produit une sortie à trois dimensions par token.
2. La tête 2 utilise [3,4,5] et produit également trois dimensions.
3. **Concaténer :** réunir les deux sorties pour obtenir six dimensions par token.
4. **Projeter :** mélanger ces sorties par une transformation finale.

**Les avantages recherchés :**

- ✓ Des têtes susceptibles de se spécialiser.
- ✓ Plusieurs représentations complémentaires.
- ✓ Une sortie unique qui combine les têtes.
- ✓ Des calculs réalisables en parallèle.

**🎓 Note sur le coût de calcul :**

L'idée est qu'avec $h=2$ têtes de dimension $d_k=3$, soit six dimensions au total, on peut garder un coût du même ordre que celui d'une tête de dimension six.

La source présente le décompte suivant pour son exemple :

- **Une tête de dimension six :** trois matrices 6×6 pour Q, K et V.
  - Paramètres : 6×6 + 6×6 + 6×6 = 108.
- **Deux têtes de dimension trois, avec découpage préalable des entrées :** trois matrices 3×3 par tête.
  - Par tête : 3×3 + 3×3 + 3×3 = 27.
  - Pour deux têtes : 27 × 2 = 54.
  - Projection de sortie : 6×6 = 36.
  - Total : 54 + 36 = 90, présenté comme approximativement comparable.

Ces nombres décrivent la construction simplifiée et n'incluent pas la projection de sortie dans les deux cas de façon identique. Le décompte standard est donné dans les notes.

**L'idée à retenir :** multiplier le nombre de têtes ne multiplie pas automatiquement le coût total par $h$ si leur dimension individuelle diminue. On organise le calcul en plusieurs voies parallèles !

C'est pourquoi 96 têtes n'impliquent pas un modèle 96 fois plus lent qu'une tête de même dimension totale. Le coût exact dépend toutefois de l'implémentation et des dimensions.

---

## Partie 3 : Les mathématiques (calculs pas à pas)

Passons aux nombres ! Nous détaillons la **tête 1**. Dans l'exemple de la source, la tête 2 suit la même procédure sur les dimensions [3,4,5].

**Conseil de lecture :** pour privilégier l'intuition, parcourez les opérations, regardez les résultats, puis revenez avec une calculatrice. Vous n'avez pas besoin de tout multiplier d'une traite. Les valeurs originales sont conservées ; les incohérences repérées sont détaillées dans les notes.

### Étape 1 : Comprendre les matrices de poids

Il faut trois matrices par tête : $\mathbf{W}^Q$, $\mathbf{W}^K$ et $\mathbf{W}^V$.

**Important :** leurs valeurs commencent de manière **aléatoire**, ici avec une initialisation de type Xavier/Glorot :

$$
\text{weights} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)
$$

**Que signifie cette notation ?**

- $\sim$ signifie « échantillonné selon ».
- $\mathcal{N}(\mu, \sigma^2)$ désigne une loi normale, ou gaussienne, de moyenne $\mu$ et de variance $\sigma^2$.
- C'est la courbe en cloche des statistiques !

**Pourquoi cette formule ?**

Avec de trop grandes valeurs, comme 10.0 partout, les signaux peuvent croître excessivement à travers les couches. Avec de trop petites valeurs, comme 0.001, ils peuvent s'atténuer. L'initialisation Xavier cherche à maintenir une échelle de signal raisonnable.

Pour nos matrices $3 \times 3$ :

$$
\sigma = \sqrt{\frac{2}{3+3}} = \sqrt{\frac{2}{6}} = \sqrt{0.333} = 0.577
$$

Chaque poids est donc tiré selon une loi normale centrée en 0, d'écart-type 0.577.

**Exemple de tirages :**

- Tirage 1 : 0.2.
- Tirage 2 : -0.3.
- Tirage 3 : 0.15.
- Et ainsi de suite pour les autres poids.

Pour simplifier les calculs, utilisons les petites valeurs suivantes.

**Matrices de la tête 1, de taille $3 \times 3$ chacune :**

$$
\mathbf{W}^Q_1 = \begin{bmatrix}
0.2 & 0.3 & -0.1 \\
0.1 & -0.2 & 0.4 \\
-0.3 & 0.1 & 0.2
\end{bmatrix}, \quad
\mathbf{W}^K_1 = \begin{bmatrix}
0.1 & -0.2 & 0.3 \\
0.2 & 0.3 & -0.1 \\
0.4 & 0.1 & 0.2
\end{bmatrix}, \quad
\mathbf{W}^V_1 = \begin{bmatrix}
0.3 & 0.1 & 0.2 \\
-0.2 & 0.3 & 0.1 \\
0.1 & -0.1 & 0.3
\end{bmatrix}
$$

### Calculer Q, K et V

Nous multiplions la représentation de chaque token, limitée aux trois premières dimensions pour cette tête dans l'exemple, par ces matrices.

**Pour « I », $[0.01, 0.80, 0.30]$ :**

Calcul de la requête :

$$
\begin{aligned}
\mathbf{Q}_I &= [0.01, 0.80, 0.30] \times \mathbf{W}^Q_1 \\
&= \begin{bmatrix}
0.01 \times 0.2 + 0.80 \times 0.1 + 0.30 \times (-0.3) \\
0.01 \times 0.3 + 0.80 \times (-0.2) + 0.30 \times 0.1 \\
0.01 \times (-0.1) + 0.80 \times 0.4 + 0.30 \times 0.2
\end{bmatrix}^T \\
&= \begin{bmatrix}
0.002 + 0.08 - 0.09 \\
0.003 - 0.16 + 0.03 \\
-0.001 + 0.32 + 0.06
\end{bmatrix}^T \\
&= [-0.008, -0.127, 0.379]
\end{aligned}
$$

Calcul de la clé :

$$
\begin{aligned}
\mathbf{K}_I &= [0.01, 0.80, 0.30] \times \mathbf{W}^K_1 \\
&= [0.001 + 0.16 + 0.12, -0.002 + 0.24 + 0.03, 0.003 - 0.08 + 0.06] \\
&= [0.281, 0.268, -0.017]
\end{aligned}
$$

Calcul de la valeur :

$$
\begin{aligned}
\mathbf{V}_I &= [0.01, 0.80, 0.30] \times \mathbf{W}^V_1 \\
&= [0.003 - 0.16 + 0.03, 0.001 + 0.24 - 0.03, 0.002 + 0.08 + 0.09] \\
&= [-0.127, 0.211, 0.172]
\end{aligned}
$$

**Pour « love », $[0.44, 1.14, 0.05]$ :**

Avec la même méthode de multiplication et d'addition :

$$
\begin{aligned}
\mathbf{Q}_{\text{love}} &= [0.088 + 0.114 - 0.015, 0.132 - 0.228 + 0.005, -0.044 + 0.456 + 0.010] \\
&= [0.187, -0.091, 0.422]
\end{aligned}
$$

$$
\mathbf{K}_{\text{love}} = [0.044 + 0.228 + 0.020, -0.088 + 0.342 + 0.005, 0.132 - 0.114 + 0.010] = [0.292, 0.259, 0.028]
$$

$$
\mathbf{V}_{\text{love}} = [0.132 - 0.228 + 0.005, 0.044 + 0.342 - 0.005, 0.044 - 0.114 + 0.015] = [-0.091, 0.381, -0.055]
$$

**Pour « pizza », $[1.81, -0.52, -0.11]$ :**

$$
\mathbf{Q}_{\text{pizza}} = [0.362 - 0.052 + 0.033, 0.543 + 0.104 - 0.011, -0.181 - 0.208 - 0.022] = [0.343, 0.636, -0.411]
$$

$$
\mathbf{K}_{\text{pizza}} = [0.181 - 0.104 - 0.044, -0.362 - 0.156 - 0.011, 0.543 + 0.052 - 0.022] = [0.033, -0.529, 0.573]
$$

$$
\mathbf{V}_{\text{pizza}} = [0.543 + 0.104 - 0.011, 0.181 - 0.156 + 0.011, 0.362 - 0.052 - 0.033] = [0.636, 0.036, 0.277]
$$

---

## Partie 4 : Calculer la compatibilité (le produit scalaire)

Nous disposons de Q, K et V pour tous les tokens. **Comment mesurer la compatibilité entre une requête et une clé ?**

### Comprendre la ressemblance dans la vie quotidienne

Avant le produit scalaire, réfléchissons à la manière de comparer des préférences.

**Exemple 1 : Les goûts alimentaires**

Vos notes : [Pizza : 10/10, Sushis : 7/10, Brocolis : 2/10].

Les notes de votre ami A : [Pizza : 9/10, Sushis : 8/10, Brocolis : 1/10].

Vous vous ressemblez beaucoup : vous aimez pizza et sushis, mais peu les brocolis.

Les notes de votre ami B : [Pizza : 1/10, Sushis : 2/10, Brocolis : 10/10].

Vos goûts sont au contraire très différents !

**Comment calculer numériquement une compatibilité ?**

**Une approche intuitive :** multiplier les notes correspondantes et additionner les résultats.

```text
Vous et votre ami A :
Pizza : (10 × 9) = 90     ← Vous l'aimez tous deux !
Sushis : (7 × 8) = 56     ← Vous les aimez tous deux !
Brocolis : (2 × 1) = 2    ← Vous les aimez peu tous deux
Total : 90 + 56 + 2 = 148 (compatibilité ÉLEVÉE !)

Vous et votre ami B :
Pizza : (10 × 1) = 10     ← Vous l'aimez, votre ami non
Sushis : (7 × 2) = 14     ← Vous les aimez, votre ami peu
Brocolis : (2 × 10) = 20  ← Vous les aimez peu, votre ami beaucoup !
Total : 10 + 14 + 20 = 44 (compatibilité FAIBLE)
```

**C'est le PRODUIT SCALAIRE !** C'est l'opération utilisée pour les scores entre requêtes et clés.

### Le produit scalaire (définition mathématique)

Pour $\mathbf{a} = [a_1, a_2, a_3]$ et $\mathbf{b} = [b_1, b_2, b_3]$ :

$$
\mathbf{a} \cdot \mathbf{b} = a_1 \times b_1 + a_2 \times b_2 + a_3 \times b_3
$$

**Exemple simple :**

$$
\begin{aligned}
[2, 3, 1] \cdot [4, 1, 2] &= (2)(4) + (3)(1) + (1)(2) \\
&= 8 + 3 + 2 \\
&= 13
\end{aligned}
$$

**Vecteurs identiques :**

$$
\begin{aligned}
[1, 2, 3] \cdot [1, 2, 3] &= (1)(1) + (2)(2) + (3)(3) \\
&= 1 + 4 + 9 \\
&= 14 \quad \text{← Identical vectors = high score!}
\end{aligned}
$$

**Vecteurs opposés :**

$$
\begin{aligned}
[1, 2, 3] \cdot [-1, -2, -3] &= (1)(-1) + (2)(-2) + (3)(-3) \\
&= -1 - 4 - 9 \\
&= -14 \quad \text{← Opposite vectors = negative score!}
\end{aligned}
$$

**Vecteurs perpendiculaires :**

$$
\begin{aligned}
[1, 0, 0] \cdot [0, 1, 0] &= (1)(0) + (0)(1) + (0)(0) \\
&= 0 + 0 + 0 \\
&= 0 \quad \text{← Perpendicular vectors = zero!}
\end{aligned}
$$

**L'intuition proposée :**

- **Produit scalaire fortement positif :** compatibilité élevée dans cet exemple.
- **Produit proche de zéro :** faible alignement ; il vaut zéro pour des vecteurs perpendiculaires.
- **Produit fortement négatif :** alignement de signe opposé.

La norme des vecteurs compte aussi : le produit scalaire n'est pas une mesure du seul angle, et un signe négatif ne prouve pas une opposition de sens. Voir les notes.

### Retour à l'application de rencontres

Reprenons nos profils avec des nombres.

**Vos préférences, la requête :** [Pizza : 10, Mariage : 10, Randonnée : 5].

**L'accroche d'Alex, la clé :** [Pizza : 9, Mariage : 9, Randonnée : 8].

**Compatibilité par produit scalaire :**

$$
\begin{aligned}
\text{score} &= [10, 10, 5] \cdot [9, 9, 8] \\
&= (10)(9) + (10)(9) + (5)(8) \\
&= 90 + 90 + 40 \\
&= 220 \quad \text{← High compatibility!}
\end{aligned}
$$

**L'accroche de Blake, la clé :** [Pizza : 9, Mariage : 1, Randonnée : 8].

**Compatibilité par produit scalaire :**

$$
\begin{aligned}
\text{score} &= [10, 10, 5] \cdot [9, 1, 8] \\
&= (10)(9) + (10)(1) + (5)(8) \\
&= 90 + 10 + 40 \\
&= 140 \quad \text{← Lower! Marriage mismatch hurt the score!}
\end{aligned}
$$

La dimension mariage, 10 × 1 = 10, contribue moins que celle de la pizza, 10 × 9 = 90 ! Les produits combinent ainsi les composantes des deux vecteurs.

**Dans l'attention :** Query·Key donne un score de compatibilité entre la requête d'un token et ce que propose chaque clé.

### Calculer les scores d'attention de notre phrase

**Scores pour « I » :**

Produit scalaire $\mathbf{Q}_I \cdot \mathbf{K}_I$ :

$$
\begin{aligned}
\text{score}(I \to I) &= [-0.008, -0.127, 0.379] \cdot [0.281, 0.268, -0.017] \\
&= (-0.008)(0.281) + (-0.127)(0.268) + (0.379)(-0.017) \\
&= -0.0022 - 0.0340 - 0.0064 \\
&= -0.0426
\end{aligned}
$$

Produit scalaire $\mathbf{Q}_{I} \cdot \mathbf{K}_{love}$ :

$$
\begin{aligned}
\text{score}(I \to \text{love}) &= [-0.008, -0.127, 0.379] \cdot [0.292, 0.259, 0.028] \\
&= -0.0023 - 0.0329 + 0.0106 \\
&= -0.0246
\end{aligned}
$$

Produit scalaire $\mathbf{Q}_{I} \cdot \mathbf{K}_{pizza}$ :

$$
\begin{aligned}
\text{score}(I \to \text{pizza}) &= [-0.008, -0.127, 0.379] \cdot [0.033, -0.529, 0.573] \\
&= -0.0003 + 0.0672 + 0.2172 \\
&= 0.2841
\end{aligned}
$$

### Mise à l'échelle par $\sqrt{d_k}$

**Étape essentielle :** nous divisons par $\sqrt{d_k} = \sqrt{3} = 1.732$.

**Pourquoi ?** Sous les hypothèses usuelles de l'explication, la variance des produits scalaires augmente avec $d_k$. De grands scores peuvent donner un softmax très concentré, comme $[0.001, 0.001, 0.998]$, et ainsi :

- Produire des gradients très faibles.
- Compliquer l'entraînement.
- Limiter l'apprentissage de pondérations plus nuancées.

La formule :

$$
\text{scaled\_score} = \frac{\mathbf{Q} \cdot \mathbf{K}^T}{\sqrt{d_k}}
$$

**Scores mis à l'échelle pour « I » :**

$$
\begin{aligned}
I \to I: & \quad -0.0426 / 1.732 = -0.0246 \\
I \to \text{love}: & \quad -0.0246 / 1.732 = -0.0142 \\
I \to \text{pizza}: & \quad 0.2841 / 1.732 = 0.1640
\end{aligned}
$$

---

## Partie 5 : Convertir les scores en probabilités (softmax)

Nos scores sont des nombres bruts. Il faut les convertir en poids positifs dont la somme vaut 1, soit 100 % !

### L'analogie du partage d'un gâteau

**Trois enfants ont aidé à préparer un gâteau :**

- Alice a beaucoup aidé : score de contribution = 10.
- Bob a un peu aidé : score = 5.
- Charlie a peu aidé : score = 2.

**Contribution totale :** 10 + 5 + 2 = 17.

**Comment répartir le gâteau ?**

```text
Part d'Alice : 10/17 = 59% du gâteau (la plus grande contribution !)
Part de Bob :   5/17 = 29% du gâteau
Part de Charlie : 2/17 = 12% du gâteau

Total : 59% + 29% + 12% = 100% ✓ (tout le gâteau est réparti !)
```

Cela illustre la **normalisation** : convertir des scores en proportions dont la somme vaut 100 %. Softmax ajoute une étape avant cette division.

**Problème avec une division directe : que faire des scores négatifs ?**

```text
Scores d'attention : [-0.02, 0.12, -0.05]

Des parts négatives ne conviennent pas ! Alice ne peut pas recevoir -15% du gâteau !
```

**La solution de softmax : utiliser des exponentielles pour obtenir des valeurs positives !**

### Appliquer softmax (pas à pas)

Trois étapes :

1. **Rendre positif :** appliquer l'exponentielle $e^x$ à chaque score.
2. **Additionner :** calculer la somme des exponentielles.
3. **Normaliser :** diviser chacune par cette somme.

**Formule :**

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

**Décodons la notation.**

**$x_i$ :** la iᵉ valeur de notre liste.

- $x_1$ : premier score.
- $x_2$ : deuxième score.
- $x_3$ : troisième score.

**$e^{x_i}$ :** le nombre d'Euler, 2.718…, élevé à la puissance $x_i$.

- Le résultat est positif.
- $e$ est une constante mathématique particulière, comme π.

**$\sum_{j} e^{x_j}$ :** la somme de toutes les exponentielles.

- $\sum$, sigma majuscule, signifie « additionner ».
- Exemple : $\sum_{i=1}^{3} x_i = x_1 + x_2 + x_3$.

**Un exemple simple de somme :**

Si $x = [10, 20, 30]$ :

$$
\sum_{i=1}^{3} x_i = 10 + 20 + 30 = 60
$$

C'est une notation compacte pour « additionner tous les termes » !

### Exemple de softmax avec les contributions au gâteau

Reprenons le gâteau, cette fois avec les exponentielles.

**Scores bruts :** [10, 5, 2].

**Étape 1 : Appliquer l'exponentielle** pour obtenir des nombres positifs et accentuer les différences de poids.

$$
\begin{aligned}
e^{10} &= 22,026 \\
e^{5} &= 148 \\
e^{2} &= 7.4
\end{aligned}
$$

**Étape 2 : Additionner**

$$
\text{sum} = 22,026 + 148 + 7.4 = 22,181.4
$$

**Étape 3 : Normaliser** en divisant par la somme.

$$
\begin{aligned}
\text{Alice} &= 22,026 / 22,181.4 = 99.3\% \\
\text{Bob} &= 148 / 22,181.4 = 0.67\% \\
\text{Charlie} &= 7.4 / 22,181.4 = 0.03\%
\end{aligned}
$$

**Remarquez le changement :**

- Au départ, la contribution d'Alice est deux fois celle de Bob, 10 contre 5.
- Après softmax, la part d'Alice est environ 148 fois celle de Bob, 99.3 % contre 0.67 % !

Softmax peut ainsi concentrer l'attention sur les scores les plus élevés tout en conservant une contribution des autres.

### L'intuition de softmax

**Dans cette analogie, softmax ressemble à un juge exigeant :**

- Le meilleur score peut recevoir une grande part.
- Les scores intermédiaires peuvent recevoir moins.
- Les plus faibles peuvent recevoir une part infime, strictement positive en arithmétique exacte pour des scores finis.

**Dans l'application de rencontres, la source propose :**

- Alex, score 220 : 85 % d'attention.
- Casey, score 180 : 13 %.
- Blake, score 140 : 2 %.

Ces pourcentages sont illustratifs : ils ne sont pas le softmax exact de [220, 180, 140]. L'écart de scores détermine précisément la concentration, comme le montrent les notes.

### Appliquer softmax aux scores d'attention

**Étape 1 : Calculer les exponentielles**

**Qu'est-ce que $e$ ?** Le nombre d'Euler, $e \approx 2.71828...$, une constante mathématique que l'on retrouve dans de nombreux phénomènes.

**Que fait $e^x$ ?**

- $e^0 = 1$ : toute base non nulle à la puissance zéro vaut 1.
- $e^1 = 2.718$.
- $e^2 = 7.389$.
- $e^{-1} = 0.368$ : un exposant négatif donne ici une petite valeur positive.
- $e^{-10} = 0.000045$ : un nombre très négatif donne une valeur très petite.

Ces valeurs sont arrondies.

**Propriétés utiles à softmax :**

- Positivité : $e^x > 0$, même pour $x$ négatif.
- Conservation de l'ordre : si $a > b$, alors $e^a > e^b$.
- Des écarts de scores deviennent des rapports de poids exponentiels.

Exemple :

- Scores : [1.0, 1.1, 1.2].
- Exponentielles : [2.718, 3.004, 3.320].

Revenons aux calculs :

$$
\begin{aligned}
e^{-0.0246} &= 0.9757 \\
e^{-0.0142} &= 0.9859 \\
e^{0.1640} &= 1.1782
\end{aligned}
$$

**Comment calculer $e^x$ ?** Avec une calculatrice ou une table scientifique. Pour l'intuition :

- $x$ petit et positif : un résultat un peu supérieur à 1.
- $x$ grand et positif : un grand nombre.
- $x$ négatif : un nombre positif entre 0 et 1.

**Étape 2 : Additionner**

$$
\text{sum} = 0.9757 + 0.9859 + 1.1782 = 3.1398
$$

**Étape 3 : Normaliser**

$$
\begin{aligned}
P(I \to I) &= 0.9757 / 3.1398 = 0.3107 \quad (31.07\%) \\
P(I \to \text{love}) &= 0.9859 / 3.1398 = 0.3140 \quad (31.40\%) \\
P(I \to \text{pizza}) &= 1.1782 / 3.1398 = 0.3753 \quad (37.53\%)
\end{aligned}
$$

**Observation :** avec ces nombres, « I » accorde le plus grand poids à « pizza », 37.53 %.

**⚠️ Rappel important :** les poids d'attention de cet exemple, environ 31 %, 31 % et 37 %, proviennent de **matrices choisies pour illustrer une initialisation aléatoire**. Il ne faut pas leur attribuer une signification linguistique apprise ! Certaines valeurs de la source présentent aussi des écarts de calcul détaillés dans les notes.

**Après l'entraînement**, les poids peuvent représenter des relations utiles : un sujet peut exploiter son verbe et son objet, un pronom son antécédent. Les valeurs aléatoires de cet exemple servent à expliquer la procédure, pas à démontrer un comportement appris.

### Somme pondérée des valeurs

Construisons la sortie en combinant les vecteurs de valeur selon les poids d'attention :

$$
\begin{aligned}
\text{Output}_I &= 0.3107 \times \mathbf{V}_I + 0.3140 \times \mathbf{V}_{\text{love}} + 0.3753 \times \mathbf{V}_{\text{pizza}} \\
&= 0.3107 \times [-0.127, 0.211, 0.172] \\
&\quad + 0.3140 \times [-0.091, 0.381, -0.055] \\
&\quad + 0.3753 \times [0.636, 0.036, 0.277] \\
&= [-0.0395, 0.0656, 0.0534] \\
&\quad + [-0.0286, 0.1196, -0.0173] \\
&\quad + [0.2387, 0.0135, 0.1040] \\
&= [0.1706, 0.1987, 0.1401]
\end{aligned}
$$

C'est la **sortie de la tête 1 pour « I » annoncée par la source**, un vecteur à trois dimensions combinant les informations des tokens.

### La tête 2 (version abrégée)

La tête 2 applique la même procédure aux dimensions [3, 4, 5], avec d'autres matrices de poids. Pour alléger le chapitre, la source donne directement le résultat suivant :

$$
\text{Head 2 output for "I"} = [0.245, -0.089, 0.156]
$$

Les matrices de cette seconde tête ne sont pas fournies ici : cette valeur est donc illustrative et ne peut pas être recalculée à partir des seules données du chapitre.

### Concaténation

Réunissons les sorties des deux têtes :

$$
\text{Concatenated output for "I"} = [0.1706, 0.1987, 0.1401, 0.245, -0.089, 0.156]
$$

Nous retrouvons six dimensions, qui combinent maintenant plusieurs points de vue !

### Projection de sortie

**Pourquoi cette étape ?** La concaténation juxtapose les sorties des têtes. La projection de sortie apprend à les mélanger.

Matrice de poids $\mathbf{W}^O$, de taille 6×6, entraînable :

$$
\mathbf{W}^O = \begin{bmatrix}
0.1 & 0.2 & -0.1 & 0.3 & 0.1 & -0.2 \\
0.2 & -0.1 & 0.3 & 0.1 & 0.2 & 0.1 \\
-0.1 & 0.3 & 0.1 & -0.2 & 0.3 & 0.2 \\
0.3 & 0.1 & 0.2 & -0.1 & 0.1 & 0.3 \\
0.1 & -0.2 & 0.3 & 0.2 & -0.1 & 0.1 \\
-0.2 & 0.3 & 0.1 & 0.3 & 0.2 & -0.1
\end{bmatrix}
$$

Sortie finale annoncée par la source comme un résultat simplifié :

$$
\text{Attention output for "I"} = [0.25, -0.31, 0.42, 0.18, -0.09, 0.33]
$$

Ce vecteur ne correspond pas au produit exact du vecteur concaténé et de la matrice affichés. Il est conservé pour suivre la source et le chapitre suivant ; le produit recalculé figure dans les notes de traduction.

**Formule complète de l'attention d'une tête :**

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

Elle résume la procédure : calculer les scores avec Q·Kᵀ, diviser par √dₖ, appliquer softmax, puis former une combinaison pondérée des valeurs par multiplication avec V.

L'attention multi-têtes offre plusieurs points de vue. Une tête pourrait mettre l'accent sur « qui fait quoi », une autre sur les catégories de mots, une troisième sur le ton émotionnel et une quatrième sur les relations temporelles. Ce sont des possibilités d'apprentissage, pas des rôles prédéfinis.

La projection de sortie ressemble au travail d'un réalisateur qui dispose de plusieurs caméras : l'une a filmé l'action, l'autre les visages, une troisième le décor. Il ne juxtapose pas simplement les images : il les combine pour construire une scène cohérente. C'est ce que la projection apprend à faire avec les sorties des têtes !

---

**Navigation dans le cours :** [Précédent : chapitre 4 — Encodage positionnel](chapter-04-positional-encoding.md) | [Suivant : chapitre 6 — Dropout](chapter-06-dropout.md)
