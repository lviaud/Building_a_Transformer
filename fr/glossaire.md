# Glossaire

Les termes suivants servent de référence pour garder une traduction cohérente entre les chapitres.

| Terme anglais | Traduction ou usage retenu |
| --- | --- |
| Transformer | Transformer ; nom de l'architecture |
| embedding | embedding ; représentation vectorielle apprise |
| word embedding | embedding de mot |
| positional encoding | encodage positionnel |
| learned positional embedding | embedding positionnel appris |
| token | token ; unité produite par la tokenisation |
| tokenization | tokenisation |
| vocabulary | vocabulaire ; ensemble des tokens disponibles |
| token ID | identifiant de token ; indice d'une entrée du vocabulaire |
| subword | sous-mot |
| byte | octet |
| byte-pair encoding (BPE) | BPE ; méthode fondée sur la fusion de paires fréquentes |
| merge | fusion, dans l'algorithme BPE |
| lookup table | table de correspondance ; accès à une entrée par son identifiant |
| embedding matrix | matrice d'embeddings |
| vector | vecteur |
| matrix | matrice |
| feature | caractéristique, propriété |
| vocab_size | taille du vocabulaire |
| d_model | nombre de dimensions des représentations du modèle |
| attention | attention |
| self-attention | auto-attention |
| multi-head self-attention | auto-attention multi-têtes |
| attention head | tête d'attention |
| query (Q) | requête ; vecteur comparé aux clés |
| key (K) | clé ; vecteur utilisé pour calculer un score avec une requête |
| value (V) | valeur ; vecteur d'information entrant dans la somme pondérée |
| attention score | score d'attention, avant normalisation |
| attention weight | poids d'attention, après softmax |
| weight matrix | matrice de poids ; paramètres entraînables |
| dot product | produit scalaire |
| scaled dot-product attention | attention par produit scalaire mis à l'échelle |
| softmax | softmax ; normalisation exponentielle en poids positifs de somme 1 |
| weighted sum | somme pondérée |
| concatenation | concaténation ; juxtaposition des composantes des vecteurs |
| output projection | projection de sortie |
| causal masking | masquage causal |
| dropout | dropout ; mise à zéro aléatoire d'activations à l'entraînement |
| dropout rate | taux de dropout ; probabilité de suppression |
| mask | masque ; ici, indicateurs de conservation ou de suppression |
| overfitting | surapprentissage |
| regularization | régularisation |
| expected value | espérance |
| residual connection | connexion résiduelle |
| feed-forward network | réseau feed-forward ; réseau appliqué à chaque position |
| training | entraînement |
| inference | inférence |
| backpropagation | rétropropagation |
| learning rate | taux d'apprentissage |
| gradient | gradient |
| dimension pair | paire de dimensions |
| frequency | fréquence |
| phase shift | déphasage |
| bounded | borné |
| element-wise addition | addition composante par composante |
| frozen | fixe, non entraînable |
| fingerprint | empreinte |

Dans les blocs de pseudo-code et les formules, certains noms symboliques comme `WordMeaning`, `PositionNumber`, `learning_rate`, `Output` ou `score` restent inchangés. Les phrases anglaises associées à des vecteurs sont également conservées.


