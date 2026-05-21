# Segmentation de lignes manuscrites avec U-Net

TP — Computer Vision | Segmentation sémantique sur le corpus RIMES

---

## Présentation

Ce projet implémente un réseau **U-Net** pour la segmentation de lignes manuscrites sur le corpus RIMES (lettres, questionnaires, fax numérisés). Le modèle prend en entrée une image en niveaux de gris et prédit un masque binaire indiquant les zones correspondant à des lignes de texte.

---

## Structure du projet

```
.
├── data/
│   ├── rimes_raw/
│   │   └── DVD1/          # Images .tif + annotations .xml (voir §1.1)
│   └── rimes_masks/       # Masques PNG générés automatiquement
├── dataset.py             # Chargement, augmentation, pipeline tf.data
├── model.py               # Architecture U-Net
├── train.py               # Boucle d'entraînement, IoUMetric, callbacks
├── evaluate.py            # Évaluation quantitative et visualisation
├── utils.py               # Génération des masques depuis XML, fonctions de perte
├── main.py                # Point d'entrée principal
└── requirements.txt
```

---

## 1. Installation

### 1.1 Données — Corpus RIMES

Le corpus RIMES est disponible sur Zenodo :
**[zenodo.org/records/10812725](https://zenodo.org/records/10812725)**

Téléchargez l'archive `Images_Courriers.zip`. Elle contient trois dossiers (`DVD1`, `DVD2`, `DVD3`) représentant chacun un sous-ensemble du corpus. Pour ce TP, on travaille sur un seul dossier, par exemple `DVD1`.

Décompressez-le dans `data/rimes_raw/DVD1/` :

```bash
unzip Images_Courriers.zip
mkdir -p data/rimes_raw
mv DVD1 data/rimes_raw/DVD1
```

Chaque page est représentée par une paire de fichiers à plat dans ce dossier : une image TIFF et un fichier XML du même nom. Par exemple `1_L.tif` (la lettre n°1) et `1_L.xml` (ses annotations). Les suffixes `L`, `Q` et `F` désignent respectivement les lettres, questionnaires et fax.

### 1.2 Dépendances Python

```bash
pip install -r requirements.txt
```

Dépendances principales : `tensorflow`, `albumentations`, `pillow`, `lxml`, `scikit-learn`, `matplotlib`.

---

## 2. Utilisation

### Pipeline complet (recommandé)

```bash
python main.py
```

Ce script enchaîne automatiquement :
1. La génération des masques depuis les XML RIMES
2. La construction et l'affichage du résumé du modèle
3. L'entraînement avec sauvegarde du meilleur modèle
4. L'évaluation sur le jeu de validation
5. La visualisation d'une prédiction

### Étapes individuelles

**Générer les masques :**
```bash
python utils.py
```

**Entraîner le modèle :**
```bash
python train.py
```

**Évaluer un modèle sauvegardé :**
```bash
python evaluate.py
```

---

## 3. Architecture — U-Net

Le modèle suit l'architecture U-Net classique avec des skip connections entre l'encodeur et le décodeur.

```
Entrée (512×512×1)
     │
  Encodeur (×4 blocs)         64 → 128 → 256 → 512 filtres
     │ skip connections
  Bottleneck                   1024 filtres
     │ skip connections
  Décodeur (×4 blocs)         512 → 256 → 128 → 64 filtres
     │
  Sortie (512×512×1) — sigmoid
```

Chaque bloc de convolution double applique : `Conv2D → BatchNorm → ReLU` (×2).

---

## 4. Entraînement

| Paramètre | Valeur |
|---|---|
| Taille des patches | 512 × 512 |
| Batch size | 8 |
| Époques | 30 |
| Optimiseur | AdamW |
| Learning rate | CosineDecay depuis 1e-3 |
| Weight decay | 1e-4 |
| Perte | BCE + Dice Loss (α = 0.5) |
| Métrique | IoU (seuil 0.5) |

### Augmentations (entraînement)

- `RandomCrop` 512×512
- `HorizontalFlip` (p=0.2)
- `RandomBrightnessContrast` (p=0.4)
- `GaussNoise` (p=0.3)
- `ElasticTransform` α=34, σ=4 (p=0.5)

La validation n'applique qu'un `CenterCrop` 512×512.

### Callbacks

- **ModelCheckpoint** : sauvegarde `best_unet.keras` si `val_loss` s'améliore
- **TensorBoard** : logs dans `logs/`

Visualiser les courbes d'entraînement :
```bash
tensorboard --logdir logs/
```

---

## 5. Évaluation

Les métriques calculées sur le jeu de validation sont :

- **Pixel Accuracy** : proportion de pixels correctement classés
- **IoU** (Intersection over Union) : chevauchement entre la prédiction et la vérité terrain

La fonction `visualize_prediction` affiche côte à côte l'image d'entrée, le masque de référence et la prédiction binaire.

---

## 6. Génération des masques

Les masques sont générés automatiquement à partir des annotations XML du corpus RIMES via `utils.xml_to_mask_rimes`. Pour chaque bloc de texte annoté, le nombre de lignes est estimé à partir du contenu textuel (`\n`), puis chaque ligne reçoit une dilatation verticale de 10 % pour couvrir les hampes et les jambages. Les blocs de hauteur inférieure à 20 pixels sont ignorés.

---

## Licence

Voir le fichier `LICENSE`.