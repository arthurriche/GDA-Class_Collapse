# GDA-Class_Collapse - Étude du Collapse de Classes avec Apprentissage Contrastif

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

Ce projet étudie le phénomène de "class collapse" dans l'apprentissage contrastif, un problème où les représentations apprises par un modèle s'effondrent en un seul point ou cluster, perdant ainsi leur capacité discriminative. L'étude utilise des techniques d'apprentissage contrastif avec différents paramètres alpha pour analyser et visualiser ce comportement.

## 🚀 Key Features

### Apprentissage Contrastif
- **Contrastive Loss with Alpha** - Fonction de perte contrastive paramétrable
- **Temperature Scaling** - Contrôle de la température pour la similarité
- **Normalization** - Normalisation L2 des représentations
- **Positive/Negative Pairs** - Gestion des paires positives et négatives

### Étude du Class Collapse
- **Paramètre Alpha** - Contrôle du degré de collapse (0.0 à 1.0)
- **Visualisation Interactive** - Comparaison des données originales vs apprises
- **Analyse Comparative** - Étude de l'impact de différents paramètres
- **Métriques de Collapse** - Mesures quantitatives du phénomène

### Datasets et Expérimentations
- **Dataset Synthétique** - Génération de données multi-classes
- **House Dataset** - Dataset spécialisé pour l'étude
- **Stratification** - Données stratifiées par classes et strates
- **Reproductibilité** - Seeds fixes pour la reproductibilité

## 📁 Project Structure

```
GDA-Class_Collapse/
├── script.py                    # Script principal d'expérimentation
├── train.py                     # Script d'entraînement modulaire
├── dataset.py                   # Génération et gestion des datasets
├── losses.py                    # Implémentation des fonctions de perte
├── visualization.py             # Utilitaires de visualisation
├── outputs/                     # Résultats et visualisations
│   ├── original_data_alpha_*.png    # Données originales par alpha
│   └── learned_representations_alpha_*.png  # Représentations apprises
└── README.md                    # Documentation du projet
```

## 🛠️ Installation

### Prérequis
- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Scikit-learn (optionnel)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/arthurriche/GDA-Class_Collapse.git
cd GDA-Class_Collapse

# Installer les dépendances
pip install torch torchvision
pip install numpy
pip install matplotlib
pip install scikit-learn
```

## 📈 Quick Start

### 1. Exécution Rapide

```bash
# Lancer l'expérimentation complète
python script.py

# Ou utiliser le script modulaire
python train.py
```

### 2. Expérimentation Interactive

```python
from script import train_model, create_dataset
import matplotlib.pyplot as plt

# Créer un dataset
data, labels = create_dataset(num_classes=3, num_strata=2, num_samples=100)

# Entraîner avec différents alphas
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
for alpha in alphas:
    model, data, labels = train_model(alpha=alpha, num_epochs=100)
    # Visualiser les résultats...
```

### 3. Analyse des Résultats

```python
# Les visualisations sont automatiquement sauvegardées dans outputs/
# Comparer les différents alphas pour observer le class collapse
```

## 🧮 Technical Implementation

### Contrastive Loss with Alpha

```python
class ContrastiveLossWithAlpha(nn.Module):
    def __init__(self, temperature=0.5, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, z_i, z_j):
        # Normalisation L2
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        
        # Calcul de la matrice de similarité
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Gestion des paires positives/négatives
        # ... logique de calcul de la perte
```

### Architecture du Modèle

```python
model = nn.Sequential(
    nn.Linear(2, 128),    # Couche d'entrée
    nn.ReLU(),            # Activation
    nn.Linear(128, 64)    # Couche de sortie (représentations)
)
```

### Génération de Données

```python
def create_dataset(num_classes=3, num_strata=2, num_samples=100):
    data = []
    labels = []
    for class_id in range(num_classes):
        for stratum_id in range(num_strata):
            mean = np.random.rand(2) * (class_id + 1)
            points = mean + 0.1 * np.random.randn(num_samples, 2)
            data.append(points)
            labels += [class_id] * num_samples
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels)
```

## 📊 Performance Analysis

### Impact du Paramètre Alpha

| Alpha | Comportement | Collapse | Séparabilité |
|-------|-------------|----------|--------------|
| 0.0   | Pas de collapse | Faible | Élevée |
| 0.25  | Collapse partiel | Modéré | Modérée |
| 0.5   | Collapse équilibré | Moyen | Moyenne |
| 0.75  | Collapse avancé | Élevé | Faible |
| 1.0   | Collapse total | Maximum | Minimale |

### Métriques d'Évaluation

- **Intra-class Variance** - Variance au sein des classes
- **Inter-class Distance** - Distance entre les classes
- **Collapse Ratio** - Ratio de collapse des représentations
- **Separability Score** - Score de séparabilité des classes

## 🔬 Advanced Features

### Analyse du Class Collapse

Le class collapse se produit quand :
- Les représentations d'une classe se concentrent en un point
- La variance intra-classe devient très faible
- Les classes perdent leur séparabilité

### Contrôle via Alpha

Le paramètre alpha contrôle :
- **Intensité du collapse** - Plus alpha est élevé, plus le collapse est prononcé
- **Balance exploration/exploitation** - Équilibre entre diversité et concentration
- **Robustesse des représentations** - Capacité de généralisation

### Visualisation Avancée

- **Scatter Plots** - Visualisation 2D des représentations
- **Heatmaps** - Matrices de similarité
- **Trajectories** - Évolution des représentations pendant l'entraînement
- **Distribution Analysis** - Analyse des distributions de similarité

## 🚀 Applications

### Recherche en Deep Learning
- **Étude des phénomènes d'effondrement** - Compréhension des mécanismes
- **Optimisation des fonctions de perte** - Amélioration des performances
- **Robustesse des modèles** - Évaluation de la stabilité

### Applications Pratiques
- **Computer Vision** - Classification d'images robuste
- **NLP** - Représentations de texte stables
- **Recommendation Systems** - Embeddings robustes

## 📚 Documentation Technique

### Mécanismes du Class Collapse

1. **Convergence Excessive** - Les représentations convergent trop fortement
2. **Perte de Diversité** - Réduction de la variance intra-classe
3. **Effondrement Structurel** - Perte de la structure discriminative

### Stratégies de Mitigation

- **Regularization** - Techniques de régularisation
- **Temperature Scaling** - Ajustement de la température
- **Data Augmentation** - Augmentation des données
- **Architecture Modifications** - Modifications architecturales

### Hyperparamètres

- **Learning Rate** - 0.001 (Adam optimizer)
- **Temperature** - 0.5 (contrôle de la similarité)
- **Hidden Size** - 128 (taille des couches cachées)
- **Output Size** - 64 (dimension des représentations)

## 🤝 Contributing

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👨‍💻 Author

**Arthur Riche**
- LinkedIn: [Arthur Riche](https://www.linkedin.com/in/arthurriche/)
- Email: arthur.riche@example.com

## 🙏 Acknowledgments

- **Équipe de Recherche** pour la supervision académique
- **Communauté PyTorch** pour les outils de deep learning
- **Contributeurs Open Source** pour les bibliothèques utilisées
- **Pairs de Recherche** pour les discussions et feedback

---

⭐ **Star ce repository si vous le trouvez utile !**

*Ce projet fournit une étude approfondie du phénomène de class collapse dans l'apprentissage contrastif, avec des outils de visualisation et d'analyse pour comprendre ce comportement critique.* 