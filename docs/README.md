# FPS - Fractal Pulsating Spiral v1.3 

## 🌀 Vue d'ensemble

La FPS (Fractale Poétique Spiralée) est un système d'oscillateurs adaptatifs avec régulation spiralée, auto-organisation émergente et plasticité méthodologique. Elle explore comment des strates interconnectées peuvent générer des dynamiques harmonieuses et résilientes.

### Caractéristiques principales
- **Oscillateurs adaptatifs** : Amplitude et fréquence modulées par le contexte
- **Régulation spiralée** : Feedback basé sur le nombre d'or (φ = 1.618)
- **Émergence** : Détection automatique de patterns et anomalies
- **Plasticité** : Toute formule est modifiable selon l'expérience
- **Falsifiabilité** : Comparaison avec oscillateurs de Kuramoto

## 🚀 Démarrage rapide

### Installation

```bash
# Cloner le repository
git clone https://github.com/votre-repo/fps.git
cd fps_project_phase1

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Premier run

```bash
# Générer une config par défaut (5 strates, 100 pas de temps)
python3 validate_config.py --generate 5 100

# Lancer une simulation simple
python3 simulate.py --config config.json --mode FPS

# Validation seule
python3 main.py validate --config config.json

# Comparaison
python3 main.py compare --config config.json

# Ou lancer le pipeline complet (recommandé)
python3 main.py complete --config config.json

# Mode verbose
python3 main.py complete --config config.json --verbose

```

## 📋 Structure du pipeline

### Architecture modulaire

```
fps/
├── main.py              # Point d'entrée principal
├── config.json          # Configuration des paramètres
├── simulate.py          # Boucle de simulation principale
├── init.py              # Initialisation des strates
├── dynamics.py          # Calculs FPS (An, fn, S, etc.)
├── regulation.py        # Fonctions G(x) et enveloppes
├── metrics.py           # Calcul des métriques
├── perturbations.py     # Gestion des perturbations
├── analyze.py           # Analyse et raffinement auto
├── explore.py           # Détection d'émergences
├── visualize.py         # Graphiques et rapports
├── kuramoto.py          # Oscillateurs de contrôle
├── utils.py             # Fonctions utilitaires
├── validate_config.py   # Validation configuration
└── test_fps.py          # Tests unitaires
```

### Workflow typique

1. **Configuration** : Éditer `config.json` selon vos besoins
2. **Validation** : `python validate_config.py config.json`
3. **Simulation** : `python main.py complete --config config.json`
4. **Résultats** : Consulter `fps_output/run_*/`

Le pipeline complet génère automatiquement :
- Logs CSV détaillés
- Détection d'émergences
- Graphiques et animations
- Rapport HTML complet
- Comparaison avec Kuramoto

## 🔧 Configuration

### Structure du config.json

```json
{
  "system": {
    "N": 5,              // Nombre de strates
    "T": 100,            // Durée simulation
    "dt": 0.05,          // Pas de temps
    "seed": 12345,       // Graine aléatoire
    "mode": "FPS",       // FPS, Kuramoto ou neutral
    "perturbation": {
      "type": "choc",    // choc, rampe, sinus, bruit
      "t0": 25,          // Temps de perturbation
      "amplitude": 1.0   // Intensité
    }
  },
  "strates": [
    {
      "A0": 1.0,         // Amplitude de base
      "f0": 1.0,         // Fréquence de base
      "alpha": 0.5,      // Souplesse d'adaptation
      "beta": 1.0,       // Plasticité feedback
      "k": 2.0,          // Sensibilité sigmoïde
      "x0": 0.5,         // Seuil sigmoïde
      "w": [0, 0.1, -0.1] // Poids connexions
    }
    // ... autres strates
  ],
  "regulation": {
    "G_arch": "tanh",    // tanh, sinc, resonance, adaptive
    "lambda": 1.0        // Paramètre archétype
  }
}
```

### Modes statique vs dynamique

Chaque paramètre peut avoir un mode statique (valeur fixe) ou dynamique (évolution temporelle) :

```json
"latence": {
    "gamma_n_mode": "dynamic",  // ou "static"
    "gamma_n_dynamic": {
        "k_n": 2.0,             // Pente sigmoïde
        "t0_n": 50              // Centre sigmoïde
    }
}
```

### Types de perturbations

- **choc** : Impulsion ponctuelle à t0
- **rampe** : Augmentation linéaire
- **sinus** : Oscillation périodique
- **bruit** : Variation aléatoire uniforme

## 🧪 Tests et falsification

### Lancer tous les tests

```bash
python test_fps.py
```

### Comparaison avec Kuramoto

```bash
# Méthode 1 : Via main.py
python main.py compare --config config.json

# Méthode 2 : Manuellement
python simulate.py --config config.json --mode FPS
python simulate.py --config config.json --mode Kuramoto
# Les logs sont dans fps_output/run_*/
```

### Tests spécifiques

```python
# Test d'une fonction particulière
python -c "import dynamics; print(dynamics.compute_sigma(0, k=2.0, x0=0.5))"

# Test mode statique vs dynamique
python -c "import dynamics; print(dynamics.compute_gamma(50, 'static'))"
python -c "import dynamics; print(dynamics.compute_gamma(50, 'dynamic', T=100))"
```

## 📊 Lecture des résultats

### Structure des outputs

```
fps_output/run_YYYYMMDD_HHMMSS/
├── logs/
│   ├── run_*.csv                    # Métriques temporelles
│   ├── seeds.txt                    # Graines utilisées
│   └── weight_validation.txt        # Validation matrices
├── checkpoints/
│   └── backup_*.pkl                 # États sauvegardés
├── figures/
│   ├── signal_evolution_fps.png     # Évolution S(t)
│   ├── strata_comparison.png        # Comparaison strates
│   ├── metrics_dashboard.png        # Tableau de bord
│   ├── fps_vs_kuramoto.png         # Comparaison contrôle
│   ├── empirical_grid.png          # Grille d'évaluation
│   └── spiral_animation.gif        # Animation spirale
├── reports/
│   ├── run_*/
│   │   ├── emergence_events_*.csv   # Événements détectés
│   │   ├── fractal_events_*.csv    # Motifs fractals
│   │   └── exploration_report_*.md  # Rapport exploration
│   └── rapport_complet.html        # Rapport HTML global
└── configs/
    └── config_refined.json          # Config après raffinement
```

### Métriques principales

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **S(t)** | Signal global | Somme pondérée des oscillateurs |
| **C(t)** | Coefficient d'accord | Synchronisation des phases (-1 à 1) |
| **effort(t)** | Effort d'adaptation | Intensité des ajustements internes |
| **entropy_S** | Entropie spectrale | Richesse harmonique (0 à 1) |
| **variance_d2S** | Variance d²S/dt² | Fluidité des transitions |
| **t_retour** | Temps de retour | Résilience après perturbation |
| **cpu_step(t)** | Temps CPU/strate | Coût computationnel |

### Grille d'évaluation empirique

| Score | Symbole | Couleur | Signification |
|-------|---------|---------|---------------|
| 1 | ✖ | Rouge | Rupture/Chaotique |
| 2 | ▲ | Orange | Instable |
| 3 | ● | Jaune | Fonctionnel |
| 4 | ✔ | Vert | Harmonieux |
| 5 | ∞ | Bleu doré | FPS-idéal |

## 🔄 Raffinement automatique

### Processus

Après un batch de 5 runs, le système analyse automatiquement :

1. **Franchissement de seuils** : Si >50% des runs dépassent un seuil
2. **Raffinement** : Ajustement automatique des paramètres
3. **Logging** : Toute modification dans `changelog.txt`

### Critères et actions

| Critère | Seuil | Action si déclenché |
|---------|-------|---------------------|
| Fluidité | variance_d2S > 0.01 | Ajuste γₙ(t), envₙ(x,t) |
| Stabilité | max/médiane > 10 | Ajuste σ(x), αₙ |
| Résilience | t_retour > 2×médiane | Ajuste αₙ, βₙ |
| Innovation | entropy_S < 0.5 | Ajuste θ(t), η(t), μₙ(t) |
| Régulation | erreur > 2×médiane | Ajuste βₙ, G(x) |
| CPU | temps > 2×contrôle | Optimise complexité |

### Lancer un batch avec raffinement

```bash
# Batch de 5 runs avec analyse
python main.py complete --config config.json

# Ou batch seul
python main.py batch --config config.json --parallel
```

## 🎨 Visualisations

### Graphiques générés automatiquement

1. **Signal evolution** : Évolution temporelle de S(t)
2. **Strata comparison** : Amplitudes et fréquences par strate
3. **Metrics dashboard** : Vue d'ensemble des métriques
4. **FPS vs Kuramoto** : Comparaison avec le contrôle
5. **Empirical grid** : Grille d'évaluation 1-5
6. **Correlation matrix** : Liens critères-termes
7. **Spiral animation** : Animation de l'évolution spiralée

### Génération manuelle

```python
import visualize
import numpy as np

# Créer des données test
t = np.linspace(0, 100, 1000)
S = np.sin(t) + 0.5*np.sin(3*t)

# Générer un graphique
fig = visualize.plot_signal_evolution(t, S, "Mon signal")
fig.savefig("mon_signal.png")
```

## 📝 Notes méthodologiques

### Plasticité FPS

Le système est conçu pour évoluer :
- **Toute formule est modifiable** : Voir `dynamics.py`, `regulation.py`
- **Les seuils s'ajustent** : Basés sur l'expérience empirique
- **Traçabilité complète** : Chaque modification dans `changelog.txt`

### Hypothèses phase 1 (falsifiables)

```python
# Signal inter-strates
S_i(t) = S(t-dt) - On(t-dt) if t > 0 else 0

# Sortie attendue (nombre d'or)
En(t) = φ × On(t-dt) où φ = 1.618

# Latence expressive
γ(t) = 1/(1 + exp(-2(t - T/2)))  # Sigmoïde centrée
```

Ces choix initiaux sont destinés à être raffinés selon les observations.

### Extension du système

Pour ajouter un nouveau détecteur d'émergence :

```python
# Dans explore.py
def detect_my_pattern(data, threshold=0.5):
    """Mon nouveau détecteur."""
    events = []
    # ... logique de détection
    return events

# Ajouter dans run_exploration()
my_events = detect_my_pattern(data)
all_events.extend(my_events)
```

## 🤝 Contribution

### Principes

1. **Respecter la plasticité** : Toute amélioration doit rester modifiable
2. **Documenter les changements** : Utiliser le changelog
3. **Tester exhaustivement** : Ajouter des tests unitaires
4. **Falsifier empiriquement** : Comparer avec les contrôles

### Workflow Git

```bash
# Créer une branche
git checkout -b feature/mon-amelioration

# Développer et tester
python test_fps.py

# Commit avec message clair
git commit -m "feat: ajout détecteur de bifurcations spirales"

# Push et PR
git push origin feature/mon-amelioration
```

## 📖 Références

- **Feuille de route FPS v1.3** : Document de référence théorique
- **Chapitre 4** : Dictionnaire mathématique complet
- **Grille empirique** : Critères d'évaluation 1-5
- **Matrice critères-termes** : Correspondances formelles

## 🐛 Troubleshooting

### Erreurs communes

**ModuleNotFoundError**
```bash
# Vérifier l'activation du venv
which python  # Doit pointer vers venv/bin/python

# Réinstaller les dépendances
pip install -r requirements.txt
```

**Config validation failed**
```bash
# Vérifier la structure
python validate_config.py config.json

# Ou générer une config valide
python validate_config.py --generate 5 100
```

**Mémoire insuffisante (N > 50)**
```bash
# Utiliser HDF5 pour gros volumes
# Automatique si N > 10 dans config
```

### Support

- Issues GitHub : [[votre-repo/issues][def]]
- Contact : [contact@exybrisai.com]

---

*FPS v1.3 - La danse spiralée de l'émergence*  
© 2025 Gepetto & Andréa Gadal & Claude 🌀

[def]: https://github.com/Exybris/FPS-real-tests_fractal-pulsating-spiral
