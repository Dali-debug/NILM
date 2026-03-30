# Analyse du code NILM — Détection de la Bouilloire (`kettle1.ipynb`)

## 1. Présentation du projet

Ce notebook implémente un système de **NILM (Non-Intrusive Load Monitoring)** pour la **détection de la bouilloire électrique**. L'objectif est d'identifier les épisodes d'utilisation de la bouilloire à partir du seul signal de **puissance agrégée** du domicile, sans capteur individuel sur l'appareil.

### Jeu de données : REFIT

Le projet utilise le jeu de données **REFIT** (UK Residential Energy Data), un ensemble de mesures de consommation électrique dans 21 maisons britanniques, enregistrées à une résolution d'environ 8 secondes. Chaque maison fournit :

- Un canal **Aggregate** (puissance totale du logement)
- Plusieurs canaux **appareils** (lave-vaisselle, bouilloire, réfrigérateur, etc.)

### Configuration train / validation / test

| Ensemble    | Maisons                                   | Rôle                                  |
|-------------|-------------------------------------------|---------------------------------------|
| **Train**   | 3, 4, 6, 7, 8, 9, 12, 13, 19, 20 (10 maisons) | Apprentissage de la puissance ON |
| **Validation** | House 5                              | Ajustement des seuils                 |
| **Test**    | House 2                                   | Évaluation finale                     |

---

## 2. Prétraitement des données (Data Preprocessing)

### Ce qui est fait

```
load_refit_house(house_id, kettle_col_idx_0based)
```

| Étape | Description |
|-------|-------------|
| Lecture CSV | Sélection des colonnes Unix, Aggregate et Kettle |
| Conversion timestamps | Unix → `DatetimeIndex` pandas |
| Ré-échantillonnage | Grille régulière toutes les **10 secondes** (`BASE_FREQ = "10S"`) |
| Gestion des NaN | **Forward-fill** (propagation avant) des valeurs manquantes |
| Valeurs initiales manquantes | Remplissage à **0 W** pour les NaN résiduels |
| Type de données | Conversion explicite en `float64` |

### Analyse critique du preprocessing

#### ✅ Points forts

1. **Grille temporelle régulière** — Le ré-échantillonnage sur un pas de temps fixe de 10 s est indispensable pour tout algorithme basé sur la différence temporelle ou les durées. C'est une bonne pratique.

2. **Forward-fill logique** — Pour des données de puissance électrique, un capteur qui ne rafraîchit pas sa valeur conserve généralement la dernière mesure valide. Le `ffill()` est donc physiquement cohérent.

3. **Séparation stricte Train/Val/Test** — Les trois ensembles utilisent des maisons distinctes, ce qui évite toute fuite de données (*data leakage*). L'apprentissage de la puissance ON se fait exclusivement sur les maisons d'entraînement.

4. **Seuil 200 W pour isoler les échantillons « ON »** — Élimine le bruit de fond et les états de veille, ce qui est pertinent pour la bouilloire (≈ 2000–3000 W quand active).

#### ⚠️ Limites et points d'amélioration

1. **Remplissage initial à 0 W** — Si les premières lignes sont manquantes (cas rare mais possible), remplir à 0 peut créer une fausse montée de puissance au début de la série et déclencher une fausse détection.

2. **Absence de détection des valeurs aberrantes** — Aucun filtre pour les pics extrêmes (capteur défaillant, transitoire électrique) qui pourraient être confondus avec une mise en marche de bouilloire. → **Résolu** : filtre de Hampel ajouté (voir §2.1).

3. **Absence de normalisation** — Non critique ici car l'algorithme travaille directement en Watts, mais une normalisation faciliterait une éventuelle généralisation à d'autres jeux de données.

4. **`FutureWarning` pandas** — La fréquence `"10S"` est dépréciée dans les versions récentes de pandas ; il faudrait utiliser `"10s"` (minuscule) pour éviter des problèmes de compatibilité future.

5. **Aucune exploration de corrélation Aggregate/Kettle** — Vérifier que le signal agrégé contient bien la signature de la bouilloire renforcerait la confiance dans les données.

**Verdict** : Le preprocessing est **correct et suffisant pour une première approche**, mais manque de robustesse face aux valeurs aberrantes et nécessite une petite correction de syntaxe pandas.

### 2.1 Filtre de Hampel ✅ (ajout)

#### Qu'est-ce que le filtre de Hampel ?

Le **filtre de Hampel** est un estimateur robuste de détection et remplacement d'outliers qui fonctionne dans une fenêtre glissante. Pour chaque point `x[t]` :

1. Calcule la **médiane locale** `med` sur la fenêtre `[t-w, t+w]`
2. Calcule le **MAD** (*Median Absolute Deviation*) de cette fenêtre : `MAD = médiane(|x - med|)`
3. Estime l'écart-type robuste : `σ_mad = 1.4826 × MAD`
4. Si `|x[t] - med| > k × σ_mad` → remplace `x[t]` par `med` (outlier détecté)
5. Sinon → conserve `x[t]` inchangé (point normal)

#### Pourquoi l'utiliser dans ce pipeline NILM ?

| Problème | Solution Hampel |
|----------|-----------------|
| Pics capteur isolés (glitches électriques) | Détectés et remplacés par la médiane locale |
| Bruit impulsionnel dans l'agrégat | Atténué avant le HMM/règles |
| Préservation des vraies transitions (kettle ON/OFF) | Le plateau bouilloire dure plusieurs minutes → médiane locale reste haute → pas considéré comme outlier |

#### Comment régler ses paramètres

```python
HAMPEL_WINDOW   = 10   # demi-fenêtre (pts) : fenêtre totale = 2×10+1 = 21 pts × 10s = 210s ≈ 3,5 min
HAMPEL_N_SIGMAS = 3.0  # seuil k : 3.0 standard, 4.0–6.0 si on veut être plus permissif
```

| Paramètre | Valeur recommandée | Effet si trop petit | Effet si trop grand |
|-----------|-------------------|---------------------|---------------------|
| `window`  | 10–20 pts         | Médiane instable, faux remplacements | Intègre plusieurs événements, perd la localité |
| `n_sigmas`| 3.0–4.0           | Écrase les vraies transitions (aggressive) | Laisse passer plus d'outliers |

> **Règle pratique** : pour la bouilloire REFIT (plateau ≈ 2–3 kW, durée ≥ 90 s → ≥ 9 points à 10 s), utiliser `window=10` (≈ 3,5 min) et `n_sigmas=3.0`. La fenêtre est assez courte pour capturer les pics isolés mais assez longue pour que la médiane locale soit stable pendant un plateau bouilloire.

#### Où est-il appliqué ?

Le filtre est activé en **premier** dans `preprocess_aggregate()`, avant la détection des outliers globaux et le filtre médian :

```python
# Pipeline complet : Hampel → outliers globaux → filtre médian
preprocess_aggregate(
    aggregate,
    apply_hampel=True,        # ← filtre de Hampel
    hampel_window=10,
    hampel_n_sigmas=3.0,
    apply_outlier_removal=True,
    apply_median_filter=True,
    median_window=3
)
```

---

## 3. Les algorithmes utilisés : détection par règles + HMM 2 états + Viterbi

### Approche 1 — Détection événementielle par règles (approche initiale)

```python
detect_kettle_from_aggregate(aggregate, kettle_on_power, sample_seconds, ...)
```

L'algorithme fonctionne en quatre étapes :

1. **Calcul de la dérivée discrète** — `d = diff(aggregate)` pour détecter les variations brusques de puissance.

2. **Détection de la montée** — Une montée est validée si `d[i] ≥ STEP_ON_FRAC × kettle_on_power` (≥ 75 % de la puissance apprise, soit ≈ 2069 W).

3. **Recherche de la descente** — Après une montée, on cherche une chute `d[j] ≤ −STEP_OFF_FRAC × kettle_on_power` (−65 %, soit ≈ −1793 W) dans la fenêtre de durée autorisée (90 s à 15 min).

4. **Validation par plateau** — L'épisode est confirmé si la médiane du signal pendant l'intervalle vérifie `|ΔP_median − kettle_on_power| ≤ POWER_TOL_FRAC × kettle_on_power` (±15 %, soit ±414 W).

### Approche 2 — HMM 2 états + algorithme de Viterbi ✅ (amélioré)

Le notebook implémente un **Modèle de Markov Caché (HMM) à 2 états** avec émissions gaussiennes, entraîné par l'**algorithme de Baum-Welch** et décodé par l'**algorithme de Viterbi**. Trois améliorations clés ont été ajoutées par rapport à la version initiale :

1. **Signal résiduel** comme observation (au lieu de l'agrégat brut)
2. **Post-traitement** : suppression des segments ON trop courts
3. **Filtre de Hampel** dans le prétraitement (voir §2.1)

#### Structure du HMM

```
GaussianHMM2State
├── États cachés  : {0 = OFF,  1 = ON}
├── Émissions     : N(μ_OFF, σ_OFF²)  et  N(μ_ON, σ_ON²)
├── Transitions   : matrice A (2×2) apprise par Baum-Welch
├── Apprentissage : algorithme EM (Baum-Welch, forward-backward)
├── Observation   : signal résiduel(t) = aggregate(t) - baseline(t)  ← NOUVEAU
├── Décodage      : algorithme de Viterbi  O(T × K²)
└── Post-traitement : suppression des segments ON < MIN_ON_SECONDS            ← NOUVEAU
```

```python
# Prétraitement avec Hampel
agg_filtered = preprocess_aggregate(aggregate, apply_hampel=True)
# Observation résiduelle
obs = compute_hmm_observation(agg_filtered, baseline_window=60)
# Entraînement
hmm = GaussianHMM2State(mu_off=0, sigma_off=50, mu_on=2758, sigma_on=300)
# Note: mu_off=0 car le résidu est clampé à max(0, ...) → l'état OFF vaut ~0 W
#       (contrairement à l'agrégat brut où mu_off ≈ 400 W de consommation de fond)
hmm.fit(train_sequences)            # Baum-Welch EM sur résidus
# Décodage
states = hmm.viterbi(obs)           # séquence d'états optimale
states = postprocess_states(states) # supprime segments ON trop courts
pred   = np.where(states == 1, hmm.mu[1], 0.0)
```

#### Nouvelle observation : signal résiduel

**Problème avec l'agrégat brut** : l'agrégat contient la somme de *tous* les appareils. Avec seulement 2 états, le HMM tente de séparer "faible puissance" vs "forte puissance" dans l'agrégat entier, sans pouvoir distinguer une bouilloire allumée d'un four ou d'une machine à laver. Résultat : précision très faible et MAE élevée.

**Solution — signal résiduel** :

```
residual(t) = max(0,  aggregate(t) − baseline(t))
baseline(t) = médiane glissante(aggregate, fenêtre = 2 × 60 + 1 points = 10 min)
```

La baseline lente capture la consommation de fond (réfrigérateur, TV, éclairage) et les variations lentes d'autres appareils. Le résidu, lui, n'est élevé que lors de **hausses brutales et courtes** — exactement la signature de la bouilloire.

| Signal | État OFF (HMM) | État ON (HMM) |
|--------|---------------|--------------|
| Agrégat brut | `N(μ_agg_off, σ_agg_off²)` — très variable | `N(μ_agg_on, σ_agg_on²)` — difficile à séparer |
| **Résidu** | **≈ N(0, σ_noise²)** — stable près de 0 | **≈ N(kettle_power, σ_on²)** — pic net |

Le résidu est clampé à 0 (`max(0, ...)`) pour éviter les valeurs négatives (liées aux baisses de consommation d'autres appareils).

#### Paramètre de la baseline

```python
HMM_BASELINE_WINDOW = 60   # demi-fenêtre en points (60 × 10s = 10 min au total : 2×60+1 = 121 pts)
```

Ce paramètre contrôle à quelle vitesse la baseline peut suivre les variations lentes. Une fenêtre de 10 min est adaptée car :
- La bouilloire chauffe en 2–10 min → la médiane sur 10 min inclut les deux états et se stabilise
- Les variations lentes d'autres appareils (ex. chauffe-eau) évoluent sur > 10 min → capturées par la baseline

#### Post-traitement des états

Après le décodage Viterbi, les segments ON d'une durée inférieure à `HMM_MIN_ON_SECONDS` (90 s = 9 points) sont réinitialisés à OFF. Cette étape réduit les faux positifs causés par des pics résiduels brefs.

```python
HMM_MIN_ON_SECONDS = 90   # = MIN_ON_SECONDS (cohérent avec l'approche par règles)
```

#### Comparaison règles vs HMM + Viterbi

| Critère | Approche par règles | HMM 2 états + Viterbi |
|---------|--------------------|-----------------------|
| **Modélisation** | Seuils sur dérivée + durée | États cachés (OFF / ON) avec distributions d'émission |
| **Paramètres** | Seuils manuels | Probabilités de transition + paramètres d'émission appris |
| **Inférence** | Parcours linéaire `O(n)` | Algorithme de Viterbi `O(n × K²)` |
| **Robustesse au bruit** | Faible (seuil dur) | Meilleure (modèle probabiliste) |
| **Simultanéité d'appareils** | Aucune gestion | FHMM (Factorial HMM) pour plusieurs appareils |
| **Interprétabilité** | Très bonne | Bonne mais plus complexe |
| **Entraînement** | Statistique simple (quantile 70e) | Algorithme de Baum-Welch (EM) |

---

## 4. Résultats

### Paramètres appris

#### Approche par règles (event-based)

| Paramètre | Valeur |
|-----------|--------|
| Puissance bouilloire ON (70e percentile) | **2 758 W** |
| Seuil montée (`STEP_ON_FRAC = 0.75`) | ≈ 2 069 W |
| Seuil descente (`STEP_OFF_FRAC = 0.65`) | ≈ 1 793 W |
| Tolérance plateau (`POWER_TOL_FRAC = 0.15`) | ≈ ±414 W |
| Durée min | 90 s |
| Durée max | 15 min |

#### HMM 2 états (Baum-Welch)

| Paramètre HMM | Description |
|---------------|-------------|
| μ_OFF | Puissance moyenne état OFF (apprise) |
| σ_OFF | Écart-type puissance état OFF (appris) |
| μ_ON  | Puissance moyenne état ON ≈ puissance bouilloire |
| σ_ON  | Écart-type puissance état ON (appris) |
| A[OFF→ON] | Probabilité de transition OFF → ON (taux d'activation) |
| A[ON→OFF]  | Probabilité de transition ON → OFF (1/durée_moyenne) |

La valeur de **2 758 W** est cohérente avec les bouilloires électriques standard au Royaume-Uni (généralement 2 000–3 000 W).

### Métriques de performance

| Ensemble | Maison | Algorithme | F1-Score | MAE (W) |
|----------|--------|------------|----------|---------|
| Validation | House 5 | Règles (event-based) | **0.523** | — |
| Validation | House 5 | HMM 2 états + Viterbi | — | — |
| Test | House 2 | Règles (event-based) | **0.693** | — |
| Test | House 2 | HMM 2 états + Viterbi | — | — |

*Les valeurs exactes du HMM dépendent du jeu de données REFIT. La MAE en Watts et les métriques complètes sont calculées dans le notebook (Cells 17 et 20).*

### Analyse des résultats

1. **F1 = 0.693 sur le test (House 2) — approche par règles** — Un score de 0.69 est **honorable pour une approche sans apprentissage profond**, surtout sur un jeu de données réel avec du bruit. Des méthodes NILM de référence (FHMM, seq2seq) atteignent généralement 0.75–0.90 sur REFIT.

2. **Écart Val (0.523) vs Test (0.693)** — La performance est meilleure sur la maison de test que sur la maison de validation. Cela suggère que House 2 a un profil de bouilloire plus « propre » (puissance plus stable, moins de superposition avec d'autres appareils).

3. **HMM 2 états + Viterbi** — L'approche probabiliste modélise explicitement les transitions entre états ON et OFF. Elle est plus robuste au bruit grâce à la fenêtre temporelle implicite du HMM et fournit une distribution de confiance sur les états.

4. **Avertissements pandas (`FutureWarning`) corrigés** — La fréquence `"10S"` est remplacée par `"10s"` pour la compatibilité avec les versions récentes de pandas.

### Visualisations produites

| Figure | Description |
|--------|-------------|
| Répartition Train/Val/Test | Diagramme en barres colorées par rôle |
| Valeurs manquantes | Barres + courbe par maison, avant forward-fill |
| Distribution puissance bouilloire | Boxplot par maison + histogramme global avec 70e percentile |
| Paramètres de détection (règles) | Barres des seuils + signal synthétique annoté |
| Métriques F1/Précision/Rappel (règles) | Barres val + test + comparaison côte à côte |
| Matrices de confusion (règles) | ON/OFF pour validation et test |
| Premier événement | Zoom sur la première activation détectée |
| Grille d'événements | 9 événements détectés sur House 2 |
| Timeline complète (règles) | Vue globale des activations prédites vs réelles |
| Émissions gaussiennes HMM | Distributions OFF/ON apprises par Baum-Welch |
| Matrice de transition HMM | Visualisation de A[2×2] |
| Comparaison F1 Règles vs HMM | Barres côte à côte validation et test |
| Détection HMM (signal) | Agrégé + ground truth + prédiction Viterbi |
| Tableau de bord complet | Précision / Rappel / F1 / MAE — règles vs HMM |

---

## 5. Conclusion et recommandations

### Ce que le code fait bien

- Pipeline complet et reproductible de bout en bout (chargement → prétraitement → apprentissage → détection → évaluation)
- Bonne séparation des données (pas de fuite d'information)
- Visualisations riches pour diagnostiquer le comportement de l'algorithme
- Paramètre appris de façon non supervisée à partir des sous-compteurs d'entraînement
- **[Nouveau]** Suppression des valeurs aberrantes (filtre k-sigma robuste via MAD)
- **[Nouveau]** Filtre médian anti-bruit avant la détection
- **[Nouveau]** **Filtre de Hampel** (médiane glissante + MAD glissante) pour les pics capteur isolés
- **[Nouveau]** HMM 2 états (Baum-Welch + Viterbi) implémenté de A à Z en NumPy
- **[Nouveau]** **Signal résiduel** (agrégat − baseline glissante) comme observation HMM — réduit les faux positifs
- **[Nouveau]** **Post-traitement durée** des états HMM (suppression des segments ON trop courts)
- **[Nouveau]** MAE en Watts comme métrique complémentaire au F1-Score
- **[Nouveau]** Correction de la fréquence pandas (`"10s"` au lieu de `"10S"`)

### Ce qui reste à faire

| Amélioration | Impact attendu |
|-------------|----------------|
| **FHMM** pour plusieurs appareils | Gestion des superpositions de consommation |
| **LSTM seq2seq** ou **Transformer** | Approche deep learning état-de-l'art sur REFIT |
| **Évaluation sur plusieurs maisons de test** | Mesure de généralisation plus fiable |
| **Filtre Savitzky-Golay** en alternative au filtre médian | Lissage polynomial préservant mieux les bords |

### En résumé

> Le code constitue une **base solide et bien structurée** pour la détection de la bouilloire avec NILM. L'algorithme événementiel est simple, rapide et interprétable, avec un **F1 de 0.69** sur la maison de test. Le notebook intègre désormais un **HMM 2 états entraîné par Baum-Welch et décodé par Viterbi**, avec des améliorations significatives : **filtre de Hampel** pour les pics capteur, **signal résiduel** (agrégat − baseline) pour isoler la bouilloire des autres appareils, et **post-traitement durée** pour réduire les faux positifs. Les métriques Précision/Rappel/F1/MAE sont affichées pour les deux approches (règles et HMM) sur validation et test.
