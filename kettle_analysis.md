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

2. **Absence de détection des valeurs aberrantes** — Aucun filtre pour les pics extrêmes (capteur défaillant, transitoire électrique) qui pourraient être confondus avec une mise en marche de bouilloire.

3. **Absence de normalisation** — Non critique ici car l'algorithme travaille directement en Watts, mais une normalisation faciliterait une éventuelle généralisation à d'autres jeux de données.

4. **`FutureWarning` pandas** — La fréquence `"10S"` est dépréciée dans les versions récentes de pandas ; il faudrait utiliser `"10s"` (minuscule) pour éviter des problèmes de compatibilité future.

5. **Aucune exploration de corrélation Aggregate/Kettle** — Vérifier que le signal agrégé contient bien la signature de la bouilloire renforcerait la confiance dans les données.

**Verdict** : Le preprocessing est **correct et suffisant pour une première approche**, mais manque de robustesse face aux valeurs aberrantes et nécessite une petite correction de syntaxe pandas.

---

## 3. L'algorithme utilisé : HMM et Viterbi ?

### ❌ Non — HMM et Viterbi ne sont PAS utilisés

Le code n'implémente ni Modèle de Markov Caché (Hidden Markov Model), ni l'algorithme de Viterbi. L'approche retenue est une **détection événementielle basée sur des règles** (*rule-based event detection*).

### Ce qui est réellement implémenté

```python
detect_kettle_from_aggregate(aggregate, kettle_on_power, sample_seconds, ...)
```

L'algorithme fonctionne en quatre étapes :

1. **Calcul de la dérivée discrète** — `d = diff(aggregate)` pour détecter les variations brusques de puissance.

2. **Détection de la montée** — Une montée est validée si `d[i] ≥ STEP_ON_FRAC × kettle_on_power` (≥ 75 % de la puissance apprise, soit ≈ 2069 W).

3. **Recherche de la descente** — Après une montée, on cherche une chute `d[j] ≤ −STEP_OFF_FRAC × kettle_on_power` (−65 %, soit ≈ −1793 W) dans la fenêtre de durée autorisée (90 s à 15 min).

4. **Validation par plateau** — L'épisode est confirmé si la médiane du signal pendant l'intervalle vérifie `|ΔP_median − kettle_on_power| ≤ POWER_TOL_FRAC × kettle_on_power` (±15 %, soit ±414 W).

### Comparaison avec une approche HMM / Viterbi

| Critère | Approche actuelle (règles) | Approche HMM + Viterbi |
|---------|---------------------------|------------------------|
| **Modélisation** | Seuils sur dérivée + durée | États cachés (OFF / ON) avec distributions d'émission |
| **Paramètres** | Seuils manuels | Probabilités de transition + paramètres d'émission appris |
| **Inférence** | Parcours linéaire `O(n)` | Algorithme de Viterbi `O(n × K²)` |
| **Robustesse au bruit** | Faible (seuil dur) | Meilleure (modèle probabiliste) |
| **Simultanéité d'appareils** | Aucune gestion | FHMM (Factorial HMM) pour plusieurs appareils |
| **Interprétabilité** | Très bonne | Bonne mais plus complexe |
| **Entraînement** | Statistique simple (quantile 70e) | Algorithme de Baum-Welch ou EM |

### Comment un HMM serait structuré pour ce problème

- **États cachés** : `{OFF, ON}` (2 états)  
- **Observations** : puissance agrégée ou variation de puissance  
- **Émissions** : distribution gaussienne `P(power | state)` — ex. `N(400, 100²)` pour OFF, `N(2758, 200²)` pour ON  
- **Transitions** : `P(ON → OFF)` = 1/durée_moyenne, `P(OFF → ON)` = taux d'activation  
- **Décodage** : algorithme de Viterbi pour trouver la séquence d'états la plus probable

---

## 4. Résultats

### Paramètre appris

| Paramètre | Valeur |
|-----------|--------|
| Puissance bouilloire ON (70e percentile) | **2 758 W** |
| Seuil montée (`STEP_ON_FRAC = 0.75`) | ≈ 2 069 W |
| Seuil descente (`STEP_OFF_FRAC = 0.65`) | ≈ 1 793 W |
| Tolérance plateau (`POWER_TOL_FRAC = 0.15`) | ≈ ±414 W |
| Durée min | 90 s |
| Durée max | 15 min |

La valeur de **2 758 W** est cohérente avec les bouilloires électriques standard au Royaume-Uni (généralement 2 000–3 000 W).

### Métriques de performance

| Ensemble | Maison | F1-Score |
|----------|--------|----------|
| Validation | House 5 | **0.523** |
| Test | House 2 | **0.693** |

*La Précision et le Rappel sont calculés et visualisés dans le notebook (Cell 9) mais non affichés numériquement dans les sorties.*

### Analyse des résultats

1. **F1 = 0.693 sur le test (House 2)** — Un score de 0.69 est **honorable pour une approche sans apprentissage profond**, surtout sur un jeu de données réel avec du bruit. Des méthodes NILM de référence (FHMM, seq2seq) atteignent généralement 0.75–0.90 sur REFIT.

2. **Écart Val (0.523) vs Test (0.693)** — La performance est meilleure sur la maison de test que sur la maison de validation. Cela suggère que House 2 a un profil de bouilloire plus « propre » (puissance plus stable, moins de superposition avec d'autres appareils). Cela indique aussi que la maison de validation est plus représentative des cas difficiles.

3. **Avertissements pandas (`FutureWarning`)** — Purement cosmétiques, sans impact sur les résultats, mais à corriger pour la maintenabilité.

### Visualisations produites

| Figure | Description |
|--------|-------------|
| Répartition Train/Val/Test | Diagramme en barres colorées par rôle |
| Valeurs manquantes | Barres + courbe par maison, avant forward-fill |
| Distribution puissance bouilloire | Boxplot par maison + histogramme global avec 70e percentile |
| Paramètres de détection | Barres des seuils + signal synthétique annoté |
| Métriques F1/Précision/Rappel | Barres val + test + comparaison côte à côte |
| Matrices de confusion | ON/OFF pour validation et test |
| Premier événement | Zoom sur la première activation détectée |
| Grille d'événements | 9 événements détectés sur House 2 |
| Timeline complète | Vue globale des activations prédites vs réelles |

---

## 5. Conclusion et recommandations

### Ce que le code fait bien

- Pipeline complet et reproductible de bout en bout (chargement → prétraitement → apprentissage → détection → évaluation)
- Bonne séparation des données (pas de fuite d'information)
- Visualisations riches pour diagnostiquer le comportement de l'algorithme
- Paramètre appris de façon non supervisée à partir des sous-compteurs d'entraînement

### Ce qui est absent et mériterait d'être ajouté

| Amélioration | Impact attendu |
|-------------|----------------|
| **Filtre anti-bruit** (médian ou Savitzky-Golay) avant détection | Réduit les fausses détections dues aux transitoires |
| **HMM 2 états** (Baum-Welch + Viterbi) | Modélisation probabiliste plus robuste |
| **FHMM** pour plusieurs appareils | Gestion des superpositions de consommation |
| **LSTM seq2seq** ou **Transformer** | Approche deep learning état-de-l'art sur REFIT |
| **Correction de la fréquence pandas** (`"10s"` au lieu de `"10S"`) | Compatibilité future |
| **Détection des valeurs aberrantes** | Moins de faux positifs |
| **Évaluation sur plusieurs maisons de test** | Mesure de généralisation plus fiable |
| **Rapport MAE en Watts** | Complète le F1-Score (mesure l'erreur d'estimation énergétique) |

### En résumé

> Le code constitue une **base solide et bien structurée** pour la détection de la bouilloire avec NILM. L'algorithme événementiel est simple, rapide et interprétable, avec un **F1 de 0.69** sur la maison de test. **HMM et Viterbi ne sont pas utilisés** : les intégrer serait la prochaine étape naturelle pour améliorer la robustesse probabiliste et préparer l'extension à d'autres appareils.
