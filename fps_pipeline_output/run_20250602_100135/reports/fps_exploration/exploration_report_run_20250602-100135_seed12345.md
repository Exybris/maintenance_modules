# Rapport d'exploration FPS

**Run ID :** run_20250602-100135_seed12345
**Date :** 2025-06-02 10:01:36
**Total événements :** 180

## Résumé par type d'événement

- **anomaly** : 6 événements
- **phase_cycle** : 165 événements
- **fractal_pattern** : 9 événements

## Anomaly

### 1. t=249-298
- **Métrique :** C(t)
- **Valeur :** 43.5630
- **Sévérité :** high

### 2. t=250-299
- **Métrique :** C(t)
- **Valeur :** 13.8705
- **Sévérité :** high

### 3. t=250-299
- **Métrique :** S(t)
- **Valeur :** 8.5974
- **Sévérité :** medium

### 4. t=251-276
- **Métrique :** C(t)
- **Valeur :** 6.0745
- **Sévérité :** medium

### 5. t=252-265
- **Métrique :** C(t)
- **Valeur :** 4.4202
- **Sévérité :** low

## Phase Cycle

### 1. t=5-39
- **Métrique :** S(t)
- **Valeur :** 34.0000
- **Sévérité :** medium

### 2. t=6-34
- **Métrique :** S(t)
- **Valeur :** 28.0000
- **Sévérité :** medium

### 3. t=829-854
- **Métrique :** S(t)
- **Valeur :** 25.0000
- **Sévérité :** medium

### 4. t=278-302
- **Métrique :** S(t)
- **Valeur :** 24.0000
- **Sévérité :** medium

### 5. t=7-30
- **Métrique :** S(t)
- **Valeur :** 23.0000
- **Sévérité :** medium

## Fractal Pattern

### 1. t=150-250
- **Métrique :** C(t)
- **Valeur :** 0.9538
- **Sévérité :** high
- **scale :** 10/100

### 2. t=400-500
- **Métrique :** C(t)
- **Valeur :** 0.9292
- **Sévérité :** high
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** C(t)
- **Valeur :** 0.9279
- **Sévérité :** high
- **scale :** 10/100

### 4. t=300-400
- **Métrique :** C(t)
- **Valeur :** 0.9276
- **Sévérité :** high
- **scale :** 10/100

### 5. t=100-200
- **Métrique :** C(t)
- **Valeur :** 0.9271
- **Sévérité :** high
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 9

### S(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.837
- Corrélation max : 0.837

### C(t)
- Patterns détectés : 8
- Corrélation moyenne : 0.922
- Corrélation max : 0.954

## Configuration d'exploration

```json
{
  "metrics": [
    "S(t)",
    "C(t)",
    "A_mean(t)",
    "f_mean(t)",
    "entropy_S",
    "effort(t)",
    "mean_high_effort",
    "d_effort_dt",
    "mean_abs_error"
  ],
  "window_sizes": [
    1,
    10,
    100
  ],
  "fractal_threshold": 0.8,
  "detect_fractal_patterns": true,
  "detect_anomalies": true,
  "detect_harmonics": true,
  "recurrence_window": [
    1,
    10,
    100
  ],
  "anomaly_threshold": 3.0,
  "min_duration": 3
}
```
