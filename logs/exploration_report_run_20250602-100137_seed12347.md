# Rapport d'exploration FPS

**Run ID :** run_20250602-100137_seed12347
**Date :** 2025-06-02 10:01:37
**Total événements :** 18

## Résumé par type d'événement

- **anomaly** : 5 événements
- **fractal_pattern** : 13 événements

## Anomaly

### 1. t=252-301
- **Métrique :** mean_high_effort
- **Valeur :** 32.1699
- **Sévérité :** high

### 2. t=253-302
- **Métrique :** mean_high_effort
- **Valeur :** 6.8058
- **Sévérité :** medium

### 3. t=254-303
- **Métrique :** mean_high_effort
- **Valeur :** 4.8083
- **Sévérité :** medium

### 4. t=255-304
- **Métrique :** mean_high_effort
- **Valeur :** 3.8914
- **Sévérité :** low

### 5. t=256-281
- **Métrique :** mean_high_effort
- **Valeur :** 3.3329
- **Sévérité :** low

## Fractal Pattern

### 1. t=250-350
- **Métrique :** A_mean(t)
- **Valeur :** 0.9623
- **Sévérité :** high
- **scale :** 10/100

### 2. t=300-400
- **Métrique :** mean_high_effort
- **Valeur :** 0.8945
- **Sévérité :** medium
- **scale :** 10/100

### 3. t=350-450
- **Métrique :** mean_high_effort
- **Valeur :** 0.8926
- **Sévérité :** medium
- **scale :** 10/100

### 4. t=400-500
- **Métrique :** mean_high_effort
- **Valeur :** 0.8905
- **Sévérité :** medium
- **scale :** 10/100

### 5. t=450-550
- **Métrique :** mean_high_effort
- **Valeur :** 0.8891
- **Sévérité :** medium
- **scale :** 10/100

## Motifs fractals détectés

**Nombre total :** 13

### A_mean(t)
- Patterns détectés : 1
- Corrélation moyenne : 0.962
- Corrélation max : 0.962

### mean_high_effort
- Patterns détectés : 12
- Corrélation moyenne : 0.887
- Corrélation max : 0.895

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
