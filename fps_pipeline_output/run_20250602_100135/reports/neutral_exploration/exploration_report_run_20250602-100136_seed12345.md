# Rapport d'exploration FPS

**Run ID :** run_20250602-100136_seed12345
**Date :** 2025-06-02 10:01:36
**Total événements :** 39

## Résumé par type d'événement

- **phase_cycle** : 39 événements

## Phase Cycle

### 1. t=15-23
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 2. t=35-43
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 3. t=65-73
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 4. t=85-93
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

### 5. t=115-123
- **Métrique :** S(t)
- **Valeur :** 8.0000
- **Sévérité :** low

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
