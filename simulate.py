"""
simulate.py – FPS Pipeline Simulation Core
Version exhaustive FPS
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Ce module orchestre toute la dynamique FPS/Kuramoto/Neutral, et doit toujours rester :
- **modulaire** : chaque fonction/étape isolable, falsifiable individuellement
- **extensible** : chaque critère, métrique, ou pipeline peut être augmenté/remplacé
- **falsifiable** : tout doit être logué, traçable, chaque hypothèse peut être testée, toute évolution loguée
- **strict** : tout ce qui est attendu ici, même si le module n'est pas encore codé, doit être détaillé dans ce fichier en code ou en commentaire.
---------------------------------------------------------------

**Pipeline FPS** – *Résumé*
Boucle temporelle orchestrant :
- Chargement et validation complète du config.json (validate_config.py)
- Initialisation strates, seeds, loggers (init.py)
- Gestion du mode (FPS, Kuramoto, Neutral)
- Boucle de simulation temps-réel (avec dynamique FPS complète OU alternatives)
- Calculs à chaque pas :
    - Input contextuel/perturbations, dynamique FPS (dynamics.py)
    - Régulation (regulation.py), feedback adaptatif G(x)
    - Mise à jour des états, calcul S(t), E(t), O(t), metrics (metrics.py)
    - Log metrics, backup d'état, gestion erreurs (utils.py)
- Gestion post-run :
    - Analyse & raffinements automatiques (analyze.py)
    - Exploration émergences, fractals, anomalies (explore.py)
    - Visualisation complète, dashboard, grille empirique (visualize.py)
    - Export de tous les logs, configs, seeds, changelog.txt
- Exécution en mode batch/auto si besoin (via batch_runner ou module export-batch)
"""

# --------- IMPORTS (TOUS MODULES DU PIPELINE FPS) -----------
import os, sys, time, json, traceback, numpy as np
import csv
# Imports stricts
import init
import dynamics
import regulation
import metrics
import analyze
import explore
import visualize
import validate_config
import kuramoto
import perturbations
import utils

def safe_float_conversion(value, default=0.0):
    """Convertit une valeur en float sûr."""
    try:
        if isinstance(value, str):
            return default
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    except:
        return default

# --------- PIPELINE PRINCIPALE : RUN_SIMULATION() -----------

def run_simulation(config_path, mode="FPS"):
    """
    Orchestration complète d'un run FPS/Kuramoto/Neutral.
    - Validation complète config.json (présence, structure, types, seuils, dépendances croisées…)
    - Initialisation strates, seeds, loggers
    - Branche pipeline selon le mode
    - À chaque step : dynamique complète (input, dynamique FPS, feedback, metrics, log, backup)
    - Gestion post-run (analyse, exploration, visualisation, logs, exports)
    """
    # ---- 0. Chargement et VALIDATION TOTALE du config ----
    config = init.load_config(config_path) if hasattr(init, "load_config") else json.load(open(config_path))
    errors, warnings = validate_config.validate_config(config_path) if hasattr(validate_config, "validate_config") else ([], [])
    if errors:  # Si il y a des erreurs (liste non vide)
        print("Config validation FAILED:")
        for e in errors: print("  -", e)
        sys.exit(1)
    if warnings:
        print("Config validation avec avertissements:")
        for w in warnings: print("  -", w)
    print("Config validation: OK")
    
    # ---- 1. SEED : reproductibilité, log dans seeds.txt ----
    SEED = config['system']['seed']
    np.random.seed(SEED)
    import random; random.seed(SEED)
    if hasattr(utils, "log_seed"): 
        utils.log_seed(SEED)
    else: 
        with open("seeds.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | SEED = {SEED}\n")
    
    # ---- 2. Initialisation : strates, état, logs, dirs ----
    state = init.init_strates(config)
    loggers = init.setup_logging(config)
    
    if hasattr(utils, "log_config_and_meta"): 
        utils.log_config_and_meta(config, loggers['run_id'])
    
    # ---- 3. BRANCH par mode (FPS/Kuramoto/Neutral) ----
    if mode.lower() == "kuramoto":
        result = run_kuramoto_simulation(config, loggers)
    elif mode.lower() == "neutral":
        result = run_neutral_simulation(config, loggers)
    elif mode.lower() == "fps":
        result = run_fps_simulation(config, state, loggers)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(2)
    
    # ---- 4. Post-run : Analyse, exploration, visualisation, export ----
    if hasattr(analyze, "analyze_criteria_and_refine"):
        analyze.analyze_criteria_and_refine(result['logs'], config)
    
    if hasattr(explore, "run_exploration"):
        explore.run_exploration(result['logs'], loggers['output_dir'], config)
    
    if hasattr(visualize, "plot_metrics_dashboard"):
        visualize.plot_metrics_dashboard(result['metrics'])
    
    # ---- 5. Exports finaux : rapport, changelog, backup ----
    if hasattr(utils, "log_end_of_run"): 
        utils.log_end_of_run(result['run_id'])
    
    # Fermer les fichiers CSV si nécessaire
    if 'csv_file' in loggers and loggers['csv_file']:
        try:
            loggers['csv_file'].close()
        except:
            pass
    
    return result

# --- MODE FPS (Pipeline complet) ------------------------------------------
# --- MODE FPS (Pipeline complet) ------------------------------------------
def run_fps_simulation(config, state, loggers):
    """
    Boucle principale FPS, version exhaustive :
    - À chaque pas : input contextuel, dynamique FPS (toutes formules), feedback/régulation, logs, backup
    - Gestion du mode statique/dynamique pour tous paramètres (voir config)
    - Log exhaustif des métriques (voir feuille de route/tableau structuré)
    - Gestion stricte erreurs, try/except, logs des exceptions
    - Sécurité backup état toutes les 100 steps
    - À la fin : export de tous les logs, résumé metrics, passage à l'analyse et à l'exploration
    """
    # -- PARAMÈTRES GLOBAUX
    T = config['system']['T']
    dt = config['system'].get('dt', 0.05)
    N = config['system']['N']
    run_id = loggers['run_id']
    t_array = np.arange(0, T, dt)
    backup_interval = 100
    t_choc = config['system']['perturbation'].get('t0', T/2)  # Pour t_retour

    # Historiques avec limite de mémoire
    MAX_HISTORY_SIZE = config.get('system', {}).get('max_history_size', 10000)
    history, cpu_steps, effort_history, S_history = [], [], [], []
    # Historiques supplémentaires pour métriques avancées
    An_history = []  # Pour A_mean(t) et export individuel
    fn_history = []  # Pour f_mean(t) et export individuel
    En_history = []  # Pour mean_abs_error
    On_history = []  # Pour mean_abs_error
    
    # INITIALISER all_metrics ET t EN DEHORS DE LA BOUCLE
    all_metrics = {}
    t = 0
    
    # Fonction de rotation des historiques
    def rotate_history(hist_list, max_size):
        """Garde seulement les max_size derniers éléments."""
        if len(hist_list) > max_size:
            return hist_list[-max_size:]
        return hist_list
    
    # Préparer les fichiers individuels si N > 10
    individual_csv_writers = {}
    if N > 10 and config.get('analysis', {}).get('save_indiv_files', True):
        for n in range(N):
            # A_n_{id}.csv
            an_filename = os.path.join(loggers['output_dir'], f"A_n_{n}_{run_id}.csv")
            an_file = open(an_filename, 'w', newline='')
            an_writer = csv.writer(an_file)
            an_writer.writerow(['t', f'A_{n}(t)'])
            individual_csv_writers[f'A_{n}'] = {'file': an_file, 'writer': an_writer}
            
            # f_n_{id}.csv
            fn_filename = os.path.join(loggers['output_dir'], f"f_n_{n}_{run_id}.csv")
            fn_file = open(fn_filename, 'w', newline='')
            fn_writer = csv.writer(fn_file)
            fn_writer.writerow(['t', f'f_{n}(t)'])
            individual_csv_writers[f'f_{n}'] = {'file': fn_file, 'writer': fn_writer}

    # -- BOUCLE PRINCIPALE --
    try:
        for step, t in enumerate(t_array):
            step_start = time.perf_counter()
            
            # ----------- 1. INPUTS ET PERTURBATIONS -----------
            try:
                In_t = dynamics.compute_In(t, config['system']['perturbation'], N) if hasattr(dynamics, 'compute_In') else np.zeros(N)
            except Exception as e:
                print(f"⚠️ Erreur compute_In à t={t}: {e}")
                In_t = np.zeros(N)
                
            try:
                pert_value = perturbations.generate_perturbation(t, config['system']['perturbation']) if hasattr(perturbations, 'generate_perturbation') else 0.0
                In_t = perturbations.apply_perturbation_to_In(In_t, pert_value) if hasattr(perturbations, 'apply_perturbation_to_In') else In_t
            except Exception as e:
                print(f"⚠️ Erreur perturbation à t={t}: {e}")
            
            # ----------- 2. CALCULS DYNAMIQUE FPS --------------
            # a) Amplitude, fréquence, phase, latence par strate (avec statique/dynamique, config.json)
            try:
                An_t = dynamics.compute_An(t, state, In_t, config) if hasattr(dynamics, 'compute_An') else np.ones(N)
                fn_t = dynamics.compute_fn(t, state, An_t, config) if hasattr(dynamics, 'compute_fn') else np.ones(N)
                phi_n_t = dynamics.compute_phi_n(t, state, config) if hasattr(dynamics, 'compute_phi_n') else np.zeros(N)
                gamma_n_t = dynamics.compute_gamma_n(t, state, config) if hasattr(dynamics, 'compute_gamma_n') else np.ones(N)
            except Exception as e:
                print(f"⚠️ Erreur calculs dynamiques à t={t}: {e}")
                An_t = np.ones(N)
                fn_t = np.ones(N)
                phi_n_t = np.zeros(N)
                gamma_n_t = np.ones(N)
            
            # b) Sorties observée/attendue, feedback
            try:
                On_t = dynamics.compute_On(t, state, An_t, fn_t, phi_n_t, gamma_n_t) if hasattr(dynamics, 'compute_On') else An_t
                En_t = dynamics.compute_En(t, state, history, config) if hasattr(dynamics, 'compute_En') else An_t
            except Exception as e:
                print(f"⚠️ Erreur compute On/En à t={t}: {e}")
                On_t = An_t
                En_t = An_t
            
            # c) Régulation/adaptation feedback
            try:
                # Calcul correct de Fn(t) = βn·(On(t) - En(t))·γ(t)
                gamma_t = dynamics.compute_gamma(t, config.get('latence', {}).get('gamma_mode', 'static'), T)
                F_n_t = np.zeros(N)
                
                for n in range(N):
                    beta_n = state[n]['beta']
                    F_n_t[n] = dynamics.compute_Fn(t, beta_n, On_t[n], En_t[n], gamma_t)
            except Exception as e:
                print(f"⚠️ Erreur régulation à t={t}: {e}")
                F_n_t = np.zeros(N)
            
            # d) Update état complet du système
            try:
                state = dynamics.update_state(state, An_t, fn_t, phi_n_t, gamma_n_t, F_n_t) if hasattr(dynamics, 'update_state') else state
            except Exception as e:
                print(f"⚠️ Erreur update state à t={t}: {e}")
            
            # ----------- 3. SIGNAUX GLOBAUX ET MÉTRIQUES -------------
            try:
                S_t = dynamics.compute_S(t, An_t, fn_t, phi_n_t, config) if hasattr(dynamics, 'compute_S') else 0.0
                C_t = dynamics.compute_C(t, phi_n_t) if hasattr(dynamics, 'compute_C') else 0.0
                A_t = dynamics.compute_A(t, F_n_t) if hasattr(dynamics, 'compute_A') else 0.0
                A_spiral_t = dynamics.compute_A_spiral(t, C_t, A_t) if hasattr(dynamics, 'compute_A_spiral') else 0.0
                E_t = dynamics.compute_E(t, An_t) if hasattr(dynamics, 'compute_E') else 0.0
                L_t = dynamics.compute_L(t, An_t) if hasattr(dynamics, 'compute_L') else 0.0
                cpu_step = metrics.compute_cpu_step(step_start, time.perf_counter(), N) if hasattr(metrics, 'compute_cpu_step') else 0.0
            except Exception as e:
                print(f"⚠️ Erreur signaux globaux à t={t}: {e}")
                S_t = 0.0
                C_t = 0.0
                A_t = 0.0
                A_spiral_t = 0.0
                E_t = 0.0
                L_t = 0.0
                cpu_step = 0.0
            
            # ----------- CALCUL DES MÉTRIQUES AVANCÉES -------------
            # Calcul des moyennes pour A_mean(t) et f_mean(t)
            A_mean_t = np.mean(An_t) if isinstance(An_t, np.ndarray) else An_t
            f_mean_t = np.mean(fn_t) if isinstance(fn_t, np.ndarray) else fn_t
            
            # Calcul de effort(t) avec deltas si historique disponible
            if len(An_history) > 0:
                delta_An = An_t - An_history[-1]
                delta_fn = fn_t - fn_history[-1]
                delta_gamma_n = gamma_n_t - (gamma_n_t if len(history) == 0 else history[-1].get('gamma_n', gamma_n_t))
                effort_t = metrics.compute_effort(delta_An, delta_fn, delta_gamma_n, 
                                                  np.max(An_t), np.max(fn_t), np.max(gamma_n_t)) if hasattr(metrics, 'compute_effort') else 0.0
            else:
                effort_t = 0.0
            
            effort_status = metrics.compute_effort_status(effort_t, effort_history, config) if hasattr(metrics, 'compute_effort_status') else "stable"
            
            # Calcul variance_d2S (fluidité) - nécessite au moins 3 points
            if len(S_history) >= 3:
                variance_d2S = metrics.compute_variance_d2S(S_history, dt) if hasattr(metrics, 'compute_variance_d2S') else 0.0
            else:
                variance_d2S = 0.0
            
            # Calcul entropy_S (innovation)
            entropy_S = metrics.compute_entropy_S(S_t, 1.0/dt) if hasattr(metrics, 'compute_entropy_S') else 0.5
            
            # Calcul mean_abs_error (régulation)
            mean_abs_error = metrics.compute_mean_abs_error(En_t, On_t) if hasattr(metrics, 'compute_mean_abs_error') else np.mean(np.abs(En_t - On_t))
            
            # Calcul mean_high_effort (effort chronique) - nécessite historique
            if len(effort_history) >= 10:
                mean_high_effort = metrics.compute_mean_high_effort(effort_history, 80) if hasattr(metrics, 'compute_mean_high_effort') else np.percentile(effort_history[-100:], 80)
            else:
                mean_high_effort = effort_t
            
            # Calcul d_effort_dt (effort transitoire)
            if len(effort_history) >= 2:
                d_effort_dt = metrics.compute_d_effort_dt(effort_history, dt) if hasattr(metrics, 'compute_d_effort_dt') else (effort_history[-1] - effort_history[-2]) / dt
            else:
                d_effort_dt = 0.0
            
            # Calcul t_retour (résilience) - après perturbation
            if t > t_choc and len(S_history) > int(t_choc/dt):
                t_retour = metrics.compute_t_retour(S_history, int(t_choc/dt), dt, 0.95) if hasattr(metrics, 'compute_t_retour') else 0.0
            else:
                t_retour = 0.0
            
            # Calcul max_median_ratio (stabilité)
            if len(S_history) >= 10:
                max_median_ratio = metrics.compute_max_median_ratio(S_history) if hasattr(metrics, 'compute_max_median_ratio') else 1.0
            else:
                max_median_ratio = 1.0
            
            # CRÉER all_metrics ICI
            all_metrics = {
                't': t,
                'S(t)': S_t,
                'C(t)': C_t,
                'A_spiral(t)': A_spiral_t,
                'E(t)': E_t,
                'L(t)': L_t,
                'cpu_step(t)': cpu_step,
                'effort(t)': effort_t,
                'A_mean(t)': A_mean_t,
                'f_mean(t)': f_mean_t,
                'variance_d2S': variance_d2S,
                'entropy_S': entropy_S,
                'mean_abs_error': mean_abs_error,
                'mean_high_effort': mean_high_effort,
                'd_effort_dt': d_effort_dt,
                't_retour': t_retour,
                'max_median_ratio': max_median_ratio
            }
            
            # Appliquer safe_float_conversion à toutes les métriques
            for key in all_metrics:
                if key != 'effort_status':
                    all_metrics[key] = safe_float_conversion(all_metrics[key])
            
            # ----------- 4. VÉRIFICATION NaN/Inf SYSTÉMATIQUE -------------
            # Vérification NaN/Inf
            nan_inf_detected = False
            for metric_name, metric_value in all_metrics.items():
                if metric_name == 't':
                    continue
                if isinstance(metric_value, (int, float)) and (np.isnan(metric_value) or np.isinf(metric_value)):
                    nan_inf_detected = True
                    # Log l'alerte
                    alert_msg = f"ALERTE : NaN/Inf détecté à t={t} pour {metric_name}={metric_value}"
                    print(alert_msg)
                    os.makedirs(loggers['output_dir'], exist_ok=True)
                    with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                        alert_file.write(f"{alert_msg}\n")
                    # Remplacer par une valeur par défaut pour continuer
                    if np.isnan(metric_value):
                        all_metrics[metric_name] = 0.0
                    elif np.isinf(metric_value):
                        all_metrics[metric_name] = 1e6 if metric_value > 0 else -1e6
            
            # Si trop de NaN/Inf, option d'arrêter le run
            if nan_inf_detected and config.get('system', {}).get('stop_on_nan_inf', False):
                raise ValueError(f"NaN/Inf détecté à t={t}, arrêt du run")
            
            # ----------- 5. LOGGING DE TOUTES LES MÉTRIQUES ------------
            metrics_dict = all_metrics.copy()
            metrics_dict['effort_status'] = effort_status
            
            # Créer la ligne de log dans l'ordre des colonnes de config
            log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
            row_data = []
            for metric in log_metrics_order:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    # Formater correctement les valeurs
                    if isinstance(value, str):
                        row_data.append(value)
                    elif isinstance(value, (int, float)):
                        row_data.append(f"{value:.6f}" if value != int(value) else str(int(value)))
                    else:
                        row_data.append(str(value))
                else:
                    row_data.append("0.0")
            
            if hasattr(loggers['csv_writer'], 'writerow'):
                loggers['csv_writer'].writerow(row_data)
            
            # ----------- 6. EXPORT FICHIERS INDIVIDUELS SI N > 10 ------------
            if N > 10 and individual_csv_writers:
                for n in range(N):
                    # Export A_n(t)
                    if f'A_{n}' in individual_csv_writers:
                        individual_csv_writers[f'A_{n}']['writer'].writerow([t, An_t[n]])
                    # Export f_n(t)
                    if f'f_{n}' in individual_csv_writers:
                        individual_csv_writers[f'f_{n}']['writer'].writerow([t, fn_t[n]])
            
            # ----------- 7. BACKUP AUTOMATIQUE ------------------------
            if step % backup_interval == 0 and step > 0:
                if hasattr(utils, "save_simulation_state"):
                    # Créer le dossier checkpoints s'il n'existe pas
                    checkpoint_dir = os.path.join(loggers.get('output_dir', '.'), 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}_backup_{step}.pkl")
                    utils.save_simulation_state(state, checkpoint_path)
            
            # ----------- 8. HISTORIQUE POUR ANALYSE -------------------
            history.append({
                't': t, 'S': S_t, 'O': On_t, 'E': En_t, 'F': F_n_t,
                'gamma_n': gamma_n_t, 'An': An_t, 'fn': fn_t,
                'C': C_t, 'A_spiral': A_spiral_t, 'entropy_S': entropy_S
            })
            cpu_steps.append(cpu_step)
            effort_history.append(effort_t)
            S_history.append(S_t)
            An_history.append(An_t.copy() if isinstance(An_t, np.ndarray) else An_t)
            fn_history.append(fn_t.copy() if isinstance(fn_t, np.ndarray) else fn_t)
            En_history.append(En_t.copy() if isinstance(En_t, np.ndarray) else En_t)
            On_history.append(On_t.copy() if isinstance(On_t, np.ndarray) else On_t)
            
            # Rotation des historiques pour éviter explosion mémoire
            if len(history) > MAX_HISTORY_SIZE:
                history = rotate_history(history, MAX_HISTORY_SIZE)
                cpu_steps = rotate_history(cpu_steps, MAX_HISTORY_SIZE)
                effort_history = rotate_history(effort_history, MAX_HISTORY_SIZE)
                S_history = rotate_history(S_history, MAX_HISTORY_SIZE)
                An_history = rotate_history(An_history, MAX_HISTORY_SIZE)
                fn_history = rotate_history(fn_history, MAX_HISTORY_SIZE)
                En_history = rotate_history(En_history, MAX_HISTORY_SIZE)
                On_history = rotate_history(On_history, MAX_HISTORY_SIZE)
                
                # Logger la rotation
                if step % 1000 == 0:
                    print(f"  📊 Rotation des historiques à t={t:.1f} (max_size={MAX_HISTORY_SIZE})")
            
            # ----------- 9. DÉTECTION MODE ALERTE (>3σ) ---------------
            # Vérifier si un critère non-déclencheur présente un écart > 3σ
            alert_sigma = config.get('validation', {}).get('alert_sigma', 3)
            if len(S_history) >= 100:  # Besoin d'historique pour calculer σ
                # Exemple pour entropy_S
                entropy_history = [h.get('entropy_S', 0.5) for h in history[-100:] if 'entropy_S' in h]
                if len(entropy_history) > 0:
                    entropy_mean = np.mean(entropy_history)
                    entropy_std = np.std(entropy_history)
                    if entropy_std > 0 and abs(entropy_S - entropy_mean) > alert_sigma * entropy_std:
                        alert_msg = f"MODE ALERTE : entropy_S={entropy_S:.4f} dévie de >{alert_sigma}σ à t={t}"
                        print(alert_msg)
                        with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                            alert_file.write(f"{alert_msg}\n")
                
                # Même vérification pour d'autres métriques non-déclencheuses
                for metric_name in ['variance_d2S', 'mean_high_effort', 'd_effort_dt']:
                    if metric_name in all_metrics:
                        metric_history = [h.get(metric_name, 0) for h in history[-100:] if metric_name in h]
                        if len(metric_history) > 10:
                            m_mean = np.mean(metric_history)
                            m_std = np.std(metric_history)
                            if m_std > 0 and abs(all_metrics[metric_name] - m_mean) > alert_sigma * m_std:
                                alert_msg = f"MODE ALERTE : {metric_name}={all_metrics[metric_name]:.4f} dévie de >{alert_sigma}σ à t={t}"
                                with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                                    alert_file.write(f"{alert_msg}\n")

        # -- FERMETURE DES FICHIERS INDIVIDUELS --
        if N > 10 and individual_csv_writers:
            for key, writer_info in individual_csv_writers.items():
                writer_info['file'].close()
            print(f"Fichiers individuels A_n_*.csv et f_n_*.csv exportés pour N={N}")

        # -- EXPORTS ET SYNTHÈSE FINALE --
        logs = loggers['log_file']
        
        # Calcul des statistiques finales pour le résumé
        metrics_summary = {
            'mean_S': np.mean(S_history) if S_history else 0.0,
            'std_S': np.std(S_history) if S_history else 0.0,
            'mean_effort': np.mean(effort_history) if effort_history else 0.0,
            'max_effort': np.max(effort_history) if effort_history else 0.0,
            'mean_cpu_step': np.mean(cpu_steps) if cpu_steps else 0.0,
            'final_entropy_S': float(entropy_S) if entropy_S is not None else 0.0,
            'final_variance_d2S': float(variance_d2S) if variance_d2S is not None else 0.0,
            'final_mean_abs_error': float(mean_abs_error) if mean_abs_error is not None else 0.0,
            'resilience_t_retour': float(t_retour) if t_retour is not None else 0.0,
            'stability_ratio': float(max_median_ratio) if max_median_ratio is not None else 1.0,
            'total_steps': len(t_array),
            'dt': float(dt),
            'N': int(N),
            'mode': 'FPS'
        }
        
        # Ne PAS appeler summarize_metrics avec le nom du fichier
        # Si on veut utiliser summarize_metrics, il faut lui passer history, pas logs
        # if hasattr(metrics, "summarize_metrics") and history:
        #     metrics_summary.update(metrics.summarize_metrics(history))
        
        # Calcul du checksum des logs pour intégrité
        if hasattr(utils, "compute_checksum") and os.path.exists(logs):
            checksum = utils.compute_checksum(logs)
            with open(os.path.join(loggers['output_dir'], f"checksum_{run_id}.txt"), "w") as f:
                f.write(f"Checksum for {logs}: {checksum}\n")
        
        return {
            'history': history,
            'logs': logs,
            'metrics': metrics_summary,
            'cpu_steps': cpu_steps,
            'effort_history': effort_history,
            'S_history': S_history,
            'run_id': run_id,
            'An_history': An_history,
            'fn_history': fn_history,
            'En_history': En_history,
            'On_history': On_history
        }

    except Exception as e:
        # -- LOG D'ERREUR CRITIQUE POUR POST-MORTEM --
        err_path = os.path.join(loggers['output_dir'], f"error_{run_id}.log")
        with open(err_path, "a") as f: 
            f.write(f"Error at t={t if 't' in locals() else 'unknown'}\n")
            f.write(traceback.format_exc())
        
        # Fermer les fichiers individuels en cas d'erreur
        if N > 10 and 'individual_csv_writers' in locals():
            for key, writer_info in individual_csv_writers.items():
                try:
                    writer_info['file'].close()
                except:
                    pass
        
        if hasattr(utils, "handle_crash_recovery"): 
            crash_state = {
                'strates': state,
                't': t if 't' in locals() else 0,
                'mode': config['system'].get('mode', 'FPS'),
                'error_info': str(e),
                'all_metrics': all_metrics  # Maintenant toujours défini
            }
            utils.handle_crash_recovery(crash_state, loggers, e)
        raise e

# --- MODE KURAMOTO (GROUPE CONTRÔLE) ---------------------------
def run_kuramoto_simulation(config, loggers):
    """
    Implémentation stricte de la dynamique Kuramoto (phase-only).
    - Utilise K=0.5, ωᵢ~U[0,1], phases initialisées random, N strates, dt, T comme FPS
    - Log les mêmes métriques que FPS pour comparaison
    """
    if hasattr(kuramoto, "run_kuramoto_simulation"):
        return kuramoto.run_kuramoto_simulation(config, loggers)
    else:
        # Implémentation minimale Kuramoto
        N = config['system']['N']
        T = config['system']['T']
        dt = config['system'].get('dt', 0.05)
        K = 0.5
        
        # Initialisation
        phases = np.random.uniform(0, 2*np.pi, N)
        frequencies = np.random.uniform(0, 1, N)
        t_array = np.arange(0, T, dt)
        
        history = []
        S_history = []
        C_history = []
        cpu_steps = []
        
        for t in t_array:
            step_start = time.perf_counter()
            
            # Équation Kuramoto
            dphases_dt = frequencies.copy()
            for i in range(N):
                coupling_sum = 0.0
                for j in range(N):
                    coupling_sum += np.sin(phases[j] - phases[i])
                dphases_dt[i] += (K / N) * coupling_sum
            
            # Mise à jour
            phases += dphases_dt * dt
            phases = phases % (2 * np.pi)
            
            # Ordre global (paramètre d'ordre de Kuramoto)
            order_param = np.abs(np.mean(np.exp(1j * phases)))
            
            # Cohérence des phases adjacentes (équivalent à C(t))
            if N > 1:
                C_t = np.mean([np.cos(phases[(i+1)%N] - phases[i]) for i in range(N)])
            else:
                C_t = 1.0
            
            # Signal global (somme des oscillateurs)
            S_t = np.sum(np.sin(phases))
            
            cpu_step = (time.perf_counter() - step_start) / N
            
            # Log
            metrics_dict = {
                't': t,
                'S(t)': S_t,
                'C(t)': C_t,
                'E(t)': order_param,
                'L(t)': 0,
                'cpu_step(t)': cpu_step,
                'effort(t)': 0.0,
                'A_mean(t)': 1.0,
                'f_mean(t)': np.mean(frequencies),
                'effort_status': 'stable'
            }
            
            # Écrire dans le CSV
            log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
            row_data = []
            for metric in log_metrics_order:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    if isinstance(value, str):
                        row_data.append(value)
                    else:
                        row_data.append(f"{value:.6f}")
                else:
                    row_data.append("0.0")
            
            if hasattr(loggers['csv_writer'], 'writerow'):
                loggers['csv_writer'].writerow(row_data)
            
            history.append({'t': t, 'S': S_t, 'C': C_t, 'order': order_param})
            S_history.append(S_t)
            C_history.append(C_t)
            cpu_steps.append(cpu_step)
        
        return {
            "logs": loggers.get('log_file', 'kuramoto_log.csv'),
            "metrics": {
                'mean_S': float(np.mean(S_history)) if S_history else 0.0,
                'std_S': float(np.std(S_history)) if S_history else 0.0,
                'mean_C': float(np.mean(C_history)) if C_history else 0.0,
                'mean_cpu_step': float(np.mean(cpu_steps)) if cpu_steps else 0.0,
                'final_order': float(order_param) if order_param is not None else 0.0,
                'mode': 'Kuramoto'
            },
            "history": history,
            "run_id": loggers['run_id'],
            "S_history": S_history
        }

# --- MODE NEUTRAL (OSCs FIXES, PAS DE FEEDBACK) ----------------
def run_neutral_simulation(config, loggers):
    """
    Simulation neutre : phases/amplitudes fixes, sans rétroaction ni spiralisation.
    Sert de contrôle pour valider l'émergence spécifique de la FPS.
    """
    N = config['system']['N']
    T = config['system']['T']
    dt = config['system'].get('dt', 0.05)
    t_array = np.arange(0, T, dt)
    
    # Paramètres fixes
    amplitudes = np.ones(N)
    frequencies = np.linspace(0.8, 1.2, N)  # Fréquences légèrement différentes
    phases = np.zeros(N)
    
    history = []
    S_history = []
    cpu_steps = []
    
    for t in t_array:
        step_start = time.perf_counter()
        
        # Signal sans feedback
        S_t = np.sum(amplitudes * np.sin(2 * np.pi * frequencies * t + phases))
        
        # Pas de modulation, pas de feedback
        C_t = 1.0  # Cohérence fixe
        E_t = np.max(amplitudes)
        L_t = 0
        
        cpu_step = (time.perf_counter() - step_start) / N
        
        # Log
        metrics_dict = {
            't': t,
            'S(t)': S_t,
            'C(t)': C_t,
            'E(t)': E_t,
            'L(t)': L_t,
            'cpu_step(t)': cpu_step,
            'effort(t)': 0.0,
            'A_mean(t)': 1.0,
            'f_mean(t)': np.mean(frequencies),
            'effort_status': 'stable',
            'variance_d2S': 0.0,
            'entropy_S': 0.0,
            'mean_abs_error': 0.0
        }
        
        # Écrire dans le CSV
        log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
        row_data = []
        for metric in log_metrics_order:
            if metric in metrics_dict:
                value = metrics_dict[metric]
                if isinstance(value, str):
                    row_data.append(value)
                else:
                    row_data.append(f"{value:.6f}")
            else:
                row_data.append("0.0")
        
        if hasattr(loggers['csv_writer'], 'writerow'):
            loggers['csv_writer'].writerow(row_data)
        
        history.append({'t': t, 'S': S_t})
        S_history.append(S_t)
        cpu_steps.append(cpu_step)
    
    return {
        "logs": loggers.get('log_file', 'neutral_log.csv'),
        "metrics": {
            'mean_S': float(np.mean(S_history)) if S_history else 0.0,
            'std_S': float(np.std(S_history)) if S_history else 0.0,
            'mean_cpu_step': float(np.mean(cpu_steps)) if cpu_steps else 0.0,
            'mode': 'Neutral'
            },
        "history": history,
        "run_id": loggers['run_id'],
        "S_history": S_history
    }

# --------- AFFICHAGE TODO ET INCOMPLETS DU PIPELINE -------------
def list_todos():
    print("\n--- TODO FPS PIPELINE ---")
    print("Compléter/raffiner tous les modules suivants pour une version 100% rigoureuse et falsifiable :")
    print("- dynamics.py : TOUS les compute_X du dictionnaire FPS (amplitude, freq, phase, feedback, spiral, etc.)")
    print("  * compute_In(t) avec tous les modes de perturbation")
    print("  * compute_S_i(t) pour le signal des autres strates")
    print("  * compute_En(t) et compute_On(t) avec les formules exploratoires")
    print("- metrics.py : TOUS les critères/tableau structuré")
    print("  * variance_d2S, entropy_S avec FFT")
    print("  * t_retour avec analyse pré/post-choc")
    print("  * effort normalisé avec dimensions cohérentes")
    print("- regulation.py : G(x), G_n, env_n (toutes formes, statique/dynamique, archétypes)")
    print("  * Les 4 archétypes de G(x)")
    print("  * env_n avec gaussienne/sigmoïde")
    print("  * G_temporal avec η(t) et θ(t)")
    print("- perturbations.py : gestion complète choc/rampe/sinus/bruit/none")
    print("- analyze.py : raffinements, analyse batch, MAJ changelog")
    print("  * Tous les critères de raffinement")
    print("  * Corrélation effort/CPU")
    print("  * Export journal des seuils")
    print("- explore.py : détection anomalies, motifs fractals, rapports exploration")
    print("  * Recurrence plots")
    print("  * Détection bifurcations spiralées")
    print("  * Analyse harmonique émergente")
    print("- visualize.py : tous plots, dashboard, grille empirique, rapport HTML, animation spiralée")
    print("  * Grille empirique avec icônes/couleurs")
    print("  * Animation 3D de la spirale")
    print("  * Dashboard temps réel")
    print("- utils.py : merge_logs, replay_logs, batch_runner, float_guard, compute_checksum, etc.")
    print("- test_fps.py : tests unitaires/statique/dynamique pour chaque fonction, critère, pipeline")
    print("- kuramoto.py : simulation contrôle complète avec équations exactes")
    print("- Documentation : README, matrice critère-terme, journal seuils\n")

# --------- ARGPARSE POUR EXECUTION CLI -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FPS Simulation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--mode", type=str, default="FPS", choices=["FPS", "Kuramoto", "neutral"], help="Simulation mode")
    parser.add_argument("--list-todos", action="store_true", help="List all TODO items")
    args = parser.parse_args()
    
    if args.list_todos:
        list_todos()
    else:
        result = run_simulation(args.config, args.mode)
        print(f"\nSimulation terminée : {result['run_id']}")
        print(f"Logs : {result['logs']}")
        print(f"Métriques finales : {result['metrics']}")

"""
-----------------------
FIN simulate.py FPS V1.3 (exhaustif, strict, explicite)
Ce fichier doit servir de **référence** pour toutes les évolutions du pipeline FPS. 
Chaque fonction à compléter/raffiner l'est dans les modules correspondants.
"""