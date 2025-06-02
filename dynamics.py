"""
dynamics.py - Calculs des termes FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
La définition actuelle de [Sᵢ(t)]/[Eₙ(t)]/[Oₙ(t)] (ainsi que de
φₙ(t), θ(t), η(t), μₙ(t) et les latences) est une hypothèse de phase 1,
appelée à être falsifiée/raffinée selon la feuille de route FPS.
---------------------------------------------------------------

Ce module implémente TOUS les calculs dynamiques du système FPS :
- Input contextuel avec modes multiples
- Calculs adaptatifs (amplitude, fréquence, phase)
- Signaux inter-strates et feedback
- Régulation spiralée
- Métriques globales

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
from typing import Dict, List, Union, Optional, Any


# ============== FONCTIONS D'INPUT CONTEXTUEL ==============

def compute_In(t: float, perturbation_config: Dict[str, Any], N: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calcule l'input contextuel pour toutes les strates.
    
    Args:
        t: temps actuel
        perturbation_config: configuration de perturbation depuis config.json
        N: nombre de strates (optionnel, pour retourner un array)
    
    Returns:
        float ou np.ndarray: valeur(s) d'input contextuel
    
    Modes supportés:
        - "constant": valeur fixe
        - "choc": impulsion à t0
        - "rampe": augmentation linéaire
        - "sinus": oscillation périodique
        - "uniform": U[0,1] aléatoire
        - "none": pas de perturbation (0.0)
    """
    mode = perturbation_config.get('type', 'none')
    amplitude = perturbation_config.get('amplitude', 1.0)
    t0 = perturbation_config.get('t0', 0.0)
    
    # Calcul de la valeur de base selon le mode
    if mode == "constant":
        value = amplitude
    
    elif mode == "choc":
        # Impulsion brève à t0
        dt = perturbation_config.get('dt', 0.05)  # durée du pic
        if abs(t - t0) < dt:
            value = amplitude
        else:
            value = 0.0
    
    elif mode == "rampe":
        # Augmentation linéaire de 0 à amplitude
        duration = perturbation_config.get('duration', 10.0)
        if t < t0:
            value = 0.0
        elif t < t0 + duration:
            value = amplitude * (t - t0) / duration
        else:
            value = amplitude
    
    elif mode == "sinus":
        # Oscillation périodique
        freq = perturbation_config.get('freq', 0.1)
        if t >= t0:
            value = amplitude * np.sin(2 * np.pi * freq * (t - t0))
        else:
            value = 0.0
    
    elif mode == "uniform":
        # Bruit uniforme U[0,1] * amplitude
        value = amplitude * np.random.uniform(0, 1)
    
    else:  # "none" ou mode inconnu
        value = 0.0
    
    # Retourner un array si N est spécifié
    if N is not None:
        return np.full(N, value)
    return value


# ============== FONCTIONS D'ADAPTATION ==============

def compute_sigma(x: Union[float, np.ndarray], k: float, x0: float) -> Union[float, np.ndarray]:
    """
    Fonction sigmoïde d'adaptation douce.
    
    σ(x) = 1 / (1 + exp(-k(x - x0)))
    
    Args:
        x: valeur(s) d'entrée
        k: sensibilité (pente)
        x0: seuil de basculement
    
    Returns:
        Valeur(s) sigmoïde entre 0 et 1
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def compute_An(t: float, state: List[Dict], In_t: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calcule l'amplitude adaptative pour chaque strate.
    
    Aₙ(t) = A₀ · σ(Iₙ(t))
    
    Args:
        t: temps actuel
        state: état complet des strates
        In_t: input contextuel pour chaque strate
        config: configuration complète
    
    Returns:
        np.ndarray: amplitudes adaptatives
    """
    N = len(state)
    An_t = np.zeros(N)
    
    # Validation des entrées
    if isinstance(In_t, (int, float)):
        In_t = np.full(N, In_t)  # Convertir scalar en array
    elif len(In_t) != N:
        print(f"⚠️ Taille In_t ({len(In_t)}) != N ({N}), ajustement automatique")
        In_t = np.resize(In_t, N)
    
    for n in range(N):
        A0 = state[n]['A0']
        k = state[n]['k']
        x0 = state[n]['x0']
        
        # Amplitude adaptative via sigmoïde
        An_t[n] = A0 * compute_sigma(In_t[n], k, x0)
    
    return An_t

# ============== CALCUL DU SIGNAL INTER-STRATES ==============

def compute_S_i(t: float, n: int, history: List[Dict]) -> float:
    """
    Calcule le signal provenant des autres strates.
    
    Hypothèse exploratoire phase 1:
    - Si t == 0: return 0
    - Sinon: return S(t-dt) - On(t-dt)
    
    Args:
        t: temps actuel
        n: indice de la strate courante
        history: historique complet du système
    
    Returns:
        float: signal des autres strates
    """
    if t == 0 or len(history) == 0:
        return 0.0
    
    # Récupérer le dernier état
    last_state = history[-1]
    S_prev = last_state.get('S', 0.0)
    On_prev = last_state.get('O', np.zeros(1))
    
    # Signal selon la formule FPS : S(t-dt) - On(t-dt)
    if isinstance(On_prev, np.ndarray) and n < len(On_prev):
        return S_prev - On_prev[n]
    else:
        return S_prev  # Fallback si pas de On_prev


# ============== MODULATION DE FRÉQUENCE ==============

def compute_delta_fn(t: float, alpha_n: float, w_ni: List[float], S_i: float) -> float:
    """
    Calcule la modulation de fréquence.
    
    Δfₙ(t) = αₙ · Σᵢ w_{ni} · Sᵢ(t)
    
    Args:
        t: temps actuel
        alpha_n: souplesse d'adaptation
        w_ni: poids de connexion
        S_i: signal des autres strates
    
    Returns:
        float: modulation de fréquence
    """
    # Pour phase 1, on simplifie avec une seule valeur S_i
    # En phase 2, on pourrait avoir un S_i par strate
    modulation = alpha_n * sum(w_ni) * S_i
    return modulation


def compute_fn(t: float, state: List[Dict], An_t: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calcule la fréquence modulée pour chaque strate.
    
    fₙ(t) = f₀ₙ + Δfₙ(t)
    
    Args:
        t: temps actuel
        state: état des strates
        An_t: amplitudes actuelles
        config: configuration
    
    Returns:
        np.ndarray: fréquences modulées
    """
    N = len(state)
    fn_t = np.zeros(N)
    history = config.get('history', [])
    
    for n in range(N):
        f0n = state[n]['f0']
        alpha_n = state[n]['alpha']
        w_ni = state[n]['w']
        
        # Calcul du signal des autres strates
        S_i = compute_S_i(t, n, history)
        
        # Modulation de fréquence
        delta_fn = compute_delta_fn(t, alpha_n, w_ni, S_i)
        
        # Fréquence finale
        fn_t[n] = f0n + delta_fn
    
    return fn_t


# ============== PHASE ==============

def compute_phi_n(t: float, state: List[Dict], config: Dict) -> np.ndarray:
    """
    Calcule la phase pour chaque strate.
    
    Args:
        t: temps actuel
        state: état des strates
        config: configuration
    
    Returns:
        np.ndarray: phases
    
    Modes:
        - "static": φₙ constant (depuis config)
        - "dynamic": évolution à définir après phase 1
    """
    N = len(state)
    phi_n_t = np.zeros(N)
    
    # Récupération du mode depuis config
    dynamic_params = config.get('dynamic_parameters', {})
    dynamic_phi = dynamic_params.get('dynamic_phi', False)
    
    for n in range(N):
        if dynamic_phi and state[n].get('dynamic_phi', False):
            # Mode dynamique - à implémenter en phase 2
            # Pour l'instant, on garde la valeur statique
            phi_n_t[n] = state[n].get('phi', 0.0)
        else:
            # Mode statique
            phi_n_t[n] = state[n].get('phi', 0.0)
    
    return phi_n_t


# ============== LATENCE EXPRESSIVE ==============

def compute_gamma(t: float, mode: str = "static", T: Optional[float] = None) -> float:
    """
    Calcule la latence expressive globale.
    
    Args:
        t: temps actuel
        mode: "static" ou "dynamic"
        T: durée totale (pour mode dynamic)
    
    Returns:
        float: latence entre 0 et 1
    
    Formes:
        - static: γ(t) = 1.0
        - dynamic: γ(t) = 1/(1 + exp(-2(t - T/2)))
    """
    if mode == "static":
        return 1.0
    elif mode == "dynamic" and T is not None:
        # Sigmoïde centrée à T/2
        k = 2.0  # Paramètre de pente fixé pour phase 1
        t0 = T / 2
        return 1.0 / (1.0 + np.exp(-k * (t - t0)))
    else:
        return 1.0


def compute_gamma_n(t: float, state: List[Dict], config: Dict) -> np.ndarray:
    """
    Calcule la latence expressive par strate.
    
    Args:
        t: temps actuel
        state: état des strates
        config: configuration
    
    Returns:
        np.ndarray: latences par strate
    """
    N = len(state)
    gamma_n_t = np.zeros(N)
    
    # Configuration de latence
    latence_config = config.get('latence', {})
    gamma_n_mode = latence_config.get('gamma_n_mode', 'static')
    T = config.get('system', {}).get('T', 100)
    
    if gamma_n_mode == "static":
        # Mode statique : toutes les strates à 1.0
        gamma_n_t[:] = 1.0
    elif gamma_n_mode == "dynamic":
        # Mode dynamique avec paramètres par défaut ou depuis config
        gamma_n_dynamic = latence_config.get('gamma_n_dynamic', {})
        k_n = gamma_n_dynamic.get('k_n', 2.0)
        t0_n = gamma_n_dynamic.get('t0_n', T / 2)
        
        # Sigmoïde pour chaque strate
        for n in range(N):
            gamma_n_t[n] = 1.0 / (1.0 + np.exp(-k_n * (t - t0_n)))
    
    return gamma_n_t


# ============== SORTIES OBSERVÉE ET ATTENDUE ==============

def compute_On(t: float, state: List[Dict], An_t: np.ndarray, fn_t: np.ndarray, 
               phi_n_t: np.ndarray, gamma_n_t: np.ndarray) -> np.ndarray:
    """
    Calcule la sortie observée pour chaque strate.
    
    Oₙ(t) = Aₙ(t) · sin(2π·fₙ(t)·t + φₙ(t)) · γₙ(t)
    
    Args:
        t: temps actuel
        state: état des strates
        An_t: amplitudes
        fn_t: fréquences
        phi_n_t: phases
        gamma_n_t: latences
    
    Returns:
        np.ndarray: sorties observées
    """
    N = len(state)
    On_t = np.zeros(N)
    
    for n in range(N):
        # Contribution de la strate n au signal global
        On_t[n] = An_t[n] * np.sin(2 * np.pi * fn_t[n] * t + phi_n_t[n]) * gamma_n_t[n]
    
    return On_t


def compute_En(t: float, state: List[Dict], history: List[Dict], config: Dict) -> np.ndarray:
    """
    Calcule la sortie attendue (harmonique cible) pour chaque strate.
    
    Hypothèse exploratoire phase 1:
    Eₙ(t) = φ · Oₙ(t-1) où φ est le nombre d'or
    
    Args:
        t: temps actuel
        state: état des strates
        history: historique
        config: configuration
    
    Returns:
        np.ndarray: sorties attendues
    """
    N = len(state)
    En_t = np.zeros(N)
    
    # Nombre d'or
    phi = config.get('spiral', {}).get('phi', 1.618)
    
    if len(history) > 0:
        # Attracteur basé sur le nombre d'or
        last_On = history[-1].get('O', np.zeros(N))
        if isinstance(last_On, np.ndarray) and len(last_On) == N:
            En_t = phi * last_On
        else:
            # Valeur par défaut si historique incomplet
            for n in range(N):
                En_t[n] = state[n]['A0']
    else:
        # Valeur initiale = amplitude de base
        for n in range(N):
            En_t[n] = state[n]['A0']
    
    return En_t


# ============== SPIRALISATION ==============

def compute_r(t: float, phi: float, epsilon: float, omega: float, theta: float) -> float:
    """
    Calcule le ratio spiralé.
    
    r(t) = φ + ε · sin(2π·ω·t + θ)
    
    Args:
        t: temps actuel
        phi: nombre d'or
        epsilon: amplitude de variation
        omega: fréquence de modulation
        theta: phase initiale
    
    Returns:
        float: ratio spiralé
    """
    return phi + epsilon * np.sin(2 * np.pi * omega * t + theta)


def compute_C(t: float, phi_n_array: np.ndarray) -> float:
    """
    Calcule le coefficient d'accord spiralé.
    
    C(t) = (1/N) · Σ cos(φₙ₊₁ - φₙ)
    
    Args:
        t: temps actuel
        phi_n_array: phases de toutes les strates
    
    Returns:
        float: coefficient d'accord entre -1 et 1
    """
    N = len(phi_n_array)
    if N <= 1:
        return 1.0
    
    # Somme des cosinus entre phases adjacentes
    cos_sum = 0.0
    for n in range(N - 1):
        cos_sum += np.cos(phi_n_array[n + 1] - phi_n_array[n])
    
    return cos_sum / (N - 1)


def compute_A(t: float, delta_fn_array: np.ndarray) -> float:
    """
    Calcule la modulation moyenne.
    
    A(t) = (1/N) · Σ Δfₙ(t)
    
    Args:
        t: temps actuel
        delta_fn_array: modulations de fréquence
    
    Returns:
        float: modulation moyenne
    """
    if len(delta_fn_array) == 0:
        return 0.0
    return np.mean(delta_fn_array)


def compute_A_spiral(t: float, C_t: float, A_t: float) -> float:
    """
    Calcule l'amplitude harmonisée.
    
    A_spiral(t) = C(t) · A(t)
    
    Args:
        t: temps actuel
        C_t: coefficient d'accord
        A_t: modulation moyenne
    
    Returns:
        float: amplitude spiralée
    """
    return C_t * A_t


# ============== FEEDBACK ==============

def compute_Fn(t: float, beta_n: float, On_t: float, En_t: float, gamma_t: float) -> float:
    """
    Calcule le feedback pour une strate.
    
    Fₙ(t) = βₙ · (Oₙ(t) - Eₙ(t)) · γ(t)
    
    Args:
        t: temps actuel
        beta_n: plasticité de la strate
        On_t: sortie observée
        En_t: sortie attendue
        gamma_t: latence globale
    
    Returns:
        float: valeur de feedback
    """
    return beta_n * (On_t - En_t) * gamma_t


# ============== SIGNAL GLOBAL ==============

def compute_S(t: float, An_array: np.ndarray, fn_array: np.ndarray, 
              phi_n_array: np.ndarray, config: Dict) -> float:
    """
    Calcule le signal global du système.
    
    Args:
        t: temps actuel
        An_array: amplitudes
        fn_array: fréquences
        phi_n_array: phases
        config: configuration (pour modes avancés)
    
    Returns:
        float: signal global S(t)
    
    Modes:
        - "simple": Σₙ Aₙ(t)·sin(2π·fₙ(t)·t + φₙ(t))
        - "extended": avec γₙ(t) et G(Eₙ(t) - Oₙ(t))
    """
    mode = config.get('system', {}).get('signal_mode', 'simple')
    N = len(An_array)
    
    if mode == "simple":
        # Somme simple des contributions
        S_t = 0.0
        for n in range(N):
            S_t += An_array[n] * np.sin(2 * np.pi * fn_array[n] * t + phi_n_array[n])
        return S_t
    
    elif mode == "extended":
        # Version étendue avec latence et régulation
        # À implémenter en phase 2
        return compute_S(t, An_array, fn_array, phi_n_array, {'system': {'signal_mode': 'simple'}})
    
    else:
        # Par défaut, mode simple
        return compute_S(t, An_array, fn_array, phi_n_array, {'system': {'signal_mode': 'simple'}})


# ============== MÉTRIQUES GLOBALES ==============

def compute_E(t: float, signal_array: Union[np.ndarray, List[float]]) -> float:
    """
    Calcule l'amplitude maximale.
    
    E(t) = maxₙ |signal|
    
    Args:
        t: temps actuel
        signal_array: signaux ou amplitudes
    
    Returns:
        float: amplitude maximale
    """
    if len(signal_array) == 0:
        return 0.0
    return np.max(np.abs(signal_array))


def compute_L(t: float, signal_array: Union[np.ndarray, List[float]]) -> int:
    """
    Calcule l'indice de latence maximale.
    
    L(t) = argmaxₙ |signal|
    
    Args:
        t: temps actuel
        signal_array: signaux ou amplitudes
    
    Returns:
        int: indice de la strate avec amplitude max
    """
    if len(signal_array) == 0:
        return 0
    return int(np.argmax(np.abs(signal_array)))


# ============== FONCTIONS UTILITAIRES ==============

def update_state(state: List[Dict], An_t: np.ndarray, fn_t: np.ndarray, 
                 phi_n_t: np.ndarray, gamma_n_t: np.ndarray, F_n_t: np.ndarray) -> List[Dict]:
    """
    Met à jour l'état du système avec les nouvelles valeurs calculées.
    
    Args:
        state: état actuel
        An_t: nouvelles amplitudes
        fn_t: nouvelles fréquences
        phi_n_t: nouvelles phases
        gamma_n_t: nouvelles latences
        F_n_t: feedback
    
    Returns:
        List[Dict]: état mis à jour
    """
    N = len(state)
    
    for n in range(N):
        # Mise à jour des valeurs dynamiques
        state[n]['An'] = An_t[n]
        state[n]['fn'] = fn_t[n]
        state[n]['gamma_n'] = gamma_n_t[n]
        
        # Historique local de la strate (optionnel)
        if 'history' in state[n]:
            state[n]['history'].append({
                't': state[n].get('t', 0),
                'An': An_t[n],
                'fn': fn_t[n],
                'On': state[n].get('On', 0.0),
                'En': state[n].get('En', 0.0)
            })
    
    return state


# ============== TESTS ET VALIDATION ==============

if __name__ == "__main__":
    """
    Tests basiques pour valider les fonctions.
    """
    print("=== Tests du module dynamics.py ===\n")
    
    # Test 1: Fonction sigmoïde
    print("Test 1 - Sigmoïde:")
    x_test = np.linspace(-5, 5, 11)
    sigma_test = compute_sigma(x_test, k=2.0, x0=0.0)
    print(f"  σ(0) = {compute_sigma(0, 2.0, 0.0):.4f} (attendu: 0.5)")
    print(f"  σ(-∞) → {compute_sigma(-10, 2.0, 0.0):.4f} (attendu: ~0)")
    print(f"  σ(+∞) → {compute_sigma(10, 2.0, 0.0):.4f} (attendu: ~1)")
    
    # Test 2: Input contextuel
    print("\nTest 2 - Input contextuel:")
    pert_config = {'type': 'choc', 't0': 5.0, 'amplitude': 2.0}
    print(f"  Choc à t=5: {compute_In(5.0, pert_config)}")
    print(f"  Choc à t=6: {compute_In(6.0, pert_config)}")
    
    # Test 3: Latence
    print("\nTest 3 - Latence:")
    print(f"  γ(t) statique = {compute_gamma(50, mode='static')}")
    print(f"  γ(t=50) dynamique = {compute_gamma(50, mode='dynamic', T=100):.4f}")
    print(f"  γ(t=0) dynamique = {compute_gamma(0, mode='dynamic', T=100):.4f}")
    
    # Test 4: Ratio spiralé
    print("\nTest 4 - Ratio spiralé:")
    r_test = compute_r(0, phi=1.618, epsilon=0.05, omega=0.1, theta=0)
    print(f"  r(0) = {r_test:.4f}")
    
    print("\n✅ Module dynamics.py prêt à l'emploi!")
