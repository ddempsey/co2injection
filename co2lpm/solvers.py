from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
import mpmath as mp

# ------------------------------
# Pressure helper
# ------------------------------
def pressure_exp(t: np.ndarray, P0: float, K: float, q_eff: float, tp: float) -> np.ndarray:
    """Eq (9): P(t) = P0 − (q_eff/K) * (1 − e^{−t/tp})."""
    return P0 - (q_eff / K) * (1.0 - np.exp(-t / tp))

# ------------------------------
# ODE (no delay):  dC/dt = α − β C − γ C e^{−t/tp} − δ e^{−t/tp}
# ------------------------------
def rhs_nodelay(t: float, C: np.ndarray, alpha: float, beta: float, gamma: float, delta: float, tp: float) -> np.ndarray:
    e = np.exp(-t / tp)
    return alpha - beta * C - gamma * C * e - delta * e

def integrate_ode(t_span, t_eval, C0, alpha, beta, gamma, delta, tp):
    sol = solve_ivp(rhs_nodelay, t_span, [C0], t_eval=t_eval,
                    args=(alpha, beta, gamma, delta, tp),
                    rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0]

# ------------------------------
# Exact analytic (gamma‑function) evaluator for nodelay ODE
# Mirrors lpm.Cf / C2f (see manuscript §3.1)
# ------------------------------
def gamma_exact(t_array: np.ndarray, C0: float, alpha: float, beta: float, gamma: float, delta: float, tp: float) -> np.ndarray:
    b = beta * tp
    c = gamma * tp
    a = (alpha + beta * delta / gamma) * tp if gamma != 0.0 else (alpha * tp)

    if c <= 0:
        # out of validity of this closed form; fall back to numeric
        return np.nan * np.ones_like(t_array)

    X0 = C0 + (delta / gamma if gamma != 0 else 0.0)
    K = a * (c ** b)
    G0 = mp.gammainc(-b, c, mp.inf)  # upper Γ(-b, c)
    const = mp.e ** (-c) * X0

    out = np.empty_like(t_array, dtype=float)
    for i, t in enumerate(t_array):
        if t == 0.0:
            out[i] = float(X0)
            continue
        Gt = mp.gammainc(-b, c * mp.e ** (-t / tp), mp.inf)  # upper Γ(-b, c e^{-t/tp})
        mu_inv = np.exp(-b * t / tp + c * np.exp(-t / tp))
        out[i] = float(mu_inv * (const + K * (Gt - G0)))
    return out - (delta / gamma if gamma != 0 else 0.0)

def gamma_late_approx(t_array: np.ndarray, C0: float, alpha: float, beta: float, gamma: float, delta: float, tp: float) -> np.ndarray:
    b = beta * tp
    c = gamma * tp
    a = (alpha + beta * delta / gamma) * tp if gamma != 0.0 else (alpha * tp)
    X0 = C0 + (delta / gamma if gamma != 0 else 0.0)

    B = -a * c / (b * (1.0 - b))
    G_lower = mp.gammainc(-b, 0, c)
    A = X0 * mp.e ** (-c) + a * (c ** b) * G_lower
    out = (a / b) + A * np.exp(-b * t_array / tp) + B * np.exp(-t_array / tp) - (delta / gamma if gamma != 0 else 0.0)
    return np.asarray(out, dtype=float)

# ------------------------------
# DDE:  dC/dt = α − β C(t) + γ C(t−τ)
# ------------------------------
def dde_solve(t_eval: np.ndarray, C0: float, alpha: float, beta: float, gamma: float, tau: float) -> np.ndarray:
    try:
        from ddeint import ddeint  # optional dep
    except Exception as e:
        raise RuntimeError("ddeint is required for tau > 0. Install `ddeint`.") from e

    def f(Cfunc, t):
        Ct = Cfunc(t)
        Ct_tau = Cfunc(t - tau) if t > tau else C0
        return alpha - beta * Ct + gamma * Ct_tau

    hist = (lambda t: C0)
    C = ddeint(f, hist, t_eval)[:, 0]
    return np.asarray(C, dtype=float)

# Characteristic late‑time root (for analysis / tests)
def dde_lambda1(beta: float, gamma: float, tau: float) -> float:
    # λ solves λ + beta − gamma e^{−λ τ} = 0  →  λ = −beta + W(gamma τ e^{beta τ})/τ
    z = gamma * tau * np.exp(beta * tau)
    lam = -beta + lambertw(z, k=0).real / tau
    return float(lam)
