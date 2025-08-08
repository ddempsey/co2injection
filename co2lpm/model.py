from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .parameters import ChemistryParams, OperationParams, ReservoirParams
from .solvers import pressure_exp, integrate_ode, gamma_exact, dde_solve
from .utils import solubility_linear

__all__ = ["StateArrays", "LumpedParameterModel"]

_SECOND: float = 1.0
_YEAR: float = 365.25 * 24 * 3600 * _SECOND

@dataclass(frozen=True)
class StateArrays:
    t: np.ndarray            # s
    pressure: np.ndarray     # Pa (relative to hydrostatic)
    concentration: np.ndarray  # kg CO₂ / kg H₂O
    qup: np.ndarray          # kg s⁻¹ (negative = into reservoir)
    qout: np.ndarray         # kg s⁻¹ (positive = leaving reservoir)

class LumpedParameterModel:
    """
    General LPM capable of:
      • low‑gas (no degassing) when chem.dCsdP is None
      • solubility‑limited degassing when chem.dCsdP is provided
      • optional delay τ (>0): DDE dC/dt = α − β C(t) + γ C(t−τ)
    """

    def __init__(
        self,
        res: ReservoirParams | None = None,
        op: OperationParams | None = None,
        chem: ChemistryParams | None = None,
    ) -> None:
        self.res = res or ReservoirParams()
        self.op = op or OperationParams()
        self.chem = chem or ChemistryParams()

        # Short-hands
        self.K: float = self.res.K_total
        self.q0c: float = self.res.Kup * self.res.Pup       # critical rate
        self.q0: float = self.op.q0 * self.q0c              # actual rate (kg/s)
        self.q_eff: float = self.op.fq * self.q0    # net extraction
        self.P0: float = self.res.Kup * self.res.Pup / self.K
        self.P_inf: float = self.P0 - self.q_eff / self.K

        # Solubility-limited?
        if self.chem.dCsdP is None:
            self.C0 = self.chem.C0
            self.CsP = self.chem.C0
            self.dCsdP = 0.0
        else:
            self.dCsdP = self.chem.dCsdP
            self.C0  = solubility_linear(self.P0,  self.res.Pref, self.res.Tref, self.dCsdP)
            self.CsP = solubility_linear(self.P_inf, self.res.Pref, self.res.Tref, self.dCsdP)

        # Time scales (match symbols in manuscript / legacy LPM)
        M = self.res.mass_liquid
        fq, fC = self.op.fq, self.op.fC
        q0, q0c = self.q0, self.q0c
        Kup, Kout, K, Pup = self.res.Kup, self.res.Kout, self.K, self.res.Pup
        kappa, tp = self.chem.kappa, self.op.tp

        # Water residence times
        tq = M / max(fq * q0, 1e-30)
        tb = M / max(fC * q0 + kappa + Kout / K * (q0c - fq * q0), 1e-30)
        ta = M / max(kappa + (self.CsP / self.C0) * Kup / K * (Pup * Kout + fq * q0), 1e-30)

        # ODE coefficients (nodelay; §3.1)
        self.alpha = self.C0 / ta
        self.beta  = 1.0 / tb
        self.gamma = Kout / K / tq
        self.delta = (Kup * self.CsP / K + self.dCsdP * M / (K * tp)) / tq if self.dCsdP > 0 else 0.0

        # DDE coefficients when τ > 0 (override gamma/delta definition)
        if self.op.tau > 0.0:
            # dC/dt = α − β C(t) + γ C(t−τ)
            self.gamma = (1.0 - fC) * q0 / M
            self.delta = self.op.tau  # just stored for reference

    # Derived fields for plotting/post‑proc
    def _pressure(self, t: np.ndarray) -> np.ndarray:
        return pressure_exp(t, self.P0, self.K, self.q_eff, self.op.tp)

    def _qup(self, P: np.ndarray) -> np.ndarray:    # negative into reservoir
        return -self.res.Kup * (self.res.Pup - P)

    def _qout(self, P: np.ndarray) -> np.ndarray:   # positive leaving
        return  self.res.Kout * P

    @property
    def C_inf(self) -> float:
        """Steady‑state for the *no‑delay* ODE (for sanity checks)."""
        return self.alpha / self.beta

    def solve(self, t_span: Tuple[float, float], n_eval: int = 1001) -> StateArrays:
        t0, t1 = t_span
        if t1 <= t0:
            raise ValueError("t_span must satisfy t1 > t0")
        if (t1 - t0) > _YEAR:
            t_eval = np.logspace(np.log10(max(t0, 1e-6)), np.log10(t1), n_eval)
        else:
            t_eval = np.linspace(t0, t1, n_eval)

        # Concentration
        if self.op.tau > 0.0:
            C = dde_solve(t_eval, self.C0, self.alpha, self.beta, self.gamma, self.op.tau)
        else:
            # Try exact evaluator; if it returns NaNs, fall back to numeric ODE
            C_exact = gamma_exact(t_eval, self.C0, self.alpha, self.beta, self.gamma, self.delta, self.op.tp)
            if np.isnan(C_exact).any():
                C = integrate_ode((t0, t1), t_eval, self.C0, self.alpha, self.beta, self.gamma, self.delta, self.op.tp)
            else:
                C = C_exact.astype(float)

        # Pressure & fluxes
        P = self._pressure(t_eval)
        qup  = self._qup(P)
        qout = self._qout(P)

        return StateArrays(t=t_eval, pressure=P, concentration=C, qup=qup, qout=qout)
