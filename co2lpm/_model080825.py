"""co2_lpm.model
================
Core *lumped-parameter* physics translated from the original *lpm.py* into
a small, testable module.  This is **step 2** of the refactor plan:
https://chat.openai.com/share/b1ece14c-6a5e-452c-8d22-bcb2b6a57c63  (local file
``build_plan.txt``).

The initial scope keeps things intentionally *minimal* so that we can grow the
API under test-coverage instead of copy-pasting 1 500 lines of legacy code at
once.  Concretely this version implements:

* Pressure decline under constant net extraction (τ=0), eq (9).
* Mass-fluxes (up- and out-flow) that follow the pressure solution.
* A first-order ODE for dissolved CO₂ concentration (low-gas, no degassing).
* A thin `solve()` helper returning NumPy arrays so that notebooks / scripts
  don’t need to touch SciPy internals.

Later iterations will add:
* Degassing & solubility-limited regimes.
* Delay-differential dynamics (τ > 0) via `ddeint`.
* Analytic gamma-function solution for regression tests.

The public surface is therefore tiny:

```python
from co2_lpm.model import LumpedParameterModel, StateArrays
```
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .parameters import ChemistryParams, OperationParams, ReservoirParams

__all__ = [
    "StateArrays",
    "LumpedParameterModel",
]

_SECOND: float = 1.0  # alias so units are obvious in code below
_YEAR: float = 365.25 * 24 * 3600 * _SECOND


@dataclass(frozen=True)
class StateArrays:
    """Container returned by :pymeth:`LumpedParameterModel.solve`."""

    t: np.ndarray  # s
    pressure: np.ndarray  # Pa (relative to hydrostatic reference)
    concentration: np.ndarray  # kg CO₂ / kg H₂O
    qup: np.ndarray  # kg s⁻¹ (negative = into reservoir)
    qout: np.ndarray  # kg s⁻¹ (positive = leaving reservoir)


class LumpedParameterModel:
    """Low-gas reservoir with *instantaneous* pressure response (τ=0).

    The implementation purposefully stays close to eqns (5)–(10) & (20) of the
    addendum so reviewers can follow symbol-for-symbol.  All derived numerical
    values are cached on `self` to avoid recomputation inside tight ODE loops.
    """

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        res: ReservoirParams | None = None,
        op: OperationParams | None = None,
        chem: ChemistryParams | None = None,
    ) -> None:
        self.res = res or ReservoirParams()
        self.op = op or OperationParams()
        self.chem = chem or ChemistryParams()

        # --- shorthand symbols ------------------------------------------------
        self.K: float = self.res.K_total  # m³ s⁻¹ Pa⁻¹
        self.q0c: float = self.res.Kup * self.res.Pup  # kg s⁻¹, “critical”
        self.q0: float = self.op.q0 * self.q0c  # actual extraction rate (kg s⁻¹)

        # Effective extraction after reinjection (f_q ∈ [0,1])
        self.q_eff: float = self.op.fq * self.q0

        # Initial over-pressure relative to hydrostatic (Pa)
        self.P0: float = self.res.Kup * self.res.Pup / self.K

        # Characteristic pressure-relaxation time (s) – eq (8)
        self.tp: float = self.op.tp

        # Characteristic *mass* flushing time (s)
        self.t_flush: float = self.res.mass_liquid / max(self.q_eff, 1e-12)

        # Coefficients for the simple first-order concentration ODE
        #   dC/dt = α − β C(t)
        # Choose β as inverse flush-time so solution decays on similar scale
        self.beta: float = 1.0 / self.t_flush  # s⁻¹
        # Pick α such that *initial* concentration is at steady-state; this
        # ensures unit tests with q_eff=0 keep C(t)=C₀ exactly.
        self.alpha: float = self.beta * self.chem.C0  # kg CO₂ kg⁻¹ s⁻¹

        # Validate regime (τ = delay not yet implemented)
        if self.op.tau > 0.0:
            raise NotImplementedError("Delay-differential (τ>0) not yet ported")

    # ------------------------------------------------------------------
    # private helpers (vectorised)
    # ------------------------------------------------------------------
    def _pressure(self, t: np.ndarray) -> np.ndarray:
        """Eq (9) – exponential decline to *P∞*."""
        P_inf = self.P0 - self.q_eff / self.K
        return self.P0 - (self.P0 - P_inf) * (1.0 - np.exp(-t / self.tp))

    # Mass fluxes follow Darcy-like proportionality (eq 18-19)
    def _qup(self, P: np.ndarray) -> np.ndarray:  # negative means into res.
        return -self.res.Kup * (self.res.Pup - P)

    def _qout(self, P: np.ndarray) -> np.ndarray:  # positive (leaving)
        return self.res.Kout * P

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def rhs(self, t: float, C: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Derivative function dC/dt for *low-gas* regime (no degassing)."""
        # Note: could later add γ, δ terms for solubility-limited case.
        return self.alpha - self.beta * C

    def solve(self, t_span: Tuple[float, float], n_eval: int = 1001) -> StateArrays:
        """Integrate concentration from *t0* → *t1* (seconds).

        Parameters
        ----------
        t_span
            `(t0, t1)` tuple in seconds.
        n_eval
            Number of evaluation points (log-spaced if span > 1 year so that
            long integrations keep relative resolution).
        """
        t0, t1 = t_span
        if t1 <= t0:
            raise ValueError("t_span must satisfy t1 > t0")

        # Choose log-spacing for decade-long runs so that early transients keep
        # resolution; else uniform spacing is fine.
        if (t1 - t0) > _YEAR:
            t_eval = np.logspace(np.log10(max(t0, 1e-6)), np.log10(t1), n_eval)
        else:
            t_eval = np.linspace(t0, t1, n_eval)

        sol = solve_ivp(self.rhs, t_span, y0=[self.chem.C0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
        if not sol.success:
            raise RuntimeError(sol.message)

        C = sol.y[0]
        P = self._pressure(t_eval)
        qup = self._qup(P)
        qout = self._qout(P)

        return StateArrays(t=t_eval, pressure=P, concentration=C, qup=qup, qout=qout)

    # Convenience property so unit tests can grab steady-state directly
    @property
    def C_inf(self) -> float:
        """Steady-state concentration for current *α, β*."""
        return self.alpha / self.beta
