from __future__ import annotations
from typing import Final

# Appendix Eq (28): linear slope vs T (MPa⁻¹); divide by 1e6 → Pa⁻¹
_A0, _A1, _A2 = 4.5848e-3, -3.3141e-5, 1.3104e-7
_P0: Final[float] = 1.5e6  # Pa, reference where C=0.01
_C00: Final[float] = 0.01  # kg/kg at P=1.5 MPa

def solubility_slope_vs_T(T: float) -> float:
    """Return dC/dP (Pa⁻¹) using the quadratic fit in the appendix."""
    return (_A0 + _A1*T + _A2*T*T) / 1.0e6

def solubility_linear(P: float, Pref: float, T: float, dCsdP: float | None) -> float:
    """
    C_s(P) = dC/dP * ((Pref + P) - 1.5 MPa) + 0.01  (kg/kg)
    If dCsdP is None we compute it from T.
    """
    slope = dCsdP if dCsdP is not None else solubility_slope_vs_T(T)
    return slope * ((Pref + P) - _P0) + _C00
