from __future__ import annotations
from .parameters import ReservoirParams, OperationParams, ChemistryParams
from .utils import solubility_slope_vs_T

def high_gas():
    """Solubility‑limited, Ohaaki‑like."""
    res = ReservoirParams(volume=1.0e11, porosity=0.10, Kup=1.0e-4/2, Kout=1.0e-4/2,
                          Pup=1.2e6, Pref=6.0e6, Tref=260.0, rho_w=1000.0)
    op  = OperationParams(q0=1.5, fq=0.40, fC=1.00, tp=10.0*365.25*24*3600, tau=0.0)
    dCsdP = solubility_slope_vs_T(res.Tref)
    chem = ChemistryParams(C0=0.03, kappa=0.0, dCsdP=dCsdP)
    return res, op, chem

def low_gas():
    """Wairakei‑like low‑gas with capture toggles."""
    res = ReservoirParams(volume=1.0e11, porosity=0.10, Kup=1.0e-3/2, Kout=1.0e-3/2,
                          Pup=1.2e6, Pref=6.0e6, Tref=260.0, rho_w=1000.0)
    op  = OperationParams(q0=1.5, fq=0.60, fC=1.00, tp=10.0*365.25*24*3600, tau=0.0)
    chem = ChemistryParams(C0=0.003, kappa=0.0, dCsdP=None)
    return res, op, chem

def delay_demo():
    """Return a concentrating example with τ > 0."""
    res = ReservoirParams()
    op  = OperationParams(q0=1.0, fq=0.30, fC=0.20, tp=10.0*365.25*24*3600, tau=3.0*365.25*24*3600)
    chem = ChemistryParams(C0=0.003, kappa=0.0, dCsdP=None)
    return res, op, chem

