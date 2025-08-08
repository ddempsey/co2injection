from __future__ import annotations
import numpy as np
from .parameters import ReservoirParams, OperationParams, ChemistryParams

def emissions(state, res: ReservoirParams, op: OperationParams, chem: ChemistryParams):
    """
    Compute emissions partitions following the manuscript (units track your inputs).
    Returns dict with Q_dg, Q_out, Q_pp, Q_total, Q_nat0, Q_nat_inf, CsP.
    """
    t = state.t
    C = state.concentration
    K = res.K_total
    q0c = res.Kup * res.Pup
    q0 = op.q0 * q0c

    # degassing term (only if solubility slope present)
    dCsdP = chem.dCsdP if chem.dCsdP is not None else 0.0
    Q_dg = op.fq * q0 * res.mass_liquid * dCsdP / (K * op.tp) * np.exp(-t / op.tp)

    # Cs at final pressure (needed for liquid bookkeeping term)
    P0 = res.Kup * res.Pup / K
    P_inf = P0 - (1.0 - op.fq) * q0 / K
    if dCsdP == 0.0:
        CsP = chem.C0
    else:
        # same linearised formula used in the utils helper
        CsP = dCsdP * ((res.Pref + P_inf) - 1.5e6) + 0.01

    # emissions (field, plant, total)
    Q_out = state.qout * C + Q_dg - state.qup * (chem.C0 - CsP)
    Q_pp = op.fC * q0 * C
    Q_total = Q_out + Q_pp

    # baselines for plots
    Q_nat0 = res.Kup * res.Kout * res.Pup / K * chem.C0
    Q_nat_inf = res.Kup * (res.Kout * res.Pup + op.fq * q0) / K * chem.C0

    return dict(Q_dg=Q_dg, Q_out=Q_out, Q_pp=Q_pp, Q_total=Q_total,
                Q_nat0=Q_nat0, Q_nat_inf=Q_nat_inf, CsP=CsP)
