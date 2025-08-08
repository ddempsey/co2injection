import numpy as np
import os,sys
try:
    from co2lpm.parameters import *
except ImportError:
    # Add the next directory up to the path if co2lpm not in it
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from co2lpm.parameters import *
from co2lpm.model import LumpedParameterModel

def test_no_extraction_means_no_change():
    """If fq = 0 (100 % reinjection) the reservoir stays in steady state."""
    res = ReservoirParams()
    op  = OperationParams(q0=0.8, fq=0.0, fC=0.0)          # 100 % reinjection
    mdl = LumpedParameterModel(res, op)
    out = mdl.solve((0.0, 5 * 365.25 * 24 * 3600), n_eval=16)

    # pressure flat
    assert np.allclose(out.pressure, out.pressure[0]), "pressures don't match"

    # concentration flat (α = β C₀ by construction)
    assert np.allclose(out.concentration, mdl.chem.C0), "concentrations don't match"

def test_pressure_decline_matches_analytic():
    """Compare P(t) to closed-form eq (9)."""
    mdl = LumpedParameterModel()
    t = np.linspace(0.0, 3 * mdl.op.tp, 11)
    P_expected = mdl._pressure(t)
    P_numeric  = np.array([1500000.        , 1188981.86481806,  958573.96331283,
        787883.59168872,  661433.05429464,  567756.19217812,
        498358.6658659 ,  446947.71390358,  408861.5439473 ,
        380646.6152877 ,  359744.48204144])
    assert np.allclose(P_numeric, P_expected), "pressures don't match"

def test_all():
    test_no_extraction_means_no_change()
    test_pressure_decline_matches_analytic()
    print("test_co2lpm_model passed all tests")

if __name__=="__main__":
    test_all()
