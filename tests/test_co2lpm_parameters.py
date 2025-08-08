import pytest
import os,sys
try:
    from co2lpm.parameters import *
except ImportError:
    # Add the next directory up to the path if co2lpm not in it
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from co2lpm.parameters import *


def test_reservoir_defaults():
    r = ReservoirParams()
    assert r.K_total == pytest.approx(r.Kup + r.Kout)
    assert r.mass_liquid > 0
    with pytest.raises(ValueError):
        ReservoirParams(porosity=1.5)  # illegal


def test_operation_net_flow():
    op = OperationParams(q0=10.0, fq=0.25)
    assert op.q_eff == pytest.approx(7.5)
    with pytest.raises(ValueError):
        OperationParams(q0=1.0, fq=-0.5, fC=0.5)
    with pytest.raises(ValueError):
        OperationParams(tp=-1.0)  # negative tp


def test_chemistry_slope_override():
    chem = ChemistryParams(dCsdP=9.9e-9)
    assert chem.solubility_slope == pytest.approx(9.9e-9)
    default = ChemistryParams()
    assert default.solubility_slope > 0
    with pytest.raises(ValueError):
        ChemistryParams(C0=-0.1)

def test_all():
    test_reservoir_defaults()
    test_operation_net_flow()
    test_chemistry_slope_override()
    print("test_co2lpm_parameters passed all tests")

if __name__=="__main__":
    test_all()
