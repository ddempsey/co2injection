"""CO2 injection lumped-parameter model package."""
from .parameters import ReservoirParams, OperationParams, ChemistryParams
from .model import LumpedParameterModel, StateArrays

__all__ = [
    "ReservoirParams",
    "OperationParams",
    "ChemistryParams",
    "LumpedParameterModel",
    "StateArrays",
]
__version__ = "0.1.0"
