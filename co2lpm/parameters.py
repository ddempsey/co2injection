"""
Dataclass containers for all physical and operational inputs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------
# Reservoir properties
# ---------------------------------------------------------------
@dataclass(frozen=True)
class ReservoirParams:
    volume:   float = 1.0e11   # m³
    porosity: float = 0.10     # –
    Kup:      float = 3.0e-4   # m³ s⁻¹ Pa⁻¹
    Kout:     float = 3.0e-4   # m³ s⁻¹ Pa⁻¹
    Pup:      float = 3.0e6    # Pa
    Pref:     float = 10.0e6   # Pa
    Tref:     float = 280.0    # °C
    rho_w:    float = 1000.0   # kg m⁻³

    # validation
    def __post_init__(self):  # type: ignore[override]
        if not (0.0 < self.porosity <= 1.0):
            raise ValueError("porosity must be in (0, 1]")
        for name in ("volume", "Kup", "Kout", "Pup", "Pref", "rho_w"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    # convenience
    @property
    def mass_liquid(self) -> float:
        return self.volume * self.porosity * self.rho_w

    @property
    def K_total(self) -> float:
        return self.Kup + self.Kout


# ---------------------------------------------------------------
# Operational controls
# ---------------------------------------------------------------
@dataclass(frozen=True)
class OperationParams:
    q0:  float = 0.8           # kg s⁻¹
    fq:  float = 1.0           # –
    fC:  float = 1.0           # –
    tp:  float = 10.0 * 365.25 * 24 * 3600  # s
    tau: float = 0.0           # s

    def __post_init__(self):  # type: ignore[override]
        if not (0.0 <= self.fq):
            raise ValueError("fq must be in [0, inf]")
        if not (0.0 <= self.fC <= 1.0):
            raise ValueError("fC must be in [0, 1]")
        if self.tp <= 0.0:
            raise ValueError("tp must be positive")
        if self.tau < 0.0:
            raise ValueError("tau cannot be negative")

    @property
    def q_eff(self) -> float:
        return (1.0 - self.fq) * self.q0


# ---------------------------------------------------------------
# CO₂ chemistry
# ---------------------------------------------------------------
@dataclass(frozen=True)
class ChemistryParams:
    C0:      float = 0.03      # kg/kg
    kappa:   float = 0.0       # s⁻¹
    dCsdP:   float | None = None  # Pa⁻¹

    _DEFAULT_SLOPE: Final[float] = 4.0e-9

    def __post_init__(self):  # type: ignore[override]
        if self.C0 < 0.0:
            raise ValueError("C0 must be non-negative")

    @property
    def solubility_slope(self) -> float:
        return self.dCsdP if self.dCsdP is not None else self._DEFAULT_SLOPE
