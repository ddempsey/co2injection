#!/usr/bin/env python3
"""
Regenerate manuscript simulation figures.

Usage:
  python scripts/make_figures.py                 # build all figures
  python scripts/make_figures.py --only fig1     # build just fig1
  python scripts/make_figures.py --list          # list figure names
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from co2lpm.parameters import ReservoirParams, OperationParams, ChemistryParams
from co2lpm.model import LumpedParameterModel
from co2lpm.scenarios import high_gas, low_gas, delay_demo
from co2lpm.postproc import emissions
from co2lpm.plotting import figure_timeseries

_SECOND = 1.0
_YEAR = 365.25 * 24 * 3600 * _SECOND
FIGDIR = Path("figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Figure 1 – Pressure path
# -----------------------
def fig1_pressure(outfile: Path | None = None):
    """
    Pressure response P(t) for a nominal case, with annotated P0 and P∞.
    """
    res = ReservoirParams(Kup=1e-3, Kout=1e-3, Pup=1.2e6, Pref=6e6, Tref=260.0)
    op = OperationParams(q0=1.0, fq=1.0, fC=1.0, tp=8.0 * _YEAR, tau=0.0)
    chem = ChemistryParams(C0=0.01, dCsdP=None)

    f,ax=plt.subplots(1,1,figsize=(5.0, 3.2))

    Ps=[]
    fqs=[0.6, 0.8, 1.0, 1.2, 1.4]
    for fq in fqs:
        op = OperationParams(q0=1.0, fq=fq, fC=1.0, tp=10.0 * _YEAR, tau=0.0)
        m = LumpedParameterModel(res, op, chem)
        st = m.solve((0.0, 30.0 * _YEAR), n_eval=800)
        t_years = st.t / _YEAR
        P = st.pressure / 1e6  # MPa

        ax.plot(t_years, P, 'k-')
        Ps.append(1*P[-1])
    
    ax_=ax.twinx()
    ax_.set_ylim(ax.get_ylim())
    ax_.set_yticks(Ps)
    ax.set_xlim(0,30)
    ax_.set_yticklabels([f'{fq:2.1f}'+r'×$q_{0,c}$' for fq in fqs])
    # plt.axhline(m.P0 / 1e6, ls="--", c="k", lw=1, label="P₀")
    ax.axhline(0, ls=":", c="k", lw=1, alpha=0.5)
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("Pressure [MPa]")
    ax.legend()
    plt.tight_layout()
    plt.show()
    # _save(outfile or FIGDIR / "nzgw25_fig1_pressure.png")


# -----------------------------------------
# Figure 2 – Analytic vs numeric approximation
# -----------------------------------------
def fig2_approx(outfile: Path | None = None):
    """
    Compare exact (gamma-form) vs numeric ODE for C(t) under gentle params.
    """
    # Gentle parameters to avoid stiffness; no delay; include a small degassing term
    res = ReservoirParams(Kup=1e-3, Kout=1e-3, Pup=1.2e6, Pref=6e6, Tref=260.0)
    op = OperationParams(q0=1.2, fq=0.4, fC=1.0, tp=6.0 * _YEAR, tau=0.0)
    chem = ChemistryParams()  # linearised solubility slope

    m = LumpedParameterModel(res, op, chem)
    st = m.solve((0.0, 1000.0 * _YEAR), n_eval=600)

    # Re-run with numeric fallback by slightly perturbing gamma to invalidate closed form
    from co2lpm.solvers import integrate_ode
    t = st.t
    C_num = integrate_ode((t[0], t[-1]), t, chem.C0, m.alpha, m.beta, m.gamma * 0.9999, m.delta, op.tp)

    f,ax=plt.subplots(1,1,figsize=(5.0, 3.2))
    ax.plot(t / _YEAR, 100 * st.concentration, 'k-', lw=2, label="exact")
    ax.plot(t / _YEAR, 100 * C_num, "k--", label="approx.")
    ax.axhline(100*m.C0, color='k', alpha=0.5, linestyle=':', label='$C_0$')
    ax.axhline(100*m.C_inf, color='k', alpha=0.5, linestyle='--', label='$C_\infty$')
    ax.set_xlim([0.1,900])
    ax.set_xlabel("time [yr]")
    ax.set_xscale('log')
    ax.set_ylabel("CO$_2$ [wt %]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # _save(outfile or FIGDIR / "nzgw25_fig2_approx.png")


# -----------------------------------------
# Figure 4 – Emissions partitions (single case)
# -----------------------------------------
def fig4_emissions(outfile: Path | None = None):
    """
    Emissions split into field, plant, total with natural baselines.
    """
    res, op, chem = high_gas()  # solubility-limited case
    m = LumpedParameterModel(res, op, chem)
    st = m.solve((0.0, 40.0 * _YEAR), n_eval=600)
    e = emissions(st, res, op, chem)

    fig = figure_timeseries(st, emissions=e)
    fig.suptitle("High-gas: pressure, concentration, and emissions", y=1.02, fontsize=10)
    plt.tight_layout()
    _save(outfile or FIGDIR / "nzgw25_fig4_emissions.png")


# -----------------------------------------
# Figure 5 – Scenario panel (low-gas vs high-gas)
# -----------------------------------------
def fig5_scenarios(outfile: Path | None = None):
    """
    Side-by-side scenarios: low-gas vs high-gas, emissions trajectories.
    """
    resA, opA, chemA = low_gas()
    resB, opB, chemB = high_gas()

    mA = LumpedParameterModel(resA, opA, chemA)
    mB = LumpedParameterModel(resB, opB, chemB)

    tspan = (0.0, 40.0 * _YEAR)
    stA = mA.solve(tspan, n_eval=600)
    stB = mB.solve(tspan, n_eval=600)
    eA = emissions(stA, resA, opA, chemA)
    eB = emissions(stB, resB, opB, chemB)

    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, st, e, title in ((axs[0], stA, eA, "Low-gas"),
                             (axs[1], stB, eB, "High-gas")):
        t_years = st.t / _YEAR
        ax.plot(t_years, e["Q_out"] / 1e6 * _YEAR, label="field")
        ax.plot(t_years, e["Q_pp"] / 1e6 * _YEAR, label="plant")
        ax.plot(t_years, e["Q_total"] / 1e6 * _YEAR, label="total")
        ax.axhline(e["Q_nat0"] / 1e6 * _YEAR, ls=":", c="k", label="nat₀")
        ax.axhline(e["Q_nat_inf"] / 1e6 * _YEAR, ls="--", c="k", label="nat∞")
        ax.set_title(title)
        ax.set_xlabel("time [yr]")
    axs[0].set_ylabel("Emissions [kt/yr]")
    axs[1].legend(loc="upper right")
    fig.tight_layout()
    _save(outfile or FIGDIR / "nzgw25_fig5_scenarios.png")


# -----------------------------------------
# Figure 6 – Policy toggles (capture fraction fC)
# -----------------------------------------
def fig6_emissions(outfile: Path | None = None):
    """
    Sensitivity to plant-capture fraction fC in a low-gas system.
    """
    res, op, chem = low_gas()
    fC_values = [0.0, 0.5, 1.0]

    tspan = (0.0, 40.0 * _YEAR)
    plt.figure(figsize=(5.0, 3.2))
    for fC in fC_values:
        op_mod = OperationParams(q0=op.q0, fq=op.fq, fC=fC, tp=op.tp, tau=op.tau)
        m = LumpedParameterModel(res, op_mod, chem)
        st = m.solve(tspan, n_eval=500)
        e = emissions(st, res, op_mod, chem)
        plt.plot(st.t / _YEAR, e["Q_total"] / 1e6 * _YEAR, label=f"fC={fC:g}")

    plt.axhline(emissions(m.solve(tspan), res, op, chem)["Q_nat_inf"] / 1e6 * _YEAR,
                ls="--", c="k", label="nat∞ (baseline)")
    plt.xlabel("time [yr]")
    plt.ylabel("Total emissions [kt/yr]")
    plt.legend()
    plt.tight_layout()
    _save(outfile or FIGDIR / "nzgw25_fig6_emissions.png")


# -----------------------------------------
# Figure 7 – Delay effects (τ > 0)
# -----------------------------------------
def fig7_emissions(outfile: Path | None = None):
    """
    Delay differential response of concentration/emissions for τ > 0.
    Requires `ddeint` installed.
    """
    res, op, chem = delay_demo()
    m = LumpedParameterModel(res, op, chem)
    st = m.solve((0.0, 40.0 * _YEAR), n_eval=600)
    e = emissions(st, res, op, chem)

    fig = figure_timeseries(st, emissions=e)
    fig.suptitle("Delay case (τ > 0): oscillatory approach to steady state", y=1.02, fontsize=10)
    plt.tight_layout()
    _save(outfile or FIGDIR / "nzgw25_fig7_emissions.png")


# -----------------------
# Helpers / CLI
# -----------------------
def _save(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"wrote {path}")

FIG_FUNCS = {
    "fig1": fig1_pressure,
    "fig2": fig2_approx,
    "fig4": fig4_emissions,
    "fig5": fig5_scenarios,
    "fig6": fig6_emissions,
    "fig7": fig7_emissions,
}

def main():
    # fig1_pressure()
    fig2_approx()

if __name__ == "__main__":
    main()
