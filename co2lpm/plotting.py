from __future__ import annotations
import matplotlib.pyplot as plt

def figure_timeseries(state, emissions=None):
    """Return a simple 3‑panel figure mirroring the manuscript plots."""
    t_years = state.t / (365.25*24*3600)

    fig, axs = plt.subplots(3, 1, figsize=(5, 8), sharex=True)

    axs[0].plot(t_years, state.pressure/1e6)
    axs[0].set_ylabel("Pressure [MPa]")

    axs[1].plot(t_years, 100*state.concentration)
    axs[1].set_ylabel("CO$_2$ [wt %]")

    if emissions:
        axs[2].plot(t_years, emissions["Q_out"]/1e6*365.25*24*3600, label="field")
        axs[2].plot(t_years, emissions["Q_pp"]/1e6*365.25*24*3600, label="plant")
        axs[2].plot(t_years, emissions["Q_total"]/1e6*365.25*24*3600, label="total")
        axs[2].axhline(emissions["Q_nat0"]/1e6*365.25*24*3600, ls=":", c="k", label="nat₀")
        axs[2].axhline(emissions["Q_nat_inf"]/1e6*365.25*24*3600, ls="--", c="k", label="nat∞")
        axs[2].legend()
    axs[2].set_ylabel("Emissions [kt/yr]")
    axs[2].set_xlabel("time [yr]")

    fig.tight_layout()
    return fig
