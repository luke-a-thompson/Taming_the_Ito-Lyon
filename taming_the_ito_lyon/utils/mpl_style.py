"""Global Matplotlib styling for paper-quality figures.

This is intended to be called once at import-time by plotting modules.
"""

from __future__ import annotations


def apply_mpl_style() -> None:
    """Apply a global Matplotlib style (ICML-ish, clean + legible).

    Safe to call multiple times.
    """
    import matplotlib as mpl
    import matplotlib.style as mplstyle
    from cycler import cycler

    # A clean base style; ships with Matplotlib (>=3.6 typically).
    # If unavailable in a given environment, we just rely on rcParams below.
    try:
        mplstyle.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    # Colorblind-friendly-ish palette.
    colors = [
        "#4C78A8",  # blue
        "#F58518",  # orange
        "#54A24B",  # green
        "#E45756",  # red
        "#72B7B2",  # teal
        "#B279A2",  # purple
        "#FF9DA6",  # pink
        "#9D755D",  # brown
        "#BAB0AC",  # gray
    ]

    mpl.rcParams.update(
        {
            # Typography
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            # Lines / markers
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            # Axes / grid
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler("color", colors),
            # Figure / savefig
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


if __name__ == "__main__":
    import numpy as np

    from taming_the_ito_lyon.training.results_plotting import (
        save_rough_volatility_two_panel_plot,
    )

    apply_mpl_style()

    data = np.load("data/rough_ou_processes/rough_ou_data_H0.70.npz")
    solution = np.asarray(data["solution"])
    driver = np.asarray(data["driver"])

    rng = np.random.default_rng(0)
    idx = rng.choice(solution.shape[0], size=8, replace=False)

    save_rough_volatility_two_panel_plot(
        left=solution[idx],
        right=driver[idx],
        out_file="z_paper_content/rough_ou_example_solution_vs_driver.png",
        n_plot=8,
        left_title="Rough OU solution (8 random paths)",
        right_title="Driver (8 random paths)",
        left_color="black",
        right_color="red",
        alpha=0.6,
        figsize=(10.0, 4.0),
    )
