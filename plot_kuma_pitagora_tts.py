#!/usr/bin/env python3
"""
Compare time-to-solution on Kuma and Pitagora for the WEST vorticity case.

Figure 1: SOLEDGE3x per-solver TTS from test_s3x_west_v3_8nodes_splitZones
          (32 MPI, 8 nodes, bcgsl+gamg, H100 GPU).
          Kuma  : toto-log-4218687 (PETSc module build2)
          Pitagora: toto.out

Figure 2: Mini-app replay of the dumped vorticity matrix on Pitagora
          (west-reuse-1753733, 32 MPI, reused-PC median vs first solve),
          with the SOLEDGE3x vorticity solve as reference.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")

# SOLEDGE3x per-solver TTS (s), one entry per time step.
# Step 1 includes GAMG PC setup; steps 2-3 reuse the preconditioner.
SOLEDGE_DATA = {
    "Filtering":   {"kuma": [19.28, 0.52, 0.53],   "pitagora": [16.00, 0.09, 0.09]},
    "FN":          {"kuma": [20.24, 1.99, 1.51],   "pitagora": [15.72, 0.39, 0.26]},
    "DiffParal G": {"kuma": [5.09, 0.23, 0.14],    "pitagora": [7.03, 0.28, 0.25]},
    "DiffParal phi": {"kuma": [6.07, 1.43, 0.75],  "pitagora": [8.39, 2.20, 1.86]},
    "DiffParal N": {"kuma": [5.85, 0.23, 0.13],    "pitagora": [6.87, 0.25, 0.22]},
    "Vort":        {"kuma": [48.32, 24.36, 24.76], "pitagora": [19.11, 3.48, 3.69]},
}

# Mini-app replay on Pitagora (west-reuse-1753733): dumped last-iteration
# vorticity matrix, 32 MPI, bcgsl+gamg config family.
MINIAPP_DATA = {
    "gmres+gamg": {"first": 17.75, "reused": 3.43},
    "bcgs+gamg":  {"first": 15.66, "reused": 2.24},
    "bcgsl+gamg": {"first": 16.14, "reused": 2.36},
}

# SOLEDGE3x vorticity solve on Pitagora (last time step, PC reused).
SOLEDGE_VORT_REFERENCE = 3.69

COLOR_KUMA = "#1f77b4"
COLOR_PITA = "#ff7f0e"
COLOR_FIRST = "#aec7e8"
COLOR_REUSED = "#2ca02c"


def plot_soledge(output_path):
    solvers = list(SOLEDGE_DATA)
    n_steps = 3
    fig, axes = plt.subplots(1, n_steps, figsize=(16, 6), sharey=False)

    for ax, step in zip(axes, range(n_steps)):
        kuma = [SOLEDGE_DATA[s]["kuma"][step] for s in solvers]
        pita = [SOLEDGE_DATA[s]["pitagora"][step] for s in solvers]

        x = np.arange(len(solvers))
        width = 0.38
        ax.bar(x - width / 2, kuma, width, label="Kuma", color=COLOR_KUMA)
        ax.bar(x + width / 2, pita, width, label="Pitagora", color=COLOR_PITA)

        for xi, (k, p) in enumerate(zip(kuma, pita)):
            ratio = k / p
            ax.text(xi, max(k, p) * 1.15, f"{ratio:.1f}x",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_yscale("log")
        all_values = kuma + pita
        ax.set_ylim(min(all_values) / 3, max(all_values) * 4)
        ax.set_xticks(x)
        ax.set_xticklabels(solvers, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("TTS (s)" if step == 0 else "")
        ax.set_title(f"Time step {step + 1}"
                     + (" (PC setup included)" if step == 0 else " (PC reused)"),
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle(
        "SOLEDGE3x TTS — WEST vorticity case (32 MPI, 8 nodes, bcgsl+gamg)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_miniapp(output_path):
    configs = list(MINIAPP_DATA)
    first = [MINIAPP_DATA[c]["first"] for c in configs]
    reused = [MINIAPP_DATA[c]["reused"] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))
    width = 0.38
    ax.bar(x - width / 2, first, width, label="First solve (PC setup)",
           color=COLOR_FIRST)
    ax.bar(x + width / 2, reused, width, label="Reused-PC median",
           color=COLOR_REUSED)

    ax.axhline(
        SOLEDGE_VORT_REFERENCE, color="crimson", linestyle="--", linewidth=1.5,
        label=f"SOLEDGE3x Vort solve, Pitagora ({SOLEDGE_VORT_REFERENCE:.2f} s)",
    )

    for xi, (f, r) in enumerate(zip(first, reused)):
        ax.text(xi - width / 2, f * 1.05, f"{f:.2f}s", ha="center", fontsize=8)
        ax.text(xi + width / 2, r * 1.05, f"{r:.2f}s", ha="center", fontsize=8)

    ax.set_yscale("log")
    ax.set_ylim(1.0, max(first) * 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("TTS (s)")
    ax.set_title(
        "Mini-app replay — WEST vorticity matrix, Pitagora (32 MPI)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_soledge(RESULTS_DIR / "kuma_vs_pitagora_tts.png")
    plot_miniapp(RESULTS_DIR / "kuma_vs_pitagora_miniapp.png")
