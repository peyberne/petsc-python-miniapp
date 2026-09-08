#!/usr/bin/env python3
"""
Benchmark PETSc solver with different options.
Optionally compares each solution to a reference solution (sol.dat).
"""
import argparse
import sys
import time
import itertools
import statistics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ != "__main__" or "--plot-results" not in sys.argv:
    from petsc4py import PETSc

def load_options_from_json(path):
    """Load solver options from a JSON file and return a list of configuration dicts.

    Expected JSON format, e.g.:

    {
      "ksp_rtol": [1e-13],
      "pc_type": ["gamg", "pbjacobi", "sor"],
      "ksp_type": ["gmres", "bcgs"],
      "use_initial_guess": [true],
      "pc_options": {
        "sor": ["pc_sor_local_symmetric"]
      }
    }

    The optional "pc_options" dict maps pc_type values to lists of extra PETSc
    options that are set only when that preconditioner is used.
    """
    with open(path, "r") as f:
        opts = json.load(f)

    required_keys = ["ksp_rtol", "pc_type", "ksp_type", "use_initial_guess"]
    for k in required_keys:
        if k not in opts:
            raise ValueError(f"Missing required key '{k}' in config file: {path}")

    pc_options = opts.get("pc_options", {})

    keys = [k for k in opts.keys() if k != "pc_options"]
    lists = [opts[k] for k in keys]

    config_list = []
    for combo in itertools.product(*lists):
        entry = {key: value for key, value in zip(keys, combo)}
        pc_type = entry.get("pc_type")
        if pc_type in pc_options:
            entry["petsc_options"] = pc_options[pc_type]
        config_list.append(entry)

    return config_list

def load_petsc_data(mat_file, rhs_file, guess_file=None, ref_file=None, use_gpu=False):
    """Load matrix, RHS, initial guess, and optional reference solution from PETSc binary files."""

    # Load matrix
    viewer_mat = PETSc.Viewer().createBinary(mat_file, 'r')
    mat = PETSc.Mat().load(viewer_mat)
    viewer_mat.destroy()

    # Load RHS
    viewer_rhs = PETSc.Viewer().createBinary(rhs_file, 'r')
    rhs = PETSc.Vec().load(viewer_rhs)
    viewer_rhs.destroy()

    # Load optional initial guess
    guess = None
    if guess_file:
        viewer_guess = PETSc.Viewer().createBinary(guess_file, 'r')
        guess = PETSc.Vec().load(viewer_guess)
        viewer_guess.destroy()

    # Load optional reference solution
    ref_sol = None
    if ref_file:
        viewer_ref = PETSc.Viewer().createBinary(ref_file, 'r')
        ref_sol = PETSc.Vec().load(viewer_ref)
        viewer_ref.destroy()

    # If GPU requested, tell PETSc to use GPU types
    if use_gpu:
        PETSc.Sys.Print("Enabling GPU types via PETSc options...")
        opts = PETSc.Options()
        opts.setValue("mat_type", "aijcusparse")
        opts.setValue("vec_type", "cuda")

        # Apply options to existing objects
        mat.setFromOptions()
        rhs.setFromOptions()
        if guess:
            guess.setFromOptions()
        if ref_sol:
            ref_sol.setFromOptions()

    return mat, rhs, guess, ref_sol


def solve_with_options(mat, rhs, initial_guess, ref_solution,
                       ksp_type, pc_type, rtol=1e-13, use_initial_guess=True):
    """Solve the system with given options and measure time, plus error vs reference solution."""

    # Create solution vector and set initial guess
    x = mat.createVecRight()
    if use_initial_guess and initial_guess is not None:
        initial_guess.copy(x)  # Copy provided initial guess into solution vector
    else:
        x.set(0.0)  # Zero initial guess

    # Create KSP solver
    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(mat)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=rtol)

    # Configure preconditioner
    pc = ksp.getPC()
    pc.setType(pc_type)

    # Initial guess
    ksp.setInitialGuessNonzero(use_initial_guess and initial_guess is not None)

    # Configuration from command line options (optional)
    ksp.setFromOptions()

    # Measure solve time
    PETSc.COMM_WORLD.barrier()
    t_start = time.time()

    try:
        ksp.solve(rhs, x)
        PETSc.COMM_WORLD.barrier()
        t_end = time.time()

        solve_time = t_end - t_start
        converged = ksp.getConvergedReason() > 0
        iterations = ksp.getIterationNumber()
        residual = ksp.getResidualNorm()
        solution_norm = x.norm(PETSc.NormType.NORM_1)

        # Error vs reference solution (if provided)
        error_l1 = None
        if ref_solution is not None:
            diff = x.copy()
            diff.axpy(-1.0, ref_solution)  # diff = x - ref_solution
            error_l1 = diff.norm(PETSc.NormType.NORM_1)
            diff.destroy()
    except Exception as e:
        PETSc.Sys.Print(f"Error during solve: {e}")
        solve_time = float('inf')
        converged = False
        iterations = -1
        residual = float('inf')
        solution_norm = float('inf')
        error_l1 = float('inf')

    ksp.destroy()
    x.destroy()

    return {
        'time': solve_time,
        'converged': converged,
        'iterations': iterations,
        'residual': residual,
        'solution_norm': solution_norm,
        'error_l1': error_l1,
    }

def test_positive_quadratic_forms(mat, number_of_tests=10, tolerance=0.0):
    """Look for vectors v for which v^H A v <= tolerance.

    Passing does not prove positive definiteness.
    Failing proves that the matrix is not positive definite.
    """

    nrows, ncols = mat.getSize()

    if nrows != ncols:
        return False

    x = mat.createVecRight()
    ax = mat.createVecLeft()

    passed = True
    minimum_value = float("inf")

    try:
        random_context = PETSc.Random().create(comm=PETSc.COMM_WORLD)

        for test_number in range(number_of_tests):
            x.setRandom(random_context)
            mat.mult(x, ax)

            # dot() is conjugating for complex PETSc scalars.
            quadratic_form = x.dot(ax)
            real_value = float(np.real(quadratic_form))
            minimum_value = min(minimum_value, real_value)

            PETSc.Sys.Print(
                f"Quadratic-form test {test_number + 1}: "
                f"x^H A x = {quadratic_form}"
            )

            if abs(np.imag(quadratic_form)) > 1e-10:
                PETSc.Sys.Print(
                    "  Non-negligible imaginary part: matrix is likely "
                    "not Hermitian."
                )
                passed = False

            if real_value <= tolerance:
                passed = False

        random_context.destroy()

    finally:
        x.destroy()
        ax.destroy()

    PETSc.Sys.Print(
        f"Minimum sampled x^H A x: {minimum_value:.6e}"
    )

    if passed:
        PETSc.Sys.Print(
            "All sampled quadratic forms were positive. "
            "This is evidence, not proof, of positive definiteness."
        )
    else:
        PETSc.Sys.Print(
            "A non-positive quadratic form was found: "
            "the matrix is not positive definite."
        )

    return passed

def inspect_matrix_properties(mat, symmetry_tol=1e-12):
    """Inspect structural and numerical matrix properties.

    Symmetry is tested numerically.
    Positive definiteness is not proven unless PETSc already knows the SPD flag.
    """

    nrows, ncols = mat.getSize()

    PETSc.Sys.Print("\n=== Matrix properties ===")
    PETSc.Sys.Print(f"Shape: {nrows} x {ncols}")

    if nrows != ncols:
        PETSc.Sys.Print("Square: False")
        PETSc.Sys.Print("Symmetric: False")
        PETSc.Sys.Print("SPD: False")
        return {
            "square": False,
            "structurally_symmetric": False,
            "symmetric": False,
            "spd_known": False,
            "spd": False,
        }

    PETSc.Sys.Print("Square: True")

    # Cheap structural test: compares sparsity pattern, not values.
    try:
        structurally_symmetric = mat.isStructurallySymmetric()
    except PETSc.Error:
        structurally_symmetric = None
        PETSc.Sys.Print(
            f"Structural symmetry test not supported for {mat.getType()}"
        )
    PETSc.Sys.Print(
        f"Structurally symmetric: {structurally_symmetric}"
    )

    # Numerical symmetry test. This is collective and can be expensive.
    symmetric = mat.isSymmetric(tol=symmetry_tol)
    PETSc.Sys.Print(
        f"Numerically symmetric, tolerance={symmetry_tol:.1e}: {symmetric}"
    )

    # PETSc may already have a symmetry flag attached to the matrix.
    symmetric_set, symmetric_flag = mat.isSymmetricKnown()
    PETSc.Sys.Print(
        f"PETSc symmetry flag known: {symmetric_set}"
    )
    if symmetric_set:
        PETSc.Sys.Print(
            f"PETSc stored symmetry flag: {symmetric_flag}"
        )

    # petsc4py does not expose MatIsSPDKnown consistently across all
    # PETSc/petsc4py versions. Try the matrix option if available.
    spd_known = False
    spd = False

    try:
        spd = bool(mat.getOption(PETSc.Mat.Option.SPD))
        spd_known = spd
    except (AttributeError, TypeError, PETSc.Error):
        pass

    if spd_known:
        PETSc.Sys.Print(f"PETSc SPD flag: {spd}")
    else:
        PETSc.Sys.Print(
            "SPD: unknown; symmetry alone does not prove positive definiteness"
        )

    return {
        "square": True,
        "structurally_symmetric": structurally_symmetric,
        "symmetric": symmetric,
        "symmetry_flag_known": symmetric_set,
        "symmetry_flag": symmetric_flag if symmetric_set else None,
        "spd_known": spd_known,
        "spd": spd if spd_known else None,
    }

def run_benchmarks(mat_file, rhs_file, guess_file=None, ref_file=None,
                   use_gpu=False, config_file=None, repetitions=1,
                   petsc_options=None):
    """Run all benchmarks."""

    # Load data
    PETSc.Sys.Print("Loading data...")
    mat, rhs, guess, ref_sol = load_petsc_data(
        mat_file, rhs_file, guess_file, ref_file, use_gpu=use_gpu
    )

    # Apply extra PETSc options
    if petsc_options:
        opts = PETSc.Options()
        for opt in petsc_options:
            if "=" in opt:
                key, val = opt.split("=", 1)
                opts.setValue(key, val)
            else:
                opts.setValue(opt, 1)
        PETSc.Sys.Print(f"Extra PETSc options: {petsc_options}")

    matrix_properties = inspect_matrix_properties(
        mat,
        symmetry_tol=1e-12,
    )

    if matrix_properties["symmetric"]:
        sampled_positive = test_positive_quadratic_forms(
            mat,
            number_of_tests=10,
            tolerance=0.0,
        )
    else:
        sampled_positive = False

    size = PETSc.COMM_WORLD.getSize()
    PETSc.Sys.Print(f"Running with {size} MPI process(es).")
    PETSc.Sys.Print(f"Matrix size: {mat.getSize()}")
    PETSc.Sys.Print(f"RHS size: {rhs.getSize()}")

    PETSc.Sys.Print("\n=== L1 Norms ===")
    rhs_norm = rhs.norm(PETSc.NormType.NORM_1)
    PETSc.Sys.Print(f"RHS L1 norm: {rhs_norm:.6e}")

    mat_norm = mat.norm(PETSc.NormType.NORM_1)
    PETSc.Sys.Print(f"Matrix L1 norm: {mat_norm:.6e}")

    if guess:
        guess_norm = guess.norm(PETSc.NormType.NORM_1)
        PETSc.Sys.Print(f"Initial guess L1 norm: {guess_norm:.6e}")

    if ref_sol:
        ref_norm = ref_sol.norm(PETSc.NormType.NORM_1)
        PETSc.Sys.Print(f"Reference solution L1 norm: {ref_norm:.6e}")

    PETSc.Sys.Print("")

    # Load solver configurations
    if config_file is not None:
        PETSc.Sys.Print(f"Loading solver options from JSON: {config_file}")
        combinations = load_options_from_json(config_file)
        use_config_file = True
    else:
        PETSc.Sys.Print("Using built-in default solver options")
        options = {
            "ksp_rtol": [1e-13],
            "pc_type": ["gamg", "pbjacobi"],
            "ksp_type": ["gmres", "bcgs", "fgmres", "lgmres", "dgmres"],
            "use_initial_guess": [True],
        }
        keys = list(options.keys())
        combinations = list(itertools.product(*[options[k] for k in keys]))
        use_config_file = False

    results = []

    PETSc.Sys.Print(f"\nTesting {len(combinations)} configurations...\n")

    for i, combo in enumerate(combinations):
        if use_config_file:
            rtol = combo["ksp_rtol"]
            pc_type = combo["pc_type"]
            ksp_type = combo["ksp_type"]
            use_guess = combo["use_initial_guess"]
            config_petsc_opts = combo.get("petsc_options", [])
        else:
            rtol = combo[0]
            pc_type = combo[1]
            ksp_type = combo[2]
            use_guess = combo[3]
            config_petsc_opts = []

        guess_label = "guess" if use_guess else "zero"
        label = f"{ksp_type}+{pc_type} | rtol={rtol:.0e} | {guess_label}"
        PETSc.Sys.Print(f"[{i+1}/{len(combinations)}] Test: {label}")

        # Apply per-config PETSc options (clear previous, then set)
        all_opts = PETSc.Options()
        all_opts.delAll()
        if petsc_options:
            for opt in petsc_options:
                if "=" in opt:
                    k, v = opt.split("=", 1)
                    all_opts.setValue(k, v)
                else:
                    all_opts.setValue(opt, 1)
        for opt in config_petsc_opts:
            if "=" in opt:
                k, v = opt.split("=", 1)
                all_opts.setValue(k, v)
            else:
                all_opts.setValue(opt, 1)

        samples = []
        for repetition in range(repetitions):
            PETSc.Sys.Print(f"  Repetition {repetition + 1}/{repetitions}")
            samples.append(solve_with_options(
                mat, rhs, guess, ref_sol, ksp_type, pc_type, rtol, use_guess
            ))

        median_time = statistics.median(sample["time"] for sample in samples)
        result = min(samples, key=lambda sample: abs(sample["time"] - median_time))
        result = result.copy()
        result["time"] = median_time
        result["time_samples"] = [sample["time"] for sample in samples]
        result["converged"] = all(sample["converged"] for sample in samples)
        result["mpi_processes"] = size

        result["ksp_type"] = ksp_type
        result["pc_type"] = pc_type
        result["ksp_rtol"] = rtol
        result["use_initial_guess"] = use_guess
        result["label"] = label

        err_str = ""
        if result['error_l1'] is not None and result['error_l1'] != float('inf'):
            err_str = f", Error L1 vs ref: {result['error_l1']:.6e}"

        PETSc.Sys.Print(
            f"  Median TTS: {result['time']:.4f}s, "
            f"Converged: {result['converged']}, "
            f"Iterations: {result['iterations']}, "
            f"Residual: {result['residual']:.2e}, "
            f"Solution L1 norm: {result['solution_norm']:.6e}"
            f"{err_str}\n"
        )

        results.append(result)

    # Print rankings for this MPI count
    by_tts = sorted(enumerate(results), key=lambda r: r[1]['time'])
    by_iter = sorted(
        enumerate(results),
        key=lambda r: r[1]['iterations'] if r[1]['iterations'] >= 0 else float('inf'),
    )

    PETSc.Sys.Print("=== Ranking by TTS (fastest first) ===")
    for rank, (idx, r) in enumerate(by_tts, 1):
        status = "ok" if r['converged'] else "FAIL"
        PETSc.Sys.Print(
            f"  {rank}. {r['label']} | {r['time']:.4f}s | "
            f"{r['iterations']} it | {status}"
        )

    PETSc.Sys.Print("\n=== Ranking by iterations (fewest first) ===")
    for rank, (idx, r) in enumerate(by_iter, 1):
        status = "ok" if r['converged'] else "FAIL"
        PETSc.Sys.Print(
            f"  {rank}. {r['label']} | {r['iterations']} it | "
            f"{r['time']:.4f}s | {status}"
        )

    PETSc.Sys.Print("")

    # Cleanup
    mat.destroy()
    rhs.destroy()
    if guess:
        guess.destroy()
    if ref_sol:
        ref_sol.destroy()

    return results

def save_results(results, output_file, test_case=None):
    """Save one MPI-size benchmark result set."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mpi_processes": PETSc.COMM_WORLD.getSize(),
        "test_case": test_case,
        "results": results,
    }
    with output_path.open("w") as output:
        json.dump(payload, output, indent=2)
    print(f"Results saved: {output_path}")


def plot_scaling_results(result_files, output_file='results/benchmark_results.png'):
    """Plot median time to solution against the number of MPI processes."""
    datasets = []
    for result_file in result_files:
        with Path(result_file).open() as source:
            datasets.append(json.load(source))

    # Extract test case name from first dataset
    test_case = None
    for ds in datasets:
        tc = ds.get("test_case")
        if tc:
            test_case = tc
            break

    mpi_counts = sorted({dataset["mpi_processes"] for dataset in datasets})
    series = {}
    failed_points = []
    failed_labels = set()
    for dataset in datasets:
        mpi_processes = dataset["mpi_processes"]
        for result in dataset["results"]:
            series.setdefault(result["label"], {})[mpi_processes] = result["time"]
            if not result["converged"]:
                failed_points.append((result["label"], mpi_processes, result["time"]))
                failed_labels.add(result["label"])

    if not series:
        raise RuntimeError("No converged solutions are available to plot")

    # Extract common solver parameters from first result
    sample = datasets[0]["results"][0]
    rtol = sample.get("ksp_rtol", None)
    use_guess = sample.get("use_initial_guess", None)
    param_parts = []
    if rtol is not None:
        param_parts.append(f"rtol = {rtol:.0e}")
    if use_guess is not None:
        param_parts.append(f"initial guess = {'yes' if use_guess else 'no'}")
    param_str = "  |  ".join(param_parts)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.get_cmap('tab20').colors
    for index, (label, points) in enumerate(sorted(series.items())):
        x_values = sorted(points)
        y_values = [points[mpi_processes] for mpi_processes in x_values]
        color = colors[index % len(colors)]
        ax.plot(
            x_values,
            y_values,
            marker='o',
            linewidth=2,
            linestyle='--' if label in failed_labels else '-',
            color=color,
            label=label,
        )
        failed_curve_points = [point for point in failed_points if point[0] == label]
        if failed_curve_points:
            ax.scatter(
                [point[1] for point in failed_curve_points],
                [point[2] for point in failed_curve_points],
                marker='x',
                s=70,
                linewidths=2,
                color=color,
                zorder=5,
            )

    title = 'PETSc GPU strong scaling'
    if test_case:
        title += f' — {test_case}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    if param_str:
        ax.set_xlabel(f'Number of MPI processes  ({param_str})', fontsize=11)
    else:
        ax.set_xlabel('Number of MPI processes', fontsize=11)
    ax.set_xticks(mpi_counts)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if failed_points:
        fig.text(
            0.5,
            0.01,
            f'Dashed curves contain a non-converged point '
            f'({len(failed_points)} point(s) total)',
            ha='center',
            fontsize=9,
            style='italic',
            color='red',
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Scaling plot saved: {output_path}")

    # Generate ranking figures
    plot_scaling_rankings(datasets, output_path.parent, test_case=test_case)


def plot_scaling_rankings(datasets, output_dir, test_case=None):
    """Generate ranking figures by TTS and by iterations for each MPI count."""
    output_dir = Path(output_dir)

    mpi_counts = sorted({d["mpi_processes"] for d in datasets})
    n_mpi = len(mpi_counts)

    # Collect per-MPI-count data
    mpi_data = {}
    for dataset in datasets:
        mpi = dataset["mpi_processes"]
        mpi_data[mpi] = dataset["results"]

    # Common labels across all MPI counts
    all_labels = sorted({r["label"] for d in datasets for r in d["results"]})
    cmap = plt.get_cmap('tab20')
    color_map = {label: cmap(i % 20) for i, label in enumerate(all_labels)}

    # Extract common solver parameters
    sample = datasets[0]["results"][0]
    rtol = sample.get("ksp_rtol", None)
    use_guess = sample.get("use_initial_guess", None)
    param_parts = []
    if rtol is not None:
        param_parts.append(f"rtol = {rtol:.0e}")
    if use_guess is not None:
        param_parts.append(f"initial guess = {'yes' if use_guess else 'no'}")
    param_str = "  |  ".join(param_parts)

    # --- Ranking by TTS ---
    fig, axes = plt.subplots(1, n_mpi, figsize=(7 * n_mpi, max(6, len(all_labels) * 0.45)),
                             sharey=False)
    if n_mpi == 1:
        axes = [axes]

    for ax, mpi in zip(axes, mpi_counts):
        results = sorted(mpi_data[mpi], key=lambda r: r["time"])
        labels = [r["label"] for r in results]
        times = [r["time"] for r in results]
        converged = [r["converged"] for r in results]
        bar_colors = [color_map[l] for l in labels]
        edge_colors = ["black" if c else "red" for c in converged]
        linewidths = [1.5 if not c else 0.5 for c in converged]

        y_pos = range(len(labels))
        bars = ax.barh(y_pos, times, color=bar_colors, edgecolor=edge_colors,
                       linewidth=linewidths, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Median TTS (s)", fontsize=10)
        ax.set_title(f"{mpi} MPI", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, (bar, t, c) in enumerate(zip(bars, times, converged)):
            marker = "" if c else " x"
            ax.text(t, i, f" {t:.2f}s{marker}", va="center", fontsize=7)

    tts_suptitle = f"Ranking by Time to Solution  ({param_str})"
    if test_case:
        tts_suptitle += f"  — {test_case}"
    fig.suptitle(tts_suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    tts_path = output_dir / "ranking_tts.png"
    fig.savefig(tts_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"TTS ranking saved: {tts_path}")

    # --- Ranking by iterations ---
    fig, axes = plt.subplots(1, n_mpi, figsize=(7 * n_mpi, max(6, len(all_labels) * 0.45)),
                             sharey=False)
    if n_mpi == 1:
        axes = [axes]

    for ax, mpi in zip(axes, mpi_counts):
        results = sorted(mpi_data[mpi],
                         key=lambda r: r["iterations"] if r["iterations"] >= 0 else float("inf"))
        labels = [r["label"] for r in results]
        iters = [max(r["iterations"], 0) for r in results]
        converged = [r["converged"] for r in results]
        bar_colors = [color_map[l] for l in labels]
        edge_colors = ["black" if c else "red" for c in converged]
        linewidths = [1.5 if not c else 0.5 for c in converged]

        y_pos = range(len(labels))
        bars = ax.barh(y_pos, iters, color=bar_colors, edgecolor=edge_colors,
                       linewidth=linewidths, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Iterations", fontsize=10)
        ax.set_title(f"{mpi} MPI", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, (bar, it, c) in enumerate(zip(bars, iters, converged)):
            marker = "" if c else " x"
            ax.text(it, i, f" {it}{marker}", va="center", fontsize=7)

    iter_suptitle = f"Ranking by Iterations  ({param_str})"
    if test_case:
        iter_suptitle += f"  — {test_case}"
    fig.suptitle(iter_suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    iter_path = output_dir / "ranking_iterations.png"
    fig.savefig(iter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Iteration ranking saved: {iter_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat")
    parser.add_argument("--rhs")
    parser.add_argument("--guess")
    parser.add_argument("--ref")
    parser.add_argument("--config")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--results-json")
    parser.add_argument("--test-case")
    parser.add_argument("--petsc-options", nargs="*", default=[],
                        help="Extra PETSc options, e.g. --petsc-options pc_sor_local_symmetric")
    parser.add_argument("--plot-results", nargs="+")
    parser.add_argument("--output", default="results/benchmark_results.png")
    args = parser.parse_args()

    if args.plot_results:
        return args
    if not args.mat or not args.rhs:
        parser.error("--mat and --rhs are required when running benchmarks")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.plot_results:
        plot_scaling_results(args.plot_results, args.output)
    else:
        results = run_benchmarks(
            args.mat, args.rhs, args.guess, args.ref,
            use_gpu=args.gpu,
            config_file=args.config,
            repetitions=args.repetitions,
            petsc_options=args.petsc_options,
        )
        if PETSc.COMM_WORLD.getRank() == 0 and args.results_json:
            save_results(results, args.results_json, test_case=args.test_case)
        PETSc.Sys.Print("\nBenchmark completed!")
