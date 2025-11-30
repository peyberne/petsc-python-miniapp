#!/usr/bin/env python3
"""
Benchmark PETSc solver with different options.
Optionally compares each solution to a reference solution (sol.dat).
"""
import sys
import time
import itertools
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt


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


def run_benchmarks(mat_file, rhs_file, guess_file=None, ref_file=None, use_gpu=False):
    """Run all benchmarks."""

    # Load data
    PETSc.Sys.Print("Loading data...")
    mat, rhs, guess, ref_sol = load_petsc_data(
        mat_file, rhs_file, guess_file, ref_file, use_gpu=use_gpu
    )

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

    # Options to test
    options = {
        "ksp_rtol": [1e-13],
        "pc_type": ["gamg", "pbjacobi"],
        "ksp_type": ["gmres", "bcgs", "fgmres", "lgmres", "dgmres"],
        "use_initial_guess": [True],
    }

    # Generate all combinations
    keys = list(options.keys())
    combinations = list(itertools.product(*[options[k] for k in keys]))

    results = []

    PETSc.Sys.Print(f"\nTesting {len(combinations)} combinations...\n")

    for i, combo in enumerate(combinations):
        rtol = combo[0]
        pc_type = combo[1]
        ksp_type = combo[2]
        use_guess = combo[3]

        label = f"{ksp_type}+{pc_type}"
        PETSc.Sys.Print(f"[{i+1}/{len(combinations)}] Test: {label}")

        result = solve_with_options(
            mat, rhs, guess, ref_sol, ksp_type, pc_type, rtol, use_guess
        )

        result['ksp_type'] = ksp_type
        result['pc_type'] = pc_type
        result['label'] = label

        err_str = ""
        if result['error_l1'] is not None and result['error_l1'] != float('inf'):
            err_str = f", Error L1 vs ref: {result['error_l1']:.6e}"

        PETSc.Sys.Print(
            f"  Time: {result['time']:.4f}s, "
            f"Converged: {result['converged']}, "
            f"Iterations: {result['iterations']}, "
            f"Residual: {result['residual']:.2e}, "
            f"Solution L1 norm: {result['solution_norm']:.6e}"
            f"{err_str}\n"
        )

        results.append(result)

    # Cleanup
    mat.destroy()
    rhs.destroy()
    if guess:
        guess.destroy()
    if ref_sol:
        ref_sol.destroy()

    return results


def plot_results(results, output_file='results/benchmark_results.png'):
    """Create a plot of the results."""

    # Filter converged results
    converged_results = [r for r in results if r['converged']]
    failed_results = [r for r in results if not r['converged']]

    if not converged_results:
        print("No solution converged!")
        return

    # Sort by time
    converged_results.sort(key=lambda x: x['time'])

    labels = [r['label'] for r in converged_results]
    times = [r['time'] for r in converged_results]
    iterations = [r['iterations'] for r in converged_results]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Time plot
    colors = ['green' if t == min(times) else 'blue' for t in times]
    bars1 = ax1.barh(range(len(labels)), times, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Solve time (seconds)', fontsize=11)
    ax1.set_title('PETSc Benchmark - Solve time by configuration', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add values on bars
    for i, (bar, time_val) in enumerate(zip(bars1, times)):
        ax1.text(time_val, i, f' {time_val:.3f}s', va='center', fontsize=8)

    # Iterations plot
    colors2 = ['orange' if it == min(iterations) else 'steelblue' for it in iterations]
    bars2 = ax2.barh(range(len(labels)), iterations, color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Number of iterations', fontsize=11)
    ax2.set_title('Number of iterations by configuration', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add values
    for i, (bar, it) in enumerate(zip(bars2, iterations)):
        ax2.text(it, i, f' {it}', va='center', fontsize=8)

    # Add info about failures
    if failed_results:
        fig.text(
            0.5,
            0.02,
            f'Note: {len(failed_results)} configuration(s) did not converge',
            ha='center',
            fontsize=10,
            style='italic',
            color='red',
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")

    # Display top 3
    print("\n=== TOP 3 fastest configurations ===")
    for i, r in enumerate(converged_results[:3], 1):
        print(f"{i}. {r['label']}: {r['time']:.4f}s ({r['iterations']} iterations)")


if __name__ == "__main__":

    # Default values
    mat_file = None
    rhs_file = None
    guess_file = None
    ref_file = None
    use_gpu = False

    # Parse arguments manually
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--mat" and i + 1 < len(sys.argv):
            mat_file = sys.argv[i + 1]
            i += 1

        elif arg == "--rhs" and i + 1 < len(sys.argv):
            rhs_file = sys.argv[i + 1]
            i += 1

        elif arg == "--guess" and i + 1 < len(sys.argv):
            guess_file = sys.argv[i + 1]
            i += 1

        elif arg == "--ref" and i + 1 < len(sys.argv):
            ref_file = sys.argv[i + 1]
            i += 1

        elif arg == "--gpu":
            use_gpu = True

        else:
            PETSc.Sys.Print(f"Warning: ignoring unknown argument: {arg}")

        i += 1

    # Sanity checks
    if mat_file is None or rhs_file is None:
        print("Usage: python benchmark_petsc.py --mat mat.dat --rhs rhs.dat "
              "[--guess guess.dat] [--ref sol.dat] [--gpu]")
        sys.exit(1)

    # Run benchmark
    results = run_benchmarks(mat_file, rhs_file, guess_file, ref_file, use_gpu)

    # Plot (rank 0 only)
    if PETSc.COMM_WORLD.getRank() == 0:
        plot_results(results)

    PETSc.Sys.Print("\nBenchmark completed!")
