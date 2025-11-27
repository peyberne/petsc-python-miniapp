#!/usr/bin/env python3
"""
Benchmark PETSc solver with different options
"""
import sys
import time
import itertools
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

def load_petsc_data(mat_file, rhs_file, guess_file=None):
    """Load matrix, RHS and initial guess from PETSc files"""
    viewer_mat = PETSc.Viewer().createBinary(mat_file, 'r')
    mat = PETSc.Mat().load(viewer_mat)
    viewer_mat.destroy()
    
    viewer_rhs = PETSc.Viewer().createBinary(rhs_file, 'r')
    rhs = PETSc.Vec().load(viewer_rhs)
    viewer_rhs.destroy()
    
    guess = None
    if guess_file:
        viewer_guess = PETSc.Viewer().createBinary(guess_file, 'r')
        guess = PETSc.Vec().load(viewer_guess)
        viewer_guess.destroy()
    
    return mat, rhs, guess

def solve_with_options(mat, rhs, initial_guess, ksp_type, pc_type, rtol=1e-13, use_initial_guess=True):
    """Solve the system with given options and measure time"""
    
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
        
    except Exception as e:
        PETSc.Sys.Print(f"Error during solve: {e}")
        solve_time = float('inf')
        converged = False
        iterations = -1
        residual = float('inf')
        solution_norm = float('inf')
    
    ksp.destroy()
    x.destroy()
    
    return {
        'time': solve_time,
        'converged': converged,
        'iterations': iterations,
        'residual': residual,
        'solution_norm': solution_norm
    }

def run_benchmarks(mat_file, rhs_file, guess_file=None):
    """Run all benchmarks"""
    
    # Load data
    PETSc.Sys.Print("Loading data...")
    mat, rhs, guess = load_petsc_data(mat_file, rhs_file, guess_file)
    
    PETSc.Sys.Print(f"Matrix size: {mat.getSize()}")
    PETSc.Sys.Print(f"RHS size: {rhs.getSize()}")
    
    # Compute and print L1 norms
    PETSc.Sys.Print("\n=== L1 Norms ===")
    rhs_norm = rhs.norm(PETSc.NormType.NORM_1)
    PETSc.Sys.Print(f"RHS L1 norm: {rhs_norm:.6e}")
    
    mat_norm = mat.norm(PETSc.NormType.NORM_1)
    PETSc.Sys.Print(f"Matrix L1 norm: {mat_norm:.6e}")
    
    if guess:
        guess_norm = guess.norm(PETSc.NormType.NORM_1)
        PETSc.Sys.Print(f"Initial guess L1 norm: {guess_norm:.6e}")
    
    PETSc.Sys.Print("")
    
    # Options to test
    options = {
        "ksp_rtol": [1e-13],
        "pc_type": ["gamg", "pbjacobi"],
        "ksp_type": ["gmres", "bcgs", "fgmres", "lgmres", "dgmres"],
        "use_initial_guess": [True]
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
        
        result = solve_with_options(mat, rhs, guess, ksp_type, pc_type, rtol, use_guess)
        
        result['ksp_type'] = ksp_type
        result['pc_type'] = pc_type
        result['label'] = label
        
        PETSc.Sys.Print(f"  Time: {result['time']:.4f}s, "
                       f"Converged: {result['converged']}, "
                       f"Iterations: {result['iterations']}, "
                       f"Residual: {result['residual']:.2e}, "
                       f"Solution L1 norm: {result['solution_norm']:.6e}\n")
        
        results.append(result)
    
    # Cleanup
    mat.destroy()
    rhs.destroy()
    if guess:
        guess.destroy()
    
    return results

def plot_results(results, output_file='benchmark_results.png'):
    """Create a plot of the results"""
    
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
        fig.text(0.5, 0.02, f'Note: {len(failed_results)} configuration(s) did not converge', 
                ha='center', fontsize=10, style='italic', color='red')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")
    
    # Display top 3
    print("\n=== TOP 3 fastest configurations ===")
    for i, r in enumerate(converged_results[:3], 1):
        print(f"{i}. {r['label']}: {r['time']:.4f}s ({r['iterations']} iterations)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark_petsc.py <matrix.dat> <rhs.dat> [guess.dat]")
        sys.exit(1)
    
    mat_file = sys.argv[1]
    rhs_file = sys.argv[2]
    guess_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Execute benchmarks
    results = run_benchmarks(mat_file, rhs_file, guess_file)
    
    # Create plot (only on process 0)
    if PETSc.COMM_WORLD.getRank() == 0:
        plot_results(results)
    
    PETSc.Sys.Print("\nBenchmark completed!")
