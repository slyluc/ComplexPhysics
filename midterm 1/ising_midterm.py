# Create the full solution package: a runnable Python script and a markdown report
import numpy as np, textwrap, os, json, math

base = "/mnt/data"
os.makedirs(base, exist_ok=True)

# ---------------------------
# 1) Write the Python script
# ---------------------------
py_code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex Physics — Midterm 1 (Ising Model) — Full Solution Code
Author: (your name)
Date: (auto-generated)
Python >= 3.9

What this script does
---------------------
Implements the ferromagnetic Ising model (J>0, h=0) and solves:
  (1) Metropolis simulations for two geometries:
      A) 1D ring (periodic) with nearest-neighbor coupling
      B) 1D ring + two additional random undirected links per node (on average degree ~4)
      (The random extra links are *frozen* during each run and symmetric.)
  (2) Estimates phase transition behavior with finite-size scaling:
      - Plots |m| vs T for various system sizes N
      - Optionally identifies T_c(N) from susceptibility peaks and extrapolates T_c vs 1/N
  (3) (Report only) Domain-wall free-energy argument for case A (and i±2 modification).

Outputs
-------
Saves figures into ./out/
- <geom>__magnetization_vs_T.png  (|m| vs T, overlaid for different N)
- <geom>__susceptibility_vs_T.png (χ(T) vs T, overlaid for different N)
- Tc_vs_invN.png                  (estimated T_c(N) vs 1/N for both geometries)

Usage examples
--------------
# Quick run (faster, fewer sweeps; good for smoke test)
python ising_midterm.py --quick

# Full(er) run (more sweeps for smoother curves)
python ising_midterm.py --geom both --sizes 64 128 256 512 --Tmin 0.5 --Tmax 4.0 --nT 15 --sweeps 8000 --burn 3000
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------

def seed_everything(seed: int | None):
    if seed is not None:
        np.random.seed(seed)

def ensure_out_dir(path: str = "out"):
    os.makedirs(path, exist_ok=True)
    return path

# ---------------------------
# Geometry builders
# ---------------------------

def build_ring_neighbors(N: int) -> List[np.ndarray]:
    """Return neighbors for ring with periodic boundary and nearest neighbors only."""
    neigh = []
    for i in range(N):
        neigh.append(np.array([(i - 1) % N, (i + 1) % N], dtype=int))
    return neigh

def build_ring_plus_random(N: int, extra_per_node: int = 2, seed: int | None = None) -> List[np.ndarray]:
    """
    Build ring neighbors and then add 'extra_per_node' random undirected links for each node.
    - Avoid self-loops and duplicates.
    - Avoid adding the immediate ring neighbors as 'extra' (they are already connected).
    Result: Average degree ~ 2 + 2*extra_per_node = 6 if we counted both directions independently,
            but because we store undirected edges once, the degree is ~ 2 + extra_per_node*2 = 6.
      NOTE: Each node *attempts* to add 2 extra links; due to collision filtering, the realized
            degree distribution is near 4 (ring degree 2 + ~2 random long-range neighbors), but not exact.
    """
    seed_everything(seed)
    # start with ring edges
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        a, b = (i, j) if i < j else (j, i)
        edges.add((a, b))

    # extra long-range links
    for i in range(N):
        forbidden = {i, (i - 1) % N, (i + 1) % N}
        attempts = 0
        added = 0
        while added < extra_per_node and attempts < 10 * N:
            attempts += 1
            j = np.random.randint(0, N)
            if j in forbidden:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if (a, b) in edges:
                continue
            # Add undirected edge
            edges.add((a, b))
            added += 1

    # Build adjacency from edges
    neigh_lists = [[] for _ in range(N)]
    for a, b in edges:
        neigh_lists[a].append(b)
        neigh_lists[b].append(a)
    neigh = [np.array(lst, dtype=int) for lst in neigh_lists]
    return neigh

# ---------------------------
# Metropolis simulation
# ---------------------------

@dataclasses.dataclass
class SimParams:
    N: int
    T: float
    sweeps: int
    burn: int
    measure_every: int = 1
    J: float = 1.0
    seed: int | None = None
    init: str = "random"  # "random" or "allup"

@dataclasses.dataclass
class SimResults:
    N: int
    T: float
    m_abs_avg: float
    m_avg: float
    m2_avg: float
    e_avg: float
    e2_avg: float
    magnetizations: np.ndarray  # time series (per sweep) of m
    energies: np.ndarray        # time series (per sweep) of H/N

def initial_spins(N: int, init: str = "random") -> np.ndarray:
    if init == "allup":
        return np.ones(N, dtype=np.int8)
    return np.random.choice([-1, 1], size=N).astype(np.int8)

def total_energy(spins: np.ndarray, neighbors: List[np.ndarray], J: float = 1.0) -> float:
    """Compute H = -J * sum_{(i<j) in edges} s_i s_j. We sum each undirected edge once."""
    H = 0.0
    for i in range(len(spins)):
        si = spins[i]
        for j in neighbors[i]:
            if j > i:  # count each undirected edge once
                H += -J * si * spins[j]
    return H

def metropolis(spins: np.ndarray, neighbors: List[np.ndarray], params: SimParams) -> SimResults:
    """
    Single-temperature Metropolis run. One 'sweep' = N proposed flips.
    Records magnetization m and energy per spin after each measured sweep.
    """
    N = params.N
    J = params.J
    T = params.T

    H = total_energy(spins, neighbors, J=J)

    n_meas = max(0, (params.sweeps - params.burn) // params.measure_every)
    m_series = np.empty(n_meas, dtype=float)
    e_series = np.empty(n_meas, dtype=float)

    k = 0
    for sweep in range(params.sweeps):
        for _ in range(N):
            i = np.random.randint(0, N)
            si = spins[i]
            # local field
            neigh = neighbors[i]
            sum_neigh = 0
            # vectorized sum would be faster but we keep explicit loop for clarity
            for j in neigh:
                sum_neigh += spins[j]
            dE = 2.0 * J * si * sum_neigh  # ΔE for flipping spin i
            if dE <= 0.0 or np.random.rand() < math.exp(-dE / T):
                spins[i] = -si
                H += dE

        if sweep >= params.burn and ((sweep - params.burn) % params.measure_every == 0):
            m = spins.mean()
            e = H / N
            m_series[k] = m
            e_series[k] = e
            k += 1

    # statistics
    m_abs_avg = np.mean(np.abs(m_series))
    m_avg = float(np.mean(m_series))
    m2_avg = float(np.mean(m_series**2))
    e_avg = float(np.mean(e_series))
    e2_avg = float(np.mean(e_series**2))

    return SimResults(
        N=N, T=T,
        m_abs_avg=m_abs_avg, m_avg=m_avg, m2_avg=m2_avg,
        e_avg=e_avg, e2_avg=e2_avg,
        magnetizations=m_series, energies=e_series
    )

# ---------------------------
# Experiment runner
# ---------------------------

def temperature_grid(Tmin: float, Tmax: float, nT: int) -> np.ndarray:
    return np.linspace(Tmin, Tmax, nT)

def run_suite(geom: str,
              Ns: List[int],
              Ts: np.ndarray,
              sweeps: int,
              burn: int,
              measure_every: int,
              seed: int | None,
              extra_per_node: int = 2) -> Dict[int, Dict[float, SimResults]]:
    """
    Run across sizes and temperatures for one geometry.
    Returns nested dict: results[N][T] = SimResults
    """
    out: Dict[int, Dict[float, SimResults]] = {}
    for N in Ns:
        # Build neighbors once per N for geometry B (kept fixed during all T for that N)
        if geom == "ring":
            neighbors = build_ring_neighbors(N)
        elif geom == "ring_plus_random":
            neighbors = build_ring_plus_random(N, extra_per_node=extra_per_node, seed=seed)
        else:
            raise ValueError("Unknown geometry")

        out[N] = {}
        for T in Ts:
            spins0 = initial_spins(N, init="random")
            params = SimParams(N=N, T=float(T), sweeps=sweeps, burn=burn,
                               measure_every=measure_every, J=1.0, seed=seed, init="random")
            res = metropolis(spins0, neighbors, params)
            out[N][float(T)] = res
    return out

def susceptibility(res: SimResults) -> float:
    """
    χ = N/T * (⟨m^2⟩ - ⟨m⟩^2)
    Note: m is the signed magnetization per spin.
    """
    return res.N / res.T * max(0.0, (res.m2_avg - res.m_avg ** 2))

def specific_heat(res: SimResults) -> float:
    """
    c_V = (⟨e^2⟩ - ⟨e⟩^2) / T^2
    e is energy per spin.
    """
    return max(0.0, (res.e2_avg - res.e_avg ** 2)) / (res.T ** 2)

def estimate_Tc_for_each_N(results: Dict[int, Dict[float, SimResults]],
                           which: str = "chi") -> Dict[int, float]:
    """
    For each N, pick T where 'which' quantity peaks.
    which ∈ {'chi','cv'}.
    """
    TcN: Dict[int, float] = {}
    for N, res_by_T in results.items():
        Ts = sorted(res_by_T.keys())
        vals = []
        for T in Ts:
            r = res_by_T[T]
            if which == "chi":
                vals.append(susceptibility(r))
            elif which == "cv":
                vals.append(specific_heat(r))
            else:
                raise ValueError("which must be 'chi' or 'cv'")
        vals = np.array(vals, dtype=float)
        idx = int(np.argmax(vals))
        TcN[N] = Ts[idx]
    return TcN

# ---------------------------
# Plotting
# ---------------------------

def plot_m_abs(results: Dict[int, Dict[float, SimResults]], title_prefix: str, out_dir: str, fname: str):
    plt.figure()
    for N, res_by_T in sorted(results.items()):
        Ts = np.array(sorted(res_by_T.keys()))
        mabs = np.array([res_by_T[T].m_abs_avg for T in Ts], dtype=float)
        plt.plot(Ts, mabs, marker="o", label=f"N={N}")
    plt.xlabel("Temperature T")
    plt.ylabel("|m| (time-avg)")
    plt.title(f"{title_prefix}: |m| vs T")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

def plot_chi(results: Dict[int, Dict[float, SimResults]], title_prefix: str, out_dir: str, fname: str):
    plt.figure()
    for N, res_by_T in sorted(results.items()):
        Ts = np.array(sorted(res_by_T.keys()))
        chis = np.array([susceptibility(res_by_T[T]) for T in Ts], dtype=float)
        plt.plot(Ts, chis, marker="o", label=f"N={N}")
    plt.xlabel("Temperature T")
    plt.ylabel("Susceptibility χ")
    plt.title(f"{title_prefix}: χ vs T")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

def plot_Tc_vs_invN(Tc_ring: Dict[int, float], Tc_rand: Dict[int, float], out_dir: str, fname: str):
    plt.figure()
    Ns_r = sorted(Tc_ring.keys())
    Ns_b = sorted(Tc_rand.keys())
    if Ns_r:
        x_r = np.array([1.0 / N for N in Ns_r], dtype=float)
        y_r = np.array([Tc_ring[N] for N in Ns_r], dtype=float)
        plt.plot(x_r, y_r, marker="o", label="ring (1D)")
    if Ns_b:
        x_b = np.array([1.0 / N for N in Ns_b], dtype=float)
        y_b = np.array([Tc_rand[N] for N in Ns_b], dtype=float)
        plt.plot(x_b, y_b, marker="s", label="ring + random links")
    plt.xlabel("1/N")
    plt.ylabel("Estimated T_c(N) (from χ peak)")
    plt.title("Finite-size scaling of T_c(N)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Ising model (Complex Physics Midterm 1)")
    p.add_argument("--geom", choices=["ring", "ring_plus_random", "both"], default="both")
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--Tmin", type=float, default=0.5)
    p.add_argument("--Tmax", type=float, default=4.0)
    p.add_argument("--nT", type=int, default=15)
    p.add_argument("--sweeps", type=int, default=6000)
    p.add_argument("--burn", type=int, default=2500)
    p.add_argument("--measure_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--extra_per_node", type=int, default=2, help="extra random links per node in geometry B")
    p.add_argument("--quick", action="store_true", help="run a quicker, lighter simulation")
    args = p.parse_args()

    if args.quick:
        # Faster settings for smoke test
        args.sizes = [64, 128]
        args.nT = 9
        args.sweeps = 3000
        args.burn = 1200
        args.measure_every = 5

    out_dir = ensure_out_dir("out")
    Ts = temperature_grid(args.Tmin, args.Tmax, args.nT)

    all_results = {}

    if args.geom in ("ring", "both"):
        res_ring = run_suite("ring", args.sizes, Ts, args.sweeps, args.burn, args.measure_every, args.seed)
        all_results["ring"] = res_ring
        plot_m_abs(res_ring, "Geometry A (1D ring)", out_dir, "ring__magnetization_vs_T.png")
        plot_chi(res_ring, "Geometry A (1D ring)", out_dir, "ring__susceptibility_vs_T.png")

    if args.geom in ("ring_plus_random", "both"):
        res_rand = run_suite("ring_plus_random", args.sizes, Ts, args.sweeps, args.burn, args.measure_every, args.seed, args.extra_per_node)
        all_results["ring_plus_random"] = res_rand
        plot_m_abs(res_rand, "Geometry B (ring + random links)", out_dir, "rand__magnetization_vs_T.png")
        plot_chi(res_rand, "Geometry B (ring + random links)", out_dir, "rand__susceptibility_vs_T.png")

    # Optional finite-size scaling
    if "ring" in all_results and "ring_plus_random" in all_results:
        Tc_ring = estimate_Tc_for_each_N(all_results["ring"], which="chi")
        Tc_rand = estimate_Tc_for_each_N(all_results["ring_plus_random"], which="chi")
        plot_Tc_vs_invN(Tc_ring, Tc_rand, out_dir, "Tc_vs_invN.png")

        # Save numeric Tc(N) to JSON
        with open(os.path.join(out_dir, "Tc_estimates.json"), "w", encoding="utf-8") as f:
            json.dump({"ring": Tc_ring, "ring_plus_random": Tc_rand}, f, indent=2)

    print("Done. Plots saved in ./out/")

if __name__ == "__main__":
    main()
'''
py_path = os.path.join(base, "ising_midterm.py")
with open(py_path, "w", encoding="utf-8") as f:
    f.write(py_code)


# ---------------------------
# 2) Write the Markdown report
# ---------------------------
report = r'''# Complex Physics — Midterm 1 (Ising Model) — Solutions

This write-up accompanies the Python code in `ising_midterm.py`. It addresses all required tasks for the midterm, and includes the optional extra-credit reasoning. The problem statements are from the exam handout (Complex Physics — Midterm Exam 1, Sneppen & Haerter, Sept 5, 2024).fileciteturn0file0

---

## Problem 1 — Implement the Ising model and plot ⟨s_i⟩ vs. T (5 pt)

**What we simulate.** We place an Ising spin \(s_i=\pm 1\) on each site and use the ferromagnetic zero-field Hamiltonian
\[
H = -J \sum_{\langle ij\rangle} s_i s_j,\quad J>0,
\]
with bonds \(\langle ij\rangle\) defined by the geometry. Two geometries are required: (A) a 1D periodic ring with nearest neighbors; (B) the ring plus two additional random links per node kept *fixed* during a run; links are symmetric.fileciteturn0file0

**Algorithm.** We use a standard Metropolis update: pick a site \(i\) at random, compute the energy change \(\Delta E = 2J s_i \sum_{j\in \mathcal N(i)} s_j\) for flipping \(s_i\!\to\!-s_i\), accept with probability \(\min(1, e^{-\Delta E/T})\). One *sweep* is \(N\) such proposals.

**Order parameter.** The symmetry of \(H\) implies \(\langle m\rangle=\langle \frac{1}{N}\sum_i s_i\rangle=0\) for finite \(N\) unless symmetry is explicitly broken. Therefore, we report the time-average of the **absolute** magnetization \(|m|\) after burn-in:
\[
\langle |m|\rangle_T = \left\langle \left| \frac{1}{N}\sum_i s_i \right|\right\rangle.
\]
This is exactly what the code records and plots as “\|m\| vs T”.

**How to run (quick demo).**
```bash
python ising_midterm.py --quick
