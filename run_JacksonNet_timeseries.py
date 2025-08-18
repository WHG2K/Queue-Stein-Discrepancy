#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_k3_timeseries_rho_gt_07.py
K=3 Jackson network, mu=0.1 at all nodes.
Only lambda_1(t) is piecewise: low on [0, T/2), high on [T/2, T];
lambda_2(t), lambda_3(t) are constant.
Routing P has first column all zeros so changing lambda_1 impacts nodes 2 and 3.
"""

import numpy as np
import matplotlib.pyplot as plt

# If sim_jackson_net.py is under src/, use:
# from src.sim_jackson_net import jackson_sim
# Otherwise, if it is in the same directory, use:
from src.sim_JacksonNet import jackson_sim


def piecewise_rate(low: float, high: float, T: float):
    """Return r(t) that equals 'low' on [0, T/2) and 'high' on [T/2, T]."""
    half = 0.5 * T
    return lambda t, low=low, high=high, half=half: (low if t < half else high)


def constant_rate(val: float):
    """Return r(t) == val."""
    return lambda t, val=val: val


def main():
    # ----- 1) Horizon and seed -----
    T = 1_000
    np.random.seed(2025)

    # ----- 2) Service rates (all equal, constant) -----
    mu = 1
    mus_t = [constant_rate(mu) for _ in range(3)]
    mumax = [mu, mu, mu]

    # ----- 3) Routing with first column = 0 (only node 1 feeds others) -----
    # P = [[0.0, 0.5, 0.4],
    #      [0.0, 0.0, 0.3],
    #      [0.0, 0.2, 0.0]]
    P = np.array([[0.0, 0.5, 0.4],
                  [0.0, 0.0, 0.3],
                  [0.0, 0.2, 0.0]], dtype=float)

    # ----- 4) External arrivals: only lambda_1 changes; lambda_2, lambda_3 constant -----
    # Use the pair that causes significant changes when only lambda_1 varies:
    lam1_low, lam1_high = 0.5, 0.7
    lam2_const, lam3_const = 0.2, 0.2

    lambdas_t = [
        piecewise_rate(lam1_low, lam1_high, T),  # node 1: low then high
        constant_rate(lam2_const),               # node 2: constant
        constant_rate(lam3_const),               # node 3: constant
    ]
    lmax = [max(lam1_low, lam1_high), lam2_const, lam3_const]  # thinning bounds

    # ----- 5) Sanity check: rho before/after (low/high for lambda_1 only) -----
    I = np.eye(3)
    inv = np.linalg.inv(I - P)

    lam_ext_low  = np.array([lam1_low,  lam2_const, lam3_const])
    lam_ext_high = np.array([lam1_high, lam2_const, lam3_const])

    lam_eff_low  = lam_ext_low  @ inv
    lam_eff_high = lam_ext_high @ inv

    rho_low  = lam_eff_low  / mu
    rho_high = lam_eff_high / mu

    print("[check] effective arrivals (before / after):")
    print("  low : ", lam_eff_low)
    print("  high: ", lam_eff_high)
    print("[check] rho (mu=0.1) (before / after):")
    print("  low : ", rho_low)   # ~ [0.500, 0.564, 0.569]
    print("  high: ", rho_high)  # ~ [0.700, 0.687, 0.686]

    # ----- 6) Run simulation -----
    init = [0, 0, 0]
    times, Xs = jackson_sim(
        T=T,
        lambdas_t=lambdas_t,
        mus_t=mus_t,
        P=P,
        lmax=lmax,
        mumax=mumax,
        init=init,
    )

    # ----- 7) Plot queue-length trajectories (unchanged) -----
    x1 = [x[0] for x in Xs]
    x2 = [x[1] for x in Xs]
    x3 = [x[2] for x in Xs]

    plt.figure(figsize=(10, 5))
    plt.step(times, x1, where='post', label='Node 1 (x1)')
    plt.step(times, x2, where='post', label='Node 2 (x2)')
    plt.step(times, x3, where='post', label='Node 3 (x3)')
    plt.axvline(0.5 * T, color='k', linestyle='--', alpha=0.5, label='T/2 change point')
    plt.xlabel("Time")
    plt.ylabel("Queue length")
    plt.title("K=3 Jackson network (mu=0.1): only λ1 changes at T/2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
