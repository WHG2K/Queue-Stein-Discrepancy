import numpy as np
from scipy.stats import wasserstein_distance
import math
import gurobipy as gp
from gurobipy import GRB
from src.discrepancy import SD_MM1_V1, SD_MM1_V2, W1_MM1
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    # parser.add_argument("--data", type=str, default="", help='data file name')

    return parser.parse_args()


def generate_discrete_dist(rho, gap=2):
    """Generate a discrete distribution with fixed gap."""
    max_support = 3 / (1 - rho)
    # Ensure integer support values
    support = np.arange(0, int(max_support) + 1, gap, dtype=int)
    # Generate random probabilities
    probs = np.random.rand(len(support))
    # Normalize probabilities to sum to 1
    probs /= probs.sum()
    return support.tolist(), probs.tolist()


def extend_to_full_support(a, q):
    """
    Given a nonnegative integer list `a` (no duplicates) and probabilities `q`,
    return (a_full, q_full) where:
      - a_full = [0, 1, ..., max(a)]
      - q_full has zeros inserted at missing support points so that it aligns with a_full.
    The original probabilities are not renormalized.
    """
    # Convert to arrays and basic checks
    a = np.asarray(a, dtype=int)
    q = np.asarray(q, dtype=float)

    if a.size == 0:
        # Nothing to extend
        return [], []

    if (a < 0).any():
        raise ValueError("`a` must contain nonnegative integers only.")

    # Check length match
    if a.shape[0] != q.shape[0]:
        raise ValueError("`a` and `q` must have the same length.")

    # Ensure no duplicates in `a`
    if len(np.unique(a)) != len(a):
        raise ValueError("`a` must have no duplicate elements.")

    # Sort by support just in case
    order = np.argsort(a)
    a = a[order]
    q = q[order]

    # Build full support from 0 to max(a)
    max_a = int(a.max())
    a_full = np.arange(0, max_a + 1, dtype=int)

    # Initialize q_full with zeros
    q_full = np.zeros_like(a_full, dtype=float)

    # Map existing probabilities to correct positions
    # Since a_full[i] == i, we can place by index directly
    q_full[a] = q

    # Return lists if you prefer list outputs
    return a_full.tolist(), q_full.tolist()


def percentile_interval(values, lower=2.5, upper=97.5):
    """
    Compute percentile interval [lower, upper] of raw values (not CI of the mean).
    NaNs/inf are removed. Returns (p_lower, median, p_upper, n_used).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    p_lo = np.quantile(x, lower / 100.0)
    p_md = np.quantile(x, 0.5)
    p_hi = np.quantile(x, upper / 100.0)
    return p_lo, p_md, p_hi, n



if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # choose environment
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')

    m = gp.Model("test")
    del m

    # --- Shared Parameters ---
    lmd = .95
    mu = 1.0
    rho = lmd / mu

    # rho_values = [0.2, 0.5, 0.8]  # 你可以改成需要的 rho 列表
    n_repeat = 100_000                  # 每个 rho 生成多少次分布

    ratios_sd1 = []
    ratios_sd2 = []

    for i in range(n_repeat):
        a, q = generate_discrete_dist(rho, gap=10)
        a_full, q_full = extend_to_full_support(a, q)

        d_sd1, status_sd1 = SD_MM1_V1(q, a, lmd, mu, verbose=False)
        d_sd2, status_sd2 = SD_MM1_V2(q, a, lmd, mu, verbose=False)
        d_sd2_true, status_sd2_true = SD_MM1_V2(q_full, a_full, lmd, mu, verbose=False)

        d_w = W1_MM1(q, a, rho, padding=100)
        d_sd1_true = d_w

        # Build ratios; skip invalid denominators
        r1 = np.nan if (d_sd1_true is None or d_sd1_true == 0) else (d_sd1 / d_sd1_true)
        r2 = np.nan if (d_sd2_true is None or d_sd2_true == 0) else (d_sd2 / d_sd2_true)

        ratios_sd1.append(r1)
        ratios_sd2.append(r2)

    # Percentile intervals of the raw ratios (default: 2.5% and 97.5%)
    lo1, md1, hi1, n1 = percentile_interval(ratios_sd1, lower=2.5, upper=97.5)
    lo2, md2, hi2, n2 = percentile_interval(ratios_sd2, lower=2.5, upper=97.5)

    print("=== Percentile intervals of raw ratios (2.5%, 50%, 97.5%) ===")
    print(f"d_sd1 / d_sd1_true: [{lo1:.6g}, {md1:.6g}, {hi1:.6g}]  (n={n1})")
    print(f"d_sd2 / d_sd2_true: [{lo2:.6g}, {md2:.6g}, {hi2:.6g}]  (n={n2})")



    # for i in range(n_repeat):
    #     a, q = generate_discrete_dist(rho, gap=10)
    #     a_full, q_full = extend_to_full_support(a, q)
    #     # print(a)
    #     # print(q)
    #     d_sd1, status_sd1 = SD_MM1_V1(q, a, lmd, mu, verbose=False)
    #     d_sd2, status_sd2 = SD_MM1_V2(q, a, lmd, mu, verbose=False)
    #     d_sd2_true, status_sd2_true = SD_MM1_V2(q_full, a_full, lmd, mu, verbose=False)
    #     d_w = W1_MM1(q, a, rho, padding=100)
    #     d_sd1_true = d_w
        
    #     # 这里写你自己的处理逻辑
    #     # print(f"d_sd: {d_sd}, d_w: {d_w}")
    #     print(f"d_sd1/d_sd1_true: {d_sd1/d_sd1_true}")
    #     print(f"d_sd2/d_sd2_true: {d_sd2/d_sd2_true}")

    