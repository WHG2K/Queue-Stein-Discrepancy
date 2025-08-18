# jackson_discrepancies.py
# Time-varying Stein discrepancy for a K=3 Jackson network (only KSD; no GSD/W1)

from src.sim_JacksonNet import jackson_sim
from src.ksd_JacksonNet import KSD_Jackson
import numpy as np
import matplotlib.pyplot as plt
import argparse
from dotenv import load_dotenv
import time
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Jackson network KSD over time (sliding windows).")
    parser.add_argument('--node', type=str, default=None, help='which node env file to load (e.g., gpu1 -> gpu1.env)')
    return parser.parse_args()

if __name__ == "__main__":

    # ------------- args & env -------------
    args = parse_arguments()
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')

    # ------------- parameters (match your latest script) -------------
    T = 1000_00
    np.random.seed(2025)

    # service: mu_i = 1 (constant)
    mu = 1.0
    mus_t = [lambda t, r=mu: r for _ in range(3)]
    mumax = [mu, mu, mu]

    # routing P (first column all zeros)
    P = np.array([[0.0, 0.5, 0.4],
                  [0.0, 0.0, 0.3],
                  [0.0, 0.2, 0.0]], dtype=float)

    # external arrivals: only lambda_1 changes at T/2; lambda_2, lambda_3 constant
    lam1_low, lam1_high = 0.6, 0.7
    lam2_const, lam3_const = 0.2, 0.2

    def lambda1_t(t): 
        return lam1_low if t < T/2 else lam1_high
    lambda2_t = lambda t: lam2_const
    lambda3_t = lambda t: lam3_const
    lambdas_t = [lambda1_t, lambda2_t, lambda3_t]

    # thinning bounds
    lmax = [max(lam1_low, lam1_high), lam2_const, lam3_const]

    # ------------- simulate trajectory -------------
    times, Xs = jackson_sim(
        T=T,
        lambdas_t=lambdas_t,
        mus_t=mus_t,
        P=P,
        lmax=lmax,
        mumax=mumax,
        init=[0, 0, 0],
    )
    # Xs[k] is the K-vector state AFTER event at times[k]

    # ------------- sample on a regular grid (like your mm1 version) -------------
    # build piecewise-constant trajectory at integer times 0..T
    grid = np.arange(0, T + 1, 1, dtype=float)
    idx = np.searchsorted(times, grid, side='right') - 1
    idx = np.clip(idx, 0, len(Xs) - 1)
    Xvals = [tuple(Xs[i]) for i in idx]  # list of (x1,x2,x3) at each integer time

    # ------------- sliding window setup -------------
    burn_frac = 0.0
    start = int(len(Xvals) * burn_frac)
    steady = Xvals[start:]

    window_size = 10_000      # window of samples (states)
    step = 1_000              # stride
    x_axis = []

    # Target parameters = AFTER-change (consistent with your MM1 script)
    lambdas_target = np.array([lam1_high, lam2_const, lam3_const], dtype=float)
    mus_target     = np.array([mu, mu, mu], dtype=float)
    P_target       = P.copy()

    KSD_exp_win = []
    KSD_imq_win = []
    KSD_exp_ham_win = []

    print(f"[info] Using AFTER-change target: lambdas={lambdas_target}, mus={mus_target}, mu={mu}")

    t0 = time.time()
    for s in range(0, len(steady) - window_size + 1, step):
        window_states = steady[s : s + window_size]        # list of tuples
        # Uniform weights; KSD_Jackson will merge duplicates internally
        q = np.full(len(window_states), 1.0 / len(window_states), dtype=float)
        # Compute KSD with exponential kernel (beta can be tuned if you like)
        ksd_exp = KSD_Jackson(
            a=window_states,
            q=q,
            lambdas=lambdas_target,
            mus=mus_target,
            P=P_target,
            kernel="exp",
            kernel_params={"beta": 0.95},
            verbose=False
        )
        ksd_imq = KSD_Jackson(
            a=window_states,
            q=q,
            lambdas=lambdas_target,
            mus=mus_target,
            P=P_target,
            kernel="imq",
            kernel_params={"c": 1.0, "alpha": 0.5},
            verbose=False
        )
        ksd_exp_ham = KSD_Jackson(
            a=window_states,
            q=q,
            lambdas=lambdas_target,
            mus=mus_target,
            P=P_target,
            kernel="exp_hamming",
            verbose=False
        )
        KSD_exp_win.append(ksd_exp)
        KSD_imq_win.append(ksd_imq)
        KSD_exp_ham_win.append(ksd_exp_ham)
        x_axis.append(s + window_size)  # end-of-window index on the regular grid
    t1 = time.time()
    print(f"[timing] sliding KSD done in {t1 - t0:.2f}s with {len(KSD_exp_win)} windows")

    # ------------- save plots -------------
    folder = (
        "./data/compare_discrepancies_jackson/"
        f"lam1_{lam1_low}_to_{lam1_high}_lam2_{lam2_const}_lam3_{lam3_const}_mu_{mu}/"
    )
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_axis, KSD_exp_win, label="KSD exp", linewidth=1.5, color="tab:green")
    ax.plot(x_axis, KSD_imq_win, label="KSD imq", linewidth=1.5, color="tab:blue")
    ax.plot(x_axis, KSD_exp_ham_win, label="KSD exp ham", linewidth=1.5, color="tab:red")
    ax.set_xlabel("Sample index (end of window)")
    ax.set_ylabel("Discrepancy")
    ax.set_title("Jackson network KSD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"ksd_exp_lam1_{lam1_low}_to_{lam1_high}_lam23_{lam2_const}_{lam3_const}_mu_{mu}.pdf"
    fig.savefig(os.path.join(folder, fname), format="pdf", dpi=300, bbox_inches="tight")
