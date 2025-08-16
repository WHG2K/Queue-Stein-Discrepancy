from src.MM1 import mm1_sim
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from src.gsd_mm1 import GSD_MM1_V1, GSD_MM1_V2
import argparse
from dotenv import load_dotenv
from typing import List, Tuple
from src.utils import list_to_distr, extend_to_full_support, W1_MM1
from src.ksd_mm1 import KSD_MM1
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    # parser.add_argument("--data", type=str, default="", help='data file name')

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # choose node
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')

    T = 1_000_000
    lmd_1 = 0.09
    lmd_2 = 0.099
    mu = 0.1
    lmax = .1
    mumax=.1
    # lambda_t = lambda t: .09
    def lambda_t(t):
        return lmd_1 if t <= T/2 else lmd_2
    mu_t = lambda t: mu
    event_times, Q = mm1_sim(T,lambda_t, mu_t, lmax,mumax)
    
    ls = np.linspace(0,T,T+1)
    arglist = np.searchsorted(event_times,ls,side='right') -1
    Qvals = [Q[i] for i in arglist]
    counts = Counter(Qvals)
    total = sum(counts.values())
    values, probs = zip(*sorted((k, v / total) for k, v in counts.items()))


    ##########################################################################
    ####################### Plot the True W1 #################################
    ##########################################################################

    # discard initial burn-in
    burn_frac = 0.0
    start = int(len(Qvals) * burn_frac)
    steady = Qvals[start:]

    # sliding window parameters
    window_size = 10_000      # number of samples in each window
    step = 1_000             # shift window by 1000 samples each time
    rho_2 = lmd_2 / mu

    W1_win = []
    GSD_V1_lp_win = []
    GSD_V1_win = []
    GSD_V2_lp_win = []
    GSD_V2_win = []
    KSD_exp_win = []
    KSD_imq_win = []
    x_axis = []
    

    print(f"rho_1: {lmd_1 / mu}, rho_2: {lmd_2 / mu}")


    t_start = time.time()
    # slide over steady-state samples
    for s in range(0, len(steady) - window_size + 1, step):
        window = steady[s : s + window_size]
        a, q = list_to_distr(window)
        max_a = max(a)
        a_, q_ = a + [max_a + 100], q + [0]
        a_full, q_full = extend_to_full_support(a_, q_)
        # a_full, q_full = extend_to_full_support(a, q)
        # compute discrepancies on the current window
        W1_win.append(W1_MM1(a, q, rho_2))
        GSD_V1_lp_win.append(GSD_MM1_V1(a, q, lmd_2, mu, verbose=False)[0])
        GSD_V1_win.append(GSD_MM1_V1(a_full, q_full, lmd_2, mu, verbose=False)[0])
        GSD_V2_lp_win.append(GSD_MM1_V2(a, q, lmd_2, mu, verbose=False)[0])
        GSD_V2_win.append(GSD_MM1_V2(a_full, q_full, lmd_2, mu, verbose=False)[0])
        KSD_exp_win.append(KSD_MM1(a, q, lmd_2, mu, kernel="exp", kernel_params={"beta": 0.95}, verbose=False))
        KSD_imq_win.append(KSD_MM1(a, q, lmd_2, mu, kernel="imq", kernel_params={"c": 1.0, "alpha": 0.5}, verbose=False))
        # record the center of window (or end) as x-coordinate
        x_axis.append(s + window_size)
    t_end = time.time()
    print(f"Elapsed time: {t_end - t_start} seconds")

    # print(x_axis[0:10])
    # print(SD_V1_win[0:10])
    # print(SD_V1_lp_win[0:10])
    # print(SD_V2_win[0:10])
    # print(SD_V2_lp_win[0:10])


    # ===================== PLOTS =====================
    folder = "./data/test_mm1_sim/"

    # ---- V1: d^{sd} and d^{sd}_{lp} ----
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, GSD_V1_win,    label=r'$d^{\mathrm{sd}}$', linewidth=1.5)
    plt.plot(x_axis, GSD_V1_lp_win, label=r'$d^{\mathrm{sd}}_{\mathrm{lp}}$', linestyle='--', linewidth=1.5)
    # Optional: mark the parameter switch point (in window-end index scale)
    # plt.axvline(T/2, color='k', linestyle=':', linewidth=1.0)
    plt.xlabel("Sample index (end of window)")
    plt.ylabel("Discrepancy")
    # plt.title("Stein discrepancy (V1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Optional: save to PDF
    filename1 = f"sd_V1_lmd1_{lmd_1}_lmd2_{lmd_2}_mu_{mu}.pdf"
    plt.savefig(folder + filename1, format="pdf", dpi=300, bbox_inches="tight")

    # ---- V2: \widetilde{d}^{sd} and \widetilde{d}^{sd}_{lp} ----
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, GSD_V2_win,    label=r'$\widetilde{d}^{\mathrm{sd}}$', linewidth=1.5)
    plt.plot(x_axis, GSD_V2_lp_win, label=r'$\widetilde{d}^{\mathrm{sd}}_{\mathrm{lp}}$', linestyle='--', linewidth=1.5)
    # Optional: mark the parameter switch point
    # plt.axvline(T/2, color='k', linestyle=':', linewidth=1.0)
    plt.xlabel("Sample index (end of window)")
    plt.ylabel("Discrepancy")
    # plt.title("Stein discrepancy (V2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Optional: save to PDF
    filename2 = f"sd_V2_lmd1_{lmd_1}_lmd2_{lmd_2}_mu_{mu}.pdf"
    plt.savefig(folder + filename2, format="pdf", dpi=300, bbox_inches="tight")

    # ---- KSD: exp and imq ----
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, KSD_exp_win, label=r'KSD exp', linewidth=1.5)
    plt.plot(x_axis, KSD_imq_win, label=r'KSD imq', linewidth=1.5)
    plt.xlabel("Sample index (end of window)")
    plt.ylabel("Discrepancy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # filename3 = f"ksd_lmd1_{lmd_1}_lmd2_{lmd_2}_mu_{mu}.pdf"
    # plt.savefig(folder + filename3, format="pdf", dpi=300, bbox_inches="tight")

    # Show figures
    # plt.show()