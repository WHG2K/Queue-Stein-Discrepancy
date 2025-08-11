from src.MM1 import mm1_sim
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from src.discrepancy import wasserstein_MM1
import argparse
from dotenv import load_dotenv

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    parser.add_argument("--data", type=str, default="", help='data file name')

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
    lmax = .1
    mumax=.1
    # lambda_t = lambda t: .09
    def lambda_t(t):
        return 0.05 if t <= T/2 else 0.07
    mu_t = lambda t: .1 
    event_times, Q = mm1_sim(T,lambda_t,mu_t, lmax,mumax)
    
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
    step = 10_000             # shift window by 1000 samples each time
    rho = 0.07 / 0.10

    W1_win = []
    x_axis = []

    # slide over steady-state samples
    for s in range(0, len(steady) - window_size + 1, step):
        window = steady[s : s + window_size]
        # compute W1 on the current window
        W1_win.append(wasserstein_MM1(window, rho, tol=1e-8))
        # record the center of window (or end) as x-coordinate
        x_axis.append(s + window_size)

    # plot sliding-window W1
    plt.figure()
    plt.plot(x_axis, W1_win, marker='o')
    plt.xlabel("Sample index (end of window)")
    plt.ylabel("1-Wasserstein distance")
    plt.title("Sliding-window convergence of W1")
    plt.grid(True)
    plt.show()

    print(max(Qvals))


    # # find the index of the maximum W1
    # max_idx = int(np.argmax(W1_win))
    # max_W1 = W1_win[max_idx]
    # print(f"Maximum W1 = {max_W1:.4f} at window ending at sample {x_axis[max_idx]}")

    # # compute the start position of that window
    # start_idx = x_axis[max_idx] - window_size
    # end_idx   = x_axis[max_idx]

    # # extract the window with the largest W1
    # window_max = steady[start_idx : end_idx]

    # # plot histogram of that window
    # plt.figure()
    # plt.hist(
    #     window_max,
    #     bins=range(min(window_max), max(window_max) + 2),
    #     align="left",
    #     rwidth=0.8
    # )
    # plt.xlabel("Queue length")
    # plt.ylabel("Frequency")
    # plt.title(
    #     f"Histogram of window with max W1\n"
    #     f"samples {start_idx}–{end_idx}, W1={max_W1:.2f}"
    # )
    # plt.grid(True)
    # plt.show()





    # plt.bar(values, probs)
    # plt.hist(Qvals,bins=75)
    # plt.plot(Qvals)

    # plt.show()