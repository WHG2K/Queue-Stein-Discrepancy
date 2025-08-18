# -*- coding: utf-8 -*-
"""
Jackson network simulation (K-node M/M/1), NHPP thinning style
- Each node i has external arrivals with rate lambda_i(t) and service rate mu_i(t)
- Upon service completion at node i, job routes to node j with prob P[i,j],
  or departs with prob 1 - sum_j P[i,j].
- Mirrors your mm1_sim structure: event-driven, keeps a global time list and
  queue-length snapshots after each event.
"""

import numpy as np
from typing import Callable, List, Tuple

# ---------- NHPP utility (same idea as your code) ----------

def nhpp_next(t: float, rate_t: Callable[[float], float], rmax: float) -> float:
    """Next arrival time after t for an NHPP with rate_t using thinning; rmax is an upper bound."""
    if rmax <= 0:
        raise ValueError("rmax must be positive for thinning.")
    next_t = t
    while True:
        next_t += np.random.exponential(1.0 / rmax)
        if rate_t(next_t) / rmax > np.random.uniform():
            return next_t

# ---------- Jackson network simulator ----------

def jackson_sim(
    T: float,
    lambdas_t: List[Callable[[float], float]],   # [lambda_1(t), ..., lambda_K(t)]
    mus_t: List[Callable[[float], float]],       # [mu_1(t), ..., mu_K(t)]
    P: np.ndarray,                               # shape (K,K), routing probs P[i,j]
    lmax: List[float],                           # upper bounds for lambda_i(t)
    mumax: List[float],                          # upper bounds for mu_i(t)
    init: List[int] = None                       # initial queue lengths (len K), default all zeros
) -> Tuple[List[float], List[List[int]]]:
    """
    Generate a trajectory on [0, T] for a K-node Jackson network (each node M/M/1).
    Returns:
        times : event times (including 0)
        Xs    : list of K-vectors of queue lengths after each event (Xs[k] is state after times[k])
    Notes:
        - When a queue goes from 0 to 1 (via external arrival or routing-in), we schedule a service NHPP for that node.
        - Service events are generated regardless, but only have effect if queue length > 0 at event time.
          (This mirrors your mm1_sim guard; when empty, we just resample the next service candidate.)
    """
    K = len(lambdas_t)
    P = np.asarray(P, dtype=float).reshape(K, K)
    dep = 1.0 - P.sum(axis=1)  # departure probabilities per node (ensure in [0,1])
    dep = np.clip(dep, 0.0, 1.0)

    # init state
    if init is None:
        X = np.zeros(K, dtype=int)
    else:
        X = np.asarray(init, dtype=int)
        if X.shape != (K,) or np.any(X < 0):
            raise ValueError("init must be a nonnegative length-K vector")

    # initialize time and event clocks
    t = 0.0
    times = [0.0]
    Xs = [X.copy()]

    # next external arrival times per node
    narr = np.array([nhpp_next(t, lambdas_t[i], lmax[i]) for i in range(K)], dtype=float)
    # next service-completion times per node (schedule only if server is/will be busy)
    nser = np.empty(K, dtype=float)
    for i in range(K):
        # If queue > 0 at t=0, schedule a service NHPP; else infinite (no pending service).
        if X[i] > 0:
            nser[i] = nhpp_next(t, mus_t[i], mumax[i])
        else:
            nser[i] = np.inf

    # helper to (re)start service clock if server was idle and a job arrives
    def ensure_service_clock(i: int, tnow: float):
        if np.isinf(nser[i]) and X[i] > 0:
            nser[i] = nhpp_next(tnow, mus_t[i], mumax[i])

    # main simulation loop
    while t < T:
        # pick the next event among all external arrivals and service completions
        i_arr = int(np.argmin(narr))         # argmin arrival
        i_ser = int(np.argmin(nser))         # argmin service
        t_next_arr = narr[i_arr]
        t_next_ser = nser[i_ser]
        t_next = min(t_next_arr, t_next_ser)

        if t_next == np.inf:  # no events scheduled (shouldn't happen if lmax>0)
            break
        if t_next > T:
            break

        t = t_next

        if t_next_arr <= t_next_ser:
            # --- External arrival at node i_arr ---
            X_before = X[i_arr]
            X[i_arr] += 1

            # Arrival process: reschedule next external arrival for node i_arr
            narr[i_arr] = nhpp_next(t, lambdas_t[i_arr], lmax[i_arr])

            # If server was idle before this arrival, start its service clock
            if X_before == 0:
                nser[i_arr] = nhpp_next(t, mus_t[i_arr], mumax[i_arr])

        else:
            # --- Service completion at node i_ser (only effective if X[i_ser] > 0) ---
            if X[i_ser] > 0:
                X[i_ser] -= 1

                # Routing: decide whether departure or to which node it goes
                u = np.random.uniform()
                if u < dep[i_ser]:
                    # departure: job leaves the network
                    pass
                else:
                    # internal routing: pick destination j by P[i_ser, :]
                    # Compute cumulative probs on-the-fly
                    row = P[i_ser, :]
                    cdf = np.cumsum(row)
                    j = int(np.searchsorted(cdf, u - dep[i_ser], side='right'))
                    # route to node j
                    X_before_j = X[j]
                    X[j] += 1
                    # if node j was idle, (re)start service clock
                    ensure_service_clock(j, t)

                # After service completion, regardless of X[i_ser] remaining size,
                # reschedule the next service candidate for node i_ser.
                nser[i_ser] = nhpp_next(t, mus_t[i_ser], mumax[i_ser])
                # If the queue has become empty, we keep the service candidate,
                # but it will have no effect unless a job arrives (mirrors your guard style).

            else:
                # Server completion fired while empty -> ignore effect, reschedule next candidate
                nser[i_ser] = nhpp_next(t, mus_t[i_ser], mumax[i_ser])

        # record snapshot
        times.append(t)
        Xs.append(X.copy())

    return times, Xs