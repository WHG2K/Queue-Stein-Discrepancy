# ksd_JacksonNet.py
# Kernel Stein Discrepancy for Jackson networks on N^K

from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np

# ---------------------------------------------------------------------
# 1) Registry: name -> kernel builder
# ---------------------------------------------------------------------

# A Jackson kernel builder constructs the Stein kernel matrix U (N x N) given:
#   - X: unique states on Z_+^K, shape (N, K), int ndarray
#   - lambdas: arrival rates lambda_i, shape (K,)
#   - mus: service rates mu_i, shape (K,)
#   - P: routing matrix P_{ij}, shape (K, K)
#   - **params: kernel-specific parameters
KernelBuilder = Callable[..., np.ndarray]
REGISTRY: Dict[str, KernelBuilder] = {}

def register_kernel(name: str):
    def deco(func: KernelBuilder):
        if name in REGISTRY:
            raise ValueError(f"Kernel '{name}' already registered.")
        REGISTRY[name] = func
        return func
    return deco


# ---------------------------------------------------------------------
# 2) Utilities
# ---------------------------------------------------------------------

def _as_2d_int_states(a: Sequence[Sequence[int]]) -> np.ndarray:
    """Ensure states are a (n, K) int array with nonnegative entries."""
    X = np.asarray(a, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if np.any(X < 0) or np.any(np.abs(X - np.round(X)) > 1e-12):
        raise ValueError("All state coordinates must be nonnegative integers.")
    return X.astype(int)

def _normalize_weights(q: Optional[Sequence[float]], n: int) -> np.ndarray:
    """Return weights summing to 1, shape (n,)."""
    if q is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(q, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError("Length of q must match number of states.")
        s = w.sum()
        if s <= 0:
            raise ValueError("Sum of weights must be positive.")
        w = w / s
    return w

def _merge_duplicate_states(X: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge duplicate rows in X and sum their weights; return (X_unique, w)."""
    Xu, inv = np.unique(X, axis=0, return_inverse=True)
    w = np.zeros(Xu.shape[0], dtype=float)
    np.add.at(w, inv, q)
    w /= w.sum()
    return Xu, w

def _pairwise_diffs(X: np.ndarray) -> np.ndarray:
    """Return pairwise diffs D[r,c,:] = X[r,:] - X[c,:], shape (N, N, K)."""
    return X[:, None, :] - X[None, :, :]

def _unit(K: int, i: int) -> np.ndarray:
    """Unit vector e_i in R^K (0-based index)."""
    e = np.zeros(K, dtype=int)
    e[i] = 1
    return e


# ---------------------------------------------------------------------
# 3) Concrete kernel: separable exponential k(x,y)=prod beta^{|x_i-y_i|}
#    i.e., beta^{||x-y||_1}. Supports scalar (shared) beta in (0,1).
# ---------------------------------------------------------------------

@register_kernel("exp")
def build_U_exp(
    X: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    P: np.ndarray,
    *,
    beta: float = 0.95
) -> np.ndarray:
    """
    Stein kernel matrix U for a Jackson network with separable exponential kernel:
      k(x,y) = prod_{i=1}^K beta^{|x_i - y_i|} = beta^{||x-y||_1},  beta in (0,1).

    Generator (on x) uses arrivals lambda_i and services mu_i with routing P_{ij}.
    We form U = sum_{u in X-trans} sum_{v in Y-trans} r_x(u) r_y(v)
                 [ k(D + dx(u) - dy(v)) - k(D + dx(u)) - k(D - dy(v)) + k(D) ],
    where D = X - Y is the (N,N,K) pairwise difference tensor.
    Boundary masks 1_{x_i>0} and 1_{y_j>0} are applied for service transitions.

    Parameters
    ----------
    X : (N,K) int ndarray
        Unique states.
    lambdas : (K,) array_like
    mus : (K,) array_like
    P : (K,K) array_like
    beta : float in (0,1)
        Exponential (AR(1)) parameter.

    Returns
    -------
    U : (N,N) ndarray
        Steinized kernel matrix.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1).")

    X = np.asarray(X, dtype=int)
    N, K = X.shape
    lam = np.asarray(lambdas, dtype=float).reshape(K)
    mu  = np.asarray(mus, dtype=float).reshape(K)
    P   = np.asarray(P, dtype=float).reshape(K, K)

    # Departure probabilities per node
    dep = 1.0 - P.sum(axis=1)
    # Small numerical guard
    dep = np.clip(dep, 0.0, 1.0)

    # Pairwise differences D[r,c,:] = X[r,:] - X[c,:]
    D = _pairwise_diffs(X)   # (N,N,K)

    # Precompute log(beta) for stable exponentiation
    logb = np.log(beta)

    def k_of_shift(shift: np.ndarray) -> np.ndarray:
        """
        Evaluate k(D + shift) for a given integer vector 'shift' of length K.
        Uses: k = exp( log(beta) * ||D + shift||_1 ) = beta^{sum |D_i + shift_i|}.
        """
        # Broadcast add: (N,N,K) + (K,) -> (N,N,K)
        A = np.abs(D + shift.reshape(1, 1, K))
        # Sum over last axis -> (N,N)
        s = A.sum(axis=2)
        return np.exp(logb * s)

    # Base kernel matrix
    K0 = k_of_shift(np.zeros(K, dtype=int))

    # Masks for service availability (row depends on x, col depends on y)
    pos_x = (X > 0).astype(float)  # (N,K)
    pos_y = pos_x.copy()           # same states set used on columns

    # Build the list of x-transitions: (dx, rate_per_row_matrix)
    x_transitions: list[Tuple[np.ndarray, np.ndarray]] = []
    # Arrivals at node i: dx = +e_i, rate lambda_i, mask ones
    ones_mat = np.ones((N, N), dtype=float)
    for i in range(K):
        rate_row = lam[i] * ones_mat  # broadcast over columns
        x_transitions.append((_unit(K, i), rate_row))

    # Services at node i:
    for i in range(K):
        # mask 1_{x_i>0} across rows -> broadcast to (N,N)
        mask_row = pos_x[:, i].reshape(N, 1)
        # (a) departure: dx = -e_i, rate mu_i * dep_i
        if dep[i] > 0:
            rate_row = mu[i] * dep[i] * mask_row
            x_transitions.append((-_unit(K, i), rate_row))
        # (b) internal routing: dx = -e_i + e_j, rate mu_i * P_{ij}
        for j in range(K):
            pij = P[i, j]
            if pij > 0:
                dx = -_unit(K, i) + _unit(K, j)
                rate_row = mu[i] * pij * mask_row
                x_transitions.append((dx, rate_row))

    # Build the list of y-transitions: (dy, rate_per_col_matrix)
    y_transitions: list[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(K):
        # Arrivals at node r: dy = +e_r, rate lambda_r
        rate_col = lam[r] * ones_mat
        y_transitions.append((_unit(K, r), rate_col))

    for r in range(K):
        mask_col = pos_y[:, r].reshape(1, N)
        # (a) departure at r: dy = -e_r, rate mu_r * dep_r
        if dep[r] > 0:
            rate_col = mu[r] * dep[r] * mask_col
            y_transitions.append((-_unit(K, r), rate_col))
        # (b) routing r->s: dy = -e_r + e_s, rate mu_r * P_{rs}
        for s in range(K):
            prs = P[r, s]
            if prs > 0:
                dy = -_unit(K, r) + _unit(K, s)
                rate_col = mu[r] * prs * mask_col
                y_transitions.append((dy, rate_col))

    # Cache for k(D + shift) evaluations
    cache: Dict[Tuple[int, ...], np.ndarray] = {}
    def Kshift(shift_vec: np.ndarray) -> np.ndarray:
        key = tuple(int(v) for v in shift_vec.tolist())
        if key not in cache:
            cache[key] = k_of_shift(shift_vec)
        return cache[key]

    # Assemble U via the bilinear (A_x, A_y) expansion
    U = np.zeros((N, N), dtype=float)
    # Precompute K0 once
    K_base = K0

    # Also precompute k(D + dx) for each x-transition and k(D - dy) for each y-transition
    # (note: k(D - dy) = Kshift(-dy))
    x_terms = [(dx, rate_x, Kshift(dx)) for (dx, rate_x) in x_transitions]
    y_terms = [(dy, rate_y, Kshift(-dy)) for (dy, rate_y) in y_transitions]

    for dx, rate_x, K_dx in x_terms:
        for dy, rate_y, K_mdy in y_terms:
            K_dx_mdy = Kshift(dx - dy)          # k(D + dx - dy)
            # Bilinear four-point stencil:
            # k(D+dx-dy) - k(D+dx) - k(D-dy) + k(D)
            block = K_dx_mdy - K_dx - K_mdy + K_base
            # Rates are separable: rate_x depends on rows, rate_y on cols
            U += rate_x * rate_y * block

    # Numerical symmetrization
    U = 0.5 * (U + U.T)
    return U







@register_kernel("imq")
def build_U_imq(
    X: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    P: np.ndarray,
    *,
    c: float = 1.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Stein kernel matrix U for a Jackson network with the IMQ kernel:
        k(x,y) = (c^2 + ||x-y||_2^2)^(-alpha),  c>0, alpha>0.

    Notes
    -----
    - Heavy-tailed compared to exponential; often more sensitive to tail mismatch.
    - Implementation mirrors `build_U_exp`/`build_U_exp_hamming`:
        * same transition list (arrivals, departures, internal routing)
        * same four-point stencil: k(D+dx-dy) - k(D+dx) - k(D-dy) + k(D)
        * only k(D+shift) changes to the IMQ evaluation.
    """
    if c <= 0:
        raise ValueError("For 'imq' kernel, c must be > 0.")
    if alpha <= 0:
        raise ValueError("For 'imq' kernel, alpha must be > 0.")

    X = np.asarray(X, dtype=int)
    N, K = X.shape
    lam = np.asarray(lambdas, dtype=float).reshape(K)
    mu  = np.asarray(mus, dtype=float).reshape(K)
    P   = np.asarray(P, dtype=float).reshape(K, K)

    dep = 1.0 - P.sum(axis=1)
    dep = np.clip(dep, 0.0, 1.0)

    # Pairwise differences tensor D[r,c,:] = X[r,:] - X[c,:]
    D = _pairwise_diffs(X)  # (N, N, K)

    def k_of_shift(shift: np.ndarray) -> np.ndarray:
        """
        IMQ evaluation on D+shift:
          r2 = sum_i (D_i + shift_i)^2
          k  = (c^2 + r2)^(-alpha)
        """
        A = D + shift.reshape(1, 1, K)            # (N,N,K)
        r2 = (A.astype(float) ** 2).sum(axis=2)   # (N,N)
        return (c * c + r2) ** (-alpha)

    K0 = k_of_shift(np.zeros(K, dtype=int))

    pos_x = (X > 0).astype(float)
    pos_y = pos_x.copy()
    ones_mat = np.ones((N, N), dtype=float)

    # x-side transitions (operate on rows)
    x_transitions: list[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(K):
        rate_row = lam[i] * ones_mat
        x_transitions.append((_unit(K, i), rate_row))
    for i in range(K):
        mask_row = pos_x[:, i].reshape(N, 1)
        if dep[i] > 0:
            rate_row = mu[i] * dep[i] * mask_row
            x_transitions.append((-_unit(K, i), rate_row))
        for j in range(K):
            pij = P[i, j]
            if pij > 0:
                dx = -_unit(K, i) + _unit(K, j)
                rate_row = mu[i] * pij * mask_row
                x_transitions.append((dx, rate_row))

    # y-side transitions (operate on cols)
    y_transitions: list[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(K):
        rate_col = lam[r] * ones_mat
        y_transitions.append((_unit(K, r), rate_col))
    for r in range(K):
        mask_col = pos_y[:, r].reshape(1, N)
        if dep[r] > 0:
            rate_col = mu[r] * dep[r] * mask_col
            y_transitions.append((-_unit(K, r), rate_col))
        for s in range(K):
            prs = P[r, s]
            if prs > 0:
                dy = -_unit(K, r) + _unit(K, s)
                rate_col = mu[r] * prs * mask_col
                y_transitions.append((dy, rate_col))

    # cache for k(D + shift)
    cache: Dict[Tuple[int, ...], np.ndarray] = {}
    def Kshift(shift_vec: np.ndarray) -> np.ndarray:
        key = tuple(int(v) for v in shift_vec.tolist())
        if key not in cache:
            cache[key] = k_of_shift(shift_vec)
        return cache[key]

    # Assemble via four-point stencil
    U = np.zeros((N, N), dtype=float)
    K_base = K0
    x_terms = [(dx, rate_x, Kshift(dx)) for (dx, rate_x) in x_transitions]
    y_terms = [(dy, rate_y, Kshift(-dy)) for (dy, rate_y) in y_transitions]

    for dx, rate_x, K_dx in x_terms:
        for dy, rate_y, K_mdy in y_terms:
            K_dx_mdy = Kshift(dx - dy)                 # k(D + dx - dy)
            block = K_dx_mdy - K_dx - K_mdy + K_base   # stencil
            U += rate_x * rate_y * block

    U = 0.5 * (U + U.T)                                # numerical symmetrization
    return U






@register_kernel("exp_hamming")
def build_U_exp_hamming(
    X: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Stein kernel matrix U for Jackson networks with the exponentiated
    (normalized) Hamming kernel, as defined in the paper:

        k(x,y) = exp( - H(x,y) ),
      where H(x,y) = (1/d) * |{ i : x_i != y_i }|.

    This implementation has **no extra parameters**.
    """
    X = np.asarray(X, dtype=int)
    N, K = X.shape
    lam = np.asarray(lambdas, dtype=float).reshape(K)
    mu  = np.asarray(mus, dtype=float).reshape(K)
    P   = np.asarray(P, dtype=float).reshape(K, K)

    dep = 1.0 - P.sum(axis=1)
    dep = np.clip(dep, 0.0, 1.0)

    # Pairwise differences D[r,c,:] = X[r,:] - X[c,:]
    D = _pairwise_diffs(X)  # (N, N, K)

    def k_of_shift(shift: np.ndarray) -> np.ndarray:
        mism = (D + shift.reshape(1, 1, K)) != 0      # (N,N,K) boolean
        h = mism.sum(axis=2).astype(float) / float(K) # normalized Hamming
        return np.exp(-h)

    K0 = k_of_shift(np.zeros(K, dtype=int))

    pos_x = (X > 0).astype(float)
    pos_y = pos_x.copy()

    ones_mat = np.ones((N, N), dtype=float)

    # x-side transitions
    x_transitions: list[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(K):
        rate_row = lam[i] * ones_mat
        x_transitions.append((_unit(K, i), rate_row))
    for i in range(K):
        mask_row = pos_x[:, i].reshape(N, 1)
        if dep[i] > 0:
            rate_row = mu[i] * dep[i] * mask_row
            x_transitions.append((-_unit(K, i), rate_row))
        for j in range(K):
            pij = P[i, j]
            if pij > 0:
                dx = -_unit(K, i) + _unit(K, j)
                rate_row = mu[i] * pij * mask_row
                x_transitions.append((dx, rate_row))

    # y-side transitions
    y_transitions: list[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(K):
        rate_col = lam[r] * ones_mat
        y_transitions.append((_unit(K, r), rate_col))
    for r in range(K):
        mask_col = pos_y[:, r].reshape(1, N)
        if dep[r] > 0:
            rate_col = mu[r] * dep[r] * mask_col
            y_transitions.append((-_unit(K, r), rate_col))
        for s in range(K):
            prs = P[r, s]
            if prs > 0:
                dy = -_unit(K, r) + _unit(K, s)
                rate_col = mu[r] * prs * mask_col
                y_transitions.append((dy, rate_col))

    # cache and assemble
    cache: Dict[Tuple[int, ...], np.ndarray] = {}
    def Kshift(shift_vec: np.ndarray) -> np.ndarray:
        key = tuple(int(v) for v in shift_vec.tolist())
        if key not in cache:
            cache[key] = k_of_shift(shift_vec)
        return cache[key]

    U = np.zeros((N, N), dtype=float)
    K_base = K0
    x_terms = [(dx, rate_x, Kshift(dx)) for (dx, rate_x) in x_transitions]
    y_terms = [(dy, rate_y, Kshift(-dy)) for (dy, rate_y) in y_transitions]

    for dx, rate_x, K_dx in x_terms:
        for dy, rate_y, K_mdy in y_terms:
            K_dx_mdy = Kshift(dx - dy)
            block = K_dx_mdy - K_dx - K_mdy + K_base
            U += rate_x * rate_y * block

    U = 0.5 * (U + U.T)
    return U


# ---------------------------------------------------------------------
# 4) Public API: KSD for Jackson network with pluggable kernels
# ---------------------------------------------------------------------

def KSD_Jackson(
    a: Sequence[Sequence[int]],
    q: Sequence[float],
    lambdas: Sequence[float],
    mus: Sequence[float],
    P: Sequence[Sequence[float]],
    kernel: str | KernelBuilder = "exp",
    kernel_params: Optional[dict] = None,
    verbose: bool = False
) -> float:
    """
    Compute the kernel Stein discrepancy (KSD) for a Jackson network on Z_+^K.

    Inputs
    ------
    a : list of states (each a length-K int vector), shape (n,K) or nested list
    q : weights (nonnegative; will be normalized), shape (n,)
    lambdas : arrival rates lambda_i, shape (K,)
    mus : service rates mu_i, shape (K,)
    P : routing matrix P_{ij}, shape (K,K)
    kernel : str or builder
        Registered name (e.g., "exp") or callable:
        U = builder(X, lambdas, mus, P, **params)
    kernel_params : dict
        Extra params for the kernel (e.g., {"beta": 0.97})
    verbose : bool

    Returns
    -------
    float
        KSD (not squared): sqrt( w^T U w ).
    """
    X = _as_2d_int_states(a)               # (n,K), ints >= 0
    q = np.asarray(q, dtype=float)
    if np.any(q < 0):
        raise ValueError("Weights must be nonnegative.")
    if q.sum() <= 0:
        raise ValueError("Sum of weights must be positive.")

    # Merge duplicates and normalize weights
    Xu, w = _merge_duplicate_states(X, q)
    N, K = Xu.shape
    if verbose:
        print(f"[KSD-Jackson] unique states N={N}, dim K={K}")

    # Resolve kernel builder
    params = {} if kernel_params is None else dict(kernel_params)
    if isinstance(kernel, str):
        if kernel not in REGISTRY:
            raise ValueError(f"Unknown kernel '{kernel}'. Available: {list(REGISTRY)}")
        builder: KernelBuilder = REGISTRY[kernel]
    else:
        builder = kernel

    # Build Stein kernel matrix U
    U = builder(Xu, np.asarray(lambdas), np.asarray(mus), np.asarray(P), **params)
    U = 0.5 * (U + U.T)

    # KSD^2 = w^T U w
    ksd2 = float(w @ (U @ w))
    if ksd2 < 0 and ksd2 > -1e-14:
        ksd2 = 0.0
    ksd = float(np.sqrt(max(0.0, ksd2)))

    if verbose:
        print(f"[KSD-Jackson] KSD^2 = {ksd2:.6e}, KSD = {ksd:.6e}")
    return ksd
