# ksd_mm1.py
# All-in-one: kernel registry, kernel builders, and the generic KSD entry for M/M/1.

from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np

# ---------------------------------------------------------------------
# 1) Registry: string name -> kernel builder function
# ---------------------------------------------------------------------

# A kernel builder constructs the Stein kernel matrix U (K x K) given:
#   - n: sorted unique supports on Z_+ (int np.ndarray)
#   - lmd, mu: M/M/1 rates
#   - **params: kernel-specific parameters
KernelBuilder = Callable[..., np.ndarray]

REGISTRY: Dict[str, KernelBuilder] = {}


def register_kernel(name: str):
    """
    Decorator to register a kernel builder under a string name.
    Example:
        @register_kernel("exp")
        def build_U_exp(n, lmd, mu, *, beta=0.95): ...
    """
    def deco(func: KernelBuilder):
        if name in REGISTRY:
            raise ValueError(f"Kernel '{name}' already registered.")
        REGISTRY[name] = func
        return func
    return deco


# ---------------------------------------------------------------------
# 2) Concrete kernel: Exponential (AR(1)) kernel
# ---------------------------------------------------------------------

@register_kernel("exp")
def build_U_exp(n: np.ndarray, lmd: float, mu: float, *, beta: float = 0.95) -> np.ndarray:
    """
    Exponential (AR(1)) kernel: k(n,m) = beta^{|n-m|}, with beta in (0,1).
    Returns the Stein kernel matrix U whose (i,j) entry is:
        U_ij = (A_{n_i} A_{n_j} k)(n_i, n_j)

    Stencils (with s = n - m):
        Tpp = 2 β^{|s|} - β^{|s+1|} - β^{|s-1|}
        Tpm = β^{|s+2|} - 2 β^{|s+1|} + β^{|s|}
        Tmp = β^{|s-2|} - 2 β^{|s-1|} + β^{|s|}
    Then:
        U = λ^2 Tpp
            + λμ 1_{m>0} Tpm
            + μλ 1_{n>0} Tmp
            + μ^2 1_{n>0}1_{m>0} Tpp
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("For 'exp' kernel, beta must be in (0,1).")

    # Pairwise state differences
    S = n[:, None] - n[None, :]              # shape (K, K)
    absS = np.abs(S)

    # Powers of beta for needed offsets
    beta_absS   = beta ** (absS)
    beta_absSp1 = beta ** (np.abs(S + 1))
    beta_absSm1 = beta ** (np.abs(S - 1))
    beta_absSp2 = beta ** (np.abs(S + 2))
    beta_absSm2 = beta ** (np.abs(S - 2))

    # Boundary indicators
    npos = (n > 0).astype(float)[:, None]    # 1_{n>0} for rows
    mpos = (n > 0).astype(float)[None, :]    # 1_{m>0} for cols
    both_pos = npos * mpos

    # Four-point stencils
    Tpp = 2.0 * beta_absS - beta_absSp1 - beta_absSm1
    Tpm = beta_absSp2 - 2.0 * beta_absSp1 + beta_absS
    Tmp = beta_absSm2 - 2.0 * beta_absSm1 + beta_absS

    lam, mmu = float(lmd), float(mu)

    U = (lam * lam) * Tpp \
        + (lam * mmu) * (mpos * Tpm) \
        + (mmu * lam) * (npos * Tmp) \
        + (mmu * mmu) * (both_pos * Tpp)

    # Numerical guard (should already be symmetric up to fp error)
    U = 0.5 * (U + U.T)
    return U



@register_kernel("imq")
def build_U_imq(n: np.ndarray, lmd: float, mu: float, *,
                c: float = 1.0, alpha: float = 0.5) -> np.ndarray:
    """
    IMQ kernel builder (1D): k(n,m) = (c^2 + (n-m)^2)^(-alpha), with c>0, alpha>0.

    Notes
    -----
    Heavy-tailed compared to exponential; improves sensitivity in heavy-traffic tails.
    Uses the same four-point stencil pattern:
        Tpp(s) = 2k(|s|) - k(|s+1|) - k(|s-1|)
        Tpm(s) = k(|s+2|) - 2k(|s+1|) + k(|s|)
        Tmp(s) = k(|s-2|) - 2k(|s-1|) + k(|s|)
    and
        U = λ^2 Tpp + λμ 1_{m>0} Tpm + μλ 1_{n>0} Tmp + μ^2 1_{n>0}1_{m>0} Tpp
    """
    if c <= 0:
        raise ValueError("For 'imq' kernel, c must be > 0.")
    if alpha <= 0:
        raise ValueError("For 'imq' kernel, alpha must be > 0.")

    S = n[:, None] - n[None, :]              # shape (K, K)
    r0 = np.abs(S)
    r_p1 = np.abs(S + 1)
    r_m1 = np.abs(S - 1)
    r_p2 = np.abs(S + 2)
    r_m2 = np.abs(S - 2)

    # k(r) = (c^2 + r^2)^(-alpha)
    def k_of(r: np.ndarray) -> np.ndarray:
        return (c * c + r.astype(float) * r.astype(float)) ** (-alpha)

    k0   = k_of(r0)
    k_p1 = k_of(r_p1)
    k_m1 = k_of(r_m1)
    k_p2 = k_of(r_p2)
    k_m2 = k_of(r_m2)

    npos = (n > 0).astype(float)[:, None]
    mpos = (n > 0).astype(float)[None, :]
    both_pos = npos * mpos

    Tpp = 2.0 * k0 - k_p1 - k_m1
    Tpm = k_p2 - 2.0 * k_p1 + k0
    Tmp = k_m2 - 2.0 * k_m1 + k0

    lam, mmu = float(lmd), float(mu)
    U = (lam * lam) * Tpp \
        + (lam * mmu) * (mpos * Tpm) \
        + (mmu * lam) * (npos * Tmp) \
        + (mmu * mmu) * (both_pos * Tpp)

    U = 0.5 * (U + U.T)                      # numerical symmetrization
    return U






# ---------------------------------------------------------------------
# ******) TEMPLATE: add new kernels here by copying and editing this block
# ---------------------------------------------------------------------
# @register_kernel("ss1")
# def build_U_ss1(n: np.ndarray, lmd: float, mu: float, *, alpha: float = 0.97) -> np.ndarray:
#     """
#     Example: Stable Spline (SS-1 / TC) kernel builder.
#     Steps:
#       1) Define k(n,m) or directly compute the needed shifted kernel values.
#       2) Form stencils Tpp, Tpm, Tmp as in `build_U_exp`.
#       3) Combine with λ, μ and boundary indicators to get U.
#       4) Symmetrize U and return.
#     """
#     # ... implement here ...
#     raise NotImplementedError("SS-1 kernel not implemented yet.")


# ---------------------------------------------------------------------
# 3) Generic entry: KSD for M/M/1 with pluggable kernel
# ---------------------------------------------------------------------

def KSD_MM1(a: list, q: list, lmd: float, mu: float,
            kernel: str | Callable[..., np.ndarray] = "exp",
            kernel_params: Optional[dict] = None,
            verbose: bool = False) -> float:
    """
    Compute the kernel Stein discrepancy (KSD) for an M/M/1 queue on Z_+.

    Parameters
    ----------
    a : list[int]
        Support points (nonnegative integers). Duplicates allowed.
    q : list[float]
        Nonnegative weights corresponding to 'a'. Will be normalized to sum to 1.
    lmd : float
        Arrival rate λ. Require mu > lmd (stability).
    mu : float
        Service rate μ.
    kernel : str or callable
        Either a registered kernel name (e.g., "exp") or a builder callable with
        signature: U = kernel(n: np.ndarray, lmd: float, mu: float, **params).
    kernel_params : dict, optional
        Extra parameters forwarded to the kernel builder (e.g., {"beta": 0.97}).
    verbose : bool

    Returns
    -------
    float
        The KSD value (non-squared): sqrt( w^T U w ).
    """
    # ---- basic checks ----
    if len(a) != len(q):
        raise ValueError("Length of 'a' and 'q' must match.")
    if mu <= lmd:
        raise ValueError("Require stability: mu > lambda.")
    params = {} if kernel_params is None else dict(kernel_params)

    # ---- validate supports: Z_+ integers ----
    a = np.asarray(a, dtype=float)
    if np.any(a < 0) or np.any(np.abs(a - np.round(a)) > 1e-12):
        raise ValueError("All support points 'a' must be nonnegative integers.")
    a = a.astype(int)

    # ---- validate weights ----
    q = np.asarray(q, dtype=float)
    if np.any(q < 0):
        raise ValueError("Weights 'q' must be nonnegative.")
    if q.sum() <= 0:
        raise ValueError("Sum of 'q' must be positive.")

    # ---- merge duplicates & normalize ----
    n, inv = np.unique(a, return_inverse=True)  # unique supports (sorted)
    w = np.zeros_like(n, dtype=float)
    np.add.at(w, inv, q)
    w = w / w.sum()

    if verbose:
        print(f"[KSD] Unique support size K={len(n)}, min={n.min()}, max={n.max()}.")

    # ---- resolve builder ----
    if isinstance(kernel, str):
        if kernel not in REGISTRY:
            raise ValueError(f"Unknown kernel '{kernel}'. Available: {list(REGISTRY)}")
        builder: KernelBuilder = REGISTRY[kernel]
    else:
        builder = kernel  # user-supplied callable

    # ---- build Stein kernel matrix U ----
    U = builder(n, lmd, mu, **params)

    # ---- KSD^2 = w^T U w ; return sqrt ----
    U = 0.5 * (U + U.T)                 # symmetrize (safety)
    ksd2 = float(w @ (U @ w))
    if ksd2 < 0 and ksd2 > -1e-14:      # guard tiny negatives due to fp error
        ksd2 = 0.0
    ksd = float(np.sqrt(max(0.0, ksd2)))

    if verbose:
        print(f"[KSD] KSD^2 = {ksd2:.6e}, KSD = {ksd:.6e}")
    return ksd


# # ---------------------------------------------------------------------
# # 4) Convenience wrapper for the exponential kernel
# # ---------------------------------------------------------------------

# def KSD_MM1_exp(a: list, q: list, lmd: float, mu: float,
#                 beta: float = 0.95,
#                 verbose: bool = False) -> float:
#     """
#     Convenience wrapper for the exponential kernel.
#     Equivalent to: KSD_MM1(..., kernel='exp', kernel_params={'beta': beta})
#     """
#     return KSD_MM1(a, q, lmd, mu, kernel="exp",
#                    kernel_params={"beta": beta},
#                    verbose=verbose)
