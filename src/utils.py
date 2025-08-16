import numpy as np
from typing import List, Tuple
from collections import Counter
from scipy.stats import wasserstein_distance
import math


def W1_MM1(a: list, q: list, rho: float, padding: int = 200, tolerance: float = 1e-9):
    """
    Computes the 1-Wasserstein distance between an empirical distribution Q
    and the theoretical M/M/1 steady-state distribution X, using rho directly.

    The theoretical distribution X is truncated at a point determined by both
    a padding value beyond the empirical support and a tolerance for the
    tail mass.

    Parameters
    ----------
    q : list or np.ndarray
        Probability weights of the empirical distribution Q.
    a : list or np.ndarray
        Support points of Q, corresponding to q.
    rho : float
        The traffic intensity (lambda / mu) of the M/M/1 queue. Must be in [0, 1).
    padding : int, optional
        The support for the calculation will extend at least this many points
        beyond the maximum support point of Q. Default is 10.
    tolerance : float, optional
        The tail mass of the truncated theoretical distribution will be less
        than this value. Default is 1e-9.

    Returns
    -------
    float
        The computed 1-Wasserstein distance.
    """
    # --- 1. Input Validation and Parameter Setup ---
    if not (0 <= rho < 1):
        raise ValueError("Traffic intensity rho must be in the range [0, 1).")
    
    # --- 2. Determine the Truncation Point for the Distributions ---
    # The final support will be [0, 1, ..., max_k]
    
    # Condition 1: Truncate based on padding
    max_a = max(a) if len(a) > 0 else 0
    trunc_point_padding = max_a + padding
    
    # Condition 2: Truncate based on tail mass tolerance
    # We need to find N such that rho**(N+1) < tolerance
    if rho > 0:
        # (N+1) * log(rho) < log(tolerance) => N+1 > log(tolerance) / log(rho) (log(rho) is negative)
        trunc_point_tol = math.ceil(math.log(tolerance) / math.log(rho)) - 1
    else: # if rho is 0, the distribution is just P(0)=1
        trunc_point_tol = 0

    # The final truncation point is the larger of the two
    max_k = max(trunc_point_padding, trunc_point_tol)
    
    # The common support for both distributions
    support = np.arange(max_k + 1)
    
    # --- 3. Construct the Full PMF for the Empirical Distribution Q ---
    emp_weights = np.zeros(max_k + 1)
    for i in range(len(a)):
        if a[i] <= max_k:
            emp_weights[a[i]] = q[i]
            
    # Normalize just in case the original q was not normalized
    if emp_weights.sum() > 0:
        emp_weights /= emp_weights.sum()

    # --- 4. Construct the Full PMF for the Theoretical Distribution X ---
    # Calculate the geometric distribution probabilities
    th_weights = (1 - rho) * (rho ** support)
    
    # The tail mass is everything from max_k + 1 onwards.
    # For a geometric distribution, this sum is exactly rho**(max_k + 1)
    tail_mass = rho**(max_k + 1)
    
    # Add the tail mass to the last point to ensure the distribution sums to 1
    th_weights[max_k] += tail_mass
    
    # --- 5. Compute and Return the Wasserstein Distance ---
    return wasserstein_distance(support, support, u_weights=emp_weights, v_weights=th_weights)


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





def list_to_distr(xs: List[int], atol: float = 1e-12) -> Tuple[List[int], List[float]]:
    """
    Convert a list of nonnegative integers into a discrete distribution.
    Returns:
      a: sorted unique values (support)
      q: probabilities aligned with a
    """
    if not xs:
        raise ValueError("Input list is empty.")
    if any((x < 0) or (int(x) != x) for x in xs):
        raise ValueError("All elements must be nonnegative integers.")

    cnt = Counter(xs)
    a = sorted(cnt.keys())
    n = len(xs)
    q = [cnt[v] / n for v in a]

    s = sum(q)
    assert abs(s - 1.0) <= atol, f"Probabilities sum to {s}, not 1 (tolerance {atol})."
    return a, q
