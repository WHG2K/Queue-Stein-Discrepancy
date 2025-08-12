import numpy as np
from typing import List, Tuple
from collections import Counter

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
