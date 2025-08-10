import numpy as np
from scipy.stats import wasserstein_distance
import math
import gurobipy as gp
from gurobipy import GRB

def W1_MM1(q: list, a: list, rho: float, padding: int = 100, tolerance: float = 1e-9):
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





def SD_MM1_V1(q: list, a: list, lmd: float, mu: float, verbose: bool = False):
    """
    Computes the W1 distance by solving a robust, sparse Linear Program.

    This version explicitly defines variables for both the slope (x_k) and
    the curvature (y_k) for maximum clarity and theoretical soundness.
    It handles unsorted input and ensures 0 and 1 are in the support.

    Parameters
    ----------
    q : list or np.ndarray
        Probability weights corresponding to 'a'.
    a : list or np.ndarray
        Support points. Does not need to be sorted.
    lmd : float
        Arrival rate for the M/M/1 queue.
    mu : float
        Service rate for the M/M/1 queue.

    Returns
    -------
    obj_val : float or None
        The optimal objective value (the W1 distance). None if not optimal.
    status : str
        The human-readable Gurobi solver status.
    """
    # --- 1. Input Validation & Data Pre-processing ---
    if len(q) != len(a):
        raise ValueError("Length of weights q and support a must be equal.")
    if mu <= lmd:
        raise ValueError("Service rate mu must be greater than arrival rate lmd.")
    rho = lmd / mu

    # --- 2. Sort the points ---
    point_map = {k: v for k, v in zip(a, q)}
    point_map.setdefault(0, 0.0)
    point_map.setdefault(1, 0.0)

    sorted_points = sorted(point_map.items())
    a = [point[0] for point in sorted_points]
    q = [point[1] for point in sorted_points]
    
    stability_bound_C = 1.0 / (mu - lmd)

    # --- 2. Identify all necessary "Critical Points" ---
    critical_points_base = set()
    for point in a:
        critical_points_base.add(point)
        if point > 0:
            critical_points_base.add(point - 1)
    
    if not critical_points_base:
        critical_points_base.add(0)

    critical_points_x = critical_points_base

    K_x = sorted(list(critical_points_x))
    p_x = len(K_x)
    x_point_to_idx = {k: j for j, k in enumerate(K_x)}
    if verbose:
        print([f"x[{i}]: {K_x[i]}" for i in range(p_x)])

    K_y = [k for k in K_x if k + 1 in x_point_to_idx] # equivalent to K_y = [k for k in K_x if k + 1 in x_point_to_idx], but faster
    p_y = len(K_y)
    y_point_to_idx = {k: j for j, k in enumerate(K_y)}
    if verbose:
        print([f"y[{i}]: {K_y[i]}" for i in range(p_y)])
    
    # --- 3. Build the Gurobi LP Model ---
    model = gp.Model("SD_MM1_V1")
    model.Params.OutputFlag = 0
    
    # --- 4. Define EXPLICIT Variables for x_k and y_k ---
    x = model.addVars(p_x, lb=-GRB.INFINITY, name="x")
    y = model.addVars(p_y, lb=-GRB.INFINITY, name="y")

    # --- 5. Set the Objective Function ---
    obj_expr = 0
    for i in range(len(a)):
        a_i, q_i = a[i], q[i]
        if q_i == 0: continue
        
        idx_ai = x_point_to_idx[a_i]
        obj_expr += q_i * lmd * x[idx_ai]
        
        if a_i > 0:
            idx_ai_minus_1 = x_point_to_idx[a_i - 1]
            obj_expr += q_i * (-mu) * x[idx_ai_minus_1]
            
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # --- 6. Set ALL Constraints Explicitly ---

    # 6a. Consistency Constraint: y_k = x_{k+1} - x_k
    for k_y_val, j_y in y_point_to_idx.items():
        j_x_curr = x_point_to_idx[k_y_val]
        j_x_next = x_point_to_idx[k_y_val + 1]
        model.addConstr(y[j_y] == x[j_x_next] - x[j_x_curr])
        if verbose:
            print(f"add constraint: y[{j_y}] = x[{j_x_next}] - x[{j_x_curr}]")

    # 6b. Stability Constraint: |\Delta^2 | <= C
    for j in range(p_x - 1):
        gap_distance = K_x[j+1] - K_x[j]
        bound = gap_distance * stability_bound_C
        model.addConstr(x[j+1] - x[j] <= bound)
        model.addConstr(x[j+1] - x[j] >= -bound)    
        if verbose:
            print(f"add constraint: |x[{j+1}] - x[{j}]| <= {gap_distance} * 1/(mu - lmd)")

    # 6c. Lipschitz-on-Generator Constraint: |lambda*y_k - mu*y_{k-1}| <= 1
    idx_y0 = y_point_to_idx[0]
    idx_x0 = x_point_to_idx[0]
    expr_init = rho * y[idx_y0] - x[idx_x0]
    model.addConstr(expr_init <= 1/mu)
    model.addConstr(expr_init >= -1/mu)
    if verbose:
        print(f"add constraint: |rho*y[{idx_y0}] - x[{idx_x0}]| <= 1/mu")
    
    for j in range(p_y - 1):
        gap_distance = K_y[j+1] - K_y[j]
        expr = rho ** gap_distance * y[j+1] - y[j]
        model.addConstr(expr <= 1/mu * (1 - rho ** gap_distance) / (1 - rho))
        model.addConstr(expr >= -1/mu * (1 - rho ** gap_distance) / (1 - rho))
        if verbose:
            if gap_distance == 1:
                print(f"add constraint: |rho*y[{j+1}] - y[{j}]| <= 1/mu")
            else:
                print(f"add constraint: |rho^{gap_distance}*y[{j+1}] - y[{j}]| <= 1/mu * (1 - rho^{gap_distance}) / (1 - rho)")

    # --- 7. Solve and Return ---
    model.optimize()
    
    # FIX: Map Gurobi status code to a human-readable string
    status_code = model.Status
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    status = status_map.get(status_code, f"UNKNOWN_STATUS_{status_code}")
    
    obj_val = -1
    if status_code == GRB.OPTIMAL:
        obj_val = model.ObjVal
        
    return obj_val, status



def generate_gapped_distribution(rho, gap_size, trunc_point):
    """
    Generates a gapped distribution based on the M/M/1 steady state.

    For a k-gap, the probability at point j*k is the sum of the original
    probabilities from j*k to (j+1)*k - 1. This version ensures the
    returned distribution is always normalized and returns standard lists.
    """
    if gap_size < 1:
        raise ValueError("Gap size must be at least 1.")

    # 1. Generate the base M/M/1 steady state PMF (this sum is < 1)
    support_np = np.arange(trunc_point + 1)
    base_pmf_np = (1 - rho) * (rho ** support_np)

    if gap_size == 1:
        # FIX 1: Normalize the base_pmf before returning it for consistency.
        base_pmf_np /= base_pmf_np.sum()
        # FIX 2: Return as standard Python lists.
        return support_np.tolist(), base_pmf_np.tolist()

    # 2. Create the new gapped distribution
    a_gapped = []
    q_gapped = []
    
    for j in range(math.ceil(trunc_point / gap_size)):
        start_idx = j * gap_size
        end_idx = min((j + 1) * gap_size, trunc_point + 1)
        
        if start_idx > trunc_point:
            break
            
        a_gapped.append(start_idx)
        
        prob_sum = np.sum(base_pmf_np[start_idx:end_idx])
        q_gapped.append(prob_sum)
        
    q_gapped_np = np.array(q_gapped)
    # This normalization step is correctly applied for gap_size > 1
    q_gapped_np /= q_gapped_np.sum()
    
    # FIX 2: Return as standard Python lists.
    return a_gapped, q_gapped_np.tolist()





if __name__ == "__main__":

    m = gp.Model("test")
    del m

    # --- Shared Parameters ---
    lmd_val = 0.9
    mu_val = 1.0
    rho_val = lmd_val / mu_val

    print(f"Testing with Parameters: lambda={lmd_val}, mu={mu_val} (rho={rho_val:.2f})")
    
    # --- Truncation Point ---
    # As requested, truncate at roughly 2 / (1-rho)
    truncation_point = math.ceil(2 / (1 - rho_val))
    print(f"Base distribution truncated at k = {truncation_point}")
    
    # --- Gap Sizes to Test ---
    gap_sizes_to_test = [1, 2, 5, 10, 20]

    for gap in gap_sizes_to_test:
        print("\n" + "="*60)
        print(f"--- Testing with Gap Size = {gap} ---")

        # 1. Generate the gapped distribution
        a_gapped, q_gapped = generate_gapped_distribution(rho_val, gap, truncation_point)
        # print(a_gapped)
        print("sum of q_gapped:", sum(q_gapped))
        print(f"Generated distribution has {len(a_gapped)} support points.")
        # print(f"Support a = {a_gapped[:10]}...") # Uncomment to see the support

        # 2. Calculate using our sparse LP
        lp_val, status = SD_MM1_V1(q_gapped, a_gapped, lmd_val, mu_val, verbose=False)
        
        # 3. Calculate using standard library
        w1_val = W1_MM1(q_gapped, a_gapped, rho_val, padding=100)
        
        # 4. Print and compare results
        print(f"  SD_MM1_V1 (LP Solver) Result: {lp_val:.8f} (Status: {status})")
        print(f"  W1_MM1 (Scipy) Result:      {w1_val:.8f}")

        
        # if lp_val is not None:
        #     diff = abs(lp_val - w1_val)
        #     print(f"  Absolute Difference:          {diff:.8f}")