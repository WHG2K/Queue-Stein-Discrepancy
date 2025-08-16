import numpy as np
from collections import Counter
import gurobipy as gp
from gurobipy import GRB
import math


def GSD_MM1_V1(a: list, q: list, lmd: float, mu: float, verbose: bool = False):
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






def GSD_MM1_V2(a: list, q: list, lmd: float, mu: float, verbose: bool = False):
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
    
    ### This is what makes the version 2 different from version 1
    stability_bound_C = 1.0 / mu

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







# def wasserstein_MM1(samples, rho, tol=1e-8):
#     """
#     Compute the 1-Wasserstein distance between empirical samples of an M/M/1 queue
#     and its theoretical stationary distribution with load rho.

#     Parameters
#     ----------
#     samples : Sequence[int]
#         Observed queue-length samples.
#     rho : float
#         Traffic intensity (lambda/mu), must satisfy 0 <= rho < 1.
#     tol : float
#         Tolerance for tail probability: truncate theoretical PMF when
#         cumulative probability >= 1 - tol AND support covers all samples.

#     Returns
#     -------
#     float
#         The 1-Wasserstein distance W between the empirical and theoretical distributions.
#     """
#     # determine how far to extend the theoretical PMF
#     max_sample = max(samples) if samples else 0

#     # build theoretical pmf until both conditions are met:
#     # 1) we've covered all observed samples (k > max_sample)
#     # 2) tail probability remaining <= tol
#     pmf = []
#     cum_prob = 0.0
#     k = 0
#     while True:
#         p = (1 - rho) * rho**k
#         pmf.append(p)
#         cum_prob += p
#         k += 1
#         # stop once we've passed the max observed value and accumulated >= 1 - tol
#         if k > max_sample and cum_prob >= 1 - tol:
#             # absorb any tiny remaining mass into the last bin
#             pmf[-1] += 1 - cum_prob
#             break

#     values_theo = np.arange(len(pmf))
#     weights_theo = np.array(pmf)

#     # build empirical distribution from samples
#     counts = Counter(samples)
#     total = len(samples)
#     # unify support between empirical and theoretical
#     support_all = np.union1d(values_theo, np.array(list(counts.keys())))
#     weights_emp = np.array([counts.get(s, 0) / total for s in support_all])

#     # compute 1-Wasserstein distance using SciPy
#     W = wasserstein_distance(
#         u_values=support_all,
#         v_values=support_all,
#         u_weights=[weights_theo[s] if s < len(weights_theo) else 0 for s in support_all],
#         v_weights=weights_emp
#     )

#     return W




# def SD_MM1_easy(q, lmd=0.07, mu=0.10, verbose=False):
#     """
#     Compute the graphic stein discrepancy of the samples Q and the steady state M/M/1, by solving the LP:
#       max_{x[0..m], y[0..m-1]} q0*lmd*x0 + sum_{i=1}^m qi*(lmd*xi - mu*x[i-1])
#       s.t. |lmd*y_i - mu*y_{i-1}| <= 1   (i=1..m-1)
#            |lmd*y_0 - mu*x_0| <= 1
#            y_i = x_{i+1} - x_i           (i=0..m-1)
#     This is an easy version when we assume that Q is supported on 0,1,...,m with weights q0,q1,...,qm

#     Parameters
#     ----------
#     q : list of float
#         Probability weights [q0, q1, ..., qm], nonnegative and sum to 1.
#     lmd : float
#         Arrival rate.
#     mu : float
#         Service rate.
#     verbose : bool
#         Whether to print solver output.

#     Returns
#     -------
#     obj_val : float
#         Optimal objective value.
#     x_vals : list of float
#         Values of x[0..m].
#     y_vals : list of float
#         Values of y[0..m-1].
#     """

#     m = len(q) - 1
#     # build model
#     model = gp.Model("mm1_dual")
#     model.Params.OutputFlag = 0

#     # variables x[0..m], y[0..m-1] (unbounded)
#     x = model.addVars(m+1, lb=-GRB.INFINITY, name="x")
#     y = model.addVars(m,   lb=-GRB.INFINITY, name="y")

#     # objective: q0*lmd*x0 + sum_{i=1}^m qi*(lmd*xi - mu*x[i-1])
#     expr = q[0] * lmd * x[0]
#     for i in range(1, m+1):
#         expr += q[i] * (lmd * x[i] - mu * x[i-1])
#     model.setObjective(expr, GRB.MAXIMIZE)

#     # |lmd*y_i - mu*y_{i-1}| <= 1 for i=1..m-1
#     for i in range(1, m):
#         model.addConstr( lmd*y[i] - mu*y[i-1] <= 1 )
#         model.addConstr(-lmd*y[i] + mu*y[i-1] <= 1 )

#     # |lmd*y_0 - mu*x_0| <= 1
#     model.addConstr( lmd*y[0] - mu*x[0] <= 1 )
#     model.addConstr(-lmd*y[0] + mu*x[0] <= 1 )

#     # y_i = x_{i+1} - x_i for i=0..m-1
#     for i in range(m):
#         model.addConstr(y[i] == x[i+1] - x[i])

#     # anchor to eliminate translation invariance
#     model.addConstr(x[0] == 0)

#     # solve
#     model.optimize()

#     # map Gurobi status code to string
#     status_code = model.Status
#     status_map = {
#         GRB.OPTIMAL:          "OPTIMAL",
#         GRB.UNBOUNDED:        "UNBOUNDED",
#         GRB.INFEASIBLE:       "INFEASIBLE",
#         GRB.INF_OR_UNBD:      "INF_OR_UNBOUNDED",
#         GRB.TIME_LIMIT:       "TIME_LIMIT",
#         GRB.SUBOPTIMAL:       "SUBOPTIMAL",
#         GRB.INTERRUPTED:      "INTERRUPTED"
#     }
#     status = status_map.get(status_code, f"STATUS_{status_code}")

#     obj_val = model.ObjVal if status_code == GRB.OPTIMAL else None
#     return obj_val, status







