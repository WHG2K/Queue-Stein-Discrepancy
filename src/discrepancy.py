import numpy as np
from collections import Counter
from scipy.stats import wasserstein_distance
import gurobipy as gp
from gurobipy import GRB



def wasserstein_MM1(samples, rho, tol=1e-8):
    """
    Compute the 1-Wasserstein distance between empirical samples of an M/M/1 queue
    and its theoretical stationary distribution with load rho.

    Parameters
    ----------
    samples : Sequence[int]
        Observed queue-length samples.
    rho : float
        Traffic intensity (lambda/mu), must satisfy 0 <= rho < 1.
    tol : float
        Tolerance for tail probability: truncate theoretical PMF when
        cumulative probability >= 1 - tol AND support covers all samples.

    Returns
    -------
    float
        The 1-Wasserstein distance W between the empirical and theoretical distributions.
    """
    # determine how far to extend the theoretical PMF
    max_sample = max(samples) if samples else 0

    # build theoretical pmf until both conditions are met:
    # 1) we've covered all observed samples (k > max_sample)
    # 2) tail probability remaining <= tol
    pmf = []
    cum_prob = 0.0
    k = 0
    while True:
        p = (1 - rho) * rho**k
        pmf.append(p)
        cum_prob += p
        k += 1
        # stop once we've passed the max observed value and accumulated >= 1 - tol
        if k > max_sample and cum_prob >= 1 - tol:
            # absorb any tiny remaining mass into the last bin
            pmf[-1] += 1 - cum_prob
            break

    values_theo = np.arange(len(pmf))
    weights_theo = np.array(pmf)

    # build empirical distribution from samples
    counts = Counter(samples)
    total = len(samples)
    # unify support between empirical and theoretical
    support_all = np.union1d(values_theo, np.array(list(counts.keys())))
    weights_emp = np.array([counts.get(s, 0) / total for s in support_all])

    # compute 1-Wasserstein distance using SciPy
    W = wasserstein_distance(
        u_values=support_all,
        v_values=support_all,
        u_weights=[weights_theo[s] if s < len(weights_theo) else 0 for s in support_all],
        v_weights=weights_emp
    )

    return W




def SD_MM1_easy(q, lmd=0.07, mu=0.10, verbose=False):
    """
    Compute the graphic stein discrepancy of the samples Q and the steady state M/M/1, by solving the LP:
      max_{x[0..m], y[0..m-1]} q0*lmd*x0 + sum_{i=1}^m qi*(lmd*xi - mu*x[i-1])
      s.t. |lmd*y_i - mu*y_{i-1}| <= 1   (i=1..m-1)
           |lmd*y_0 - mu*x_0| <= 1
           y_i = x_{i+1} - x_i           (i=0..m-1)
    This is an easy version when we assume that Q is supported on 0,1,...,m with weights q0,q1,...,qm

    Parameters
    ----------
    q : list of float
        Probability weights [q0, q1, ..., qm], nonnegative and sum to 1.
    lmd : float
        Arrival rate.
    mu : float
        Service rate.
    verbose : bool
        Whether to print solver output.

    Returns
    -------
    obj_val : float
        Optimal objective value.
    x_vals : list of float
        Values of x[0..m].
    y_vals : list of float
        Values of y[0..m-1].
    """
    # m = len(q) - 1

    # # create model
    # model = gp.Model("mm1_lp")
    # model.Params.OutputFlag = 1 if verbose else 0

    # # add variables: x[0..m], y[0..m-1], both unbounded
    # x = model.addVars(m+1, lb=-GRB.INFINITY, name="x")
    # y = model.addVars(m,   lb=-GRB.INFINITY, name="y")

    # # objective: q0*lmd*x0 + sum_{i=1}^m qi*(lmd*xi - mu*x[i-1])
    # expr = q[0] * lmd * x[0]
    # for i in range(1, m+1):
    #     expr += q[i] * (lmd * x[i] - mu * x[i-1])
    # model.setObjective(expr, GRB.MAXIMIZE)

    # # absolute-value constraints for i=1..m-1
    # for i in range(1, m):
    #     model.addConstr( lmd*y[i] - mu*y[i-1] <= 1, name=f"abs1_{i}")
    #     model.addConstr(-lmd*y[i] + mu*y[i-1] <= 1, name=f"abs2_{i}")

    # # absolute-value constraint for i=0
    # model.addConstr( lmd*y[0] - mu*x[0] <= 1, name="abs1_0")
    # model.addConstr(-lmd*y[0] + mu*x[0] <= 1, name="abs2_0")

    # # linking constraints y_i = x_{i+1} - x_i
    # for i in range(m):
    #     model.addConstr(y[i] == x[i+1] - x[i], name=f"def_y_{i}")

    # # solve
    # model.optimize()

    # # collect results
    # obj_val = model.ObjVal if model.Status == GRB.OPTIMAL else None

    # return obj_val


    m = len(q) - 1
    # build model
    model = gp.Model("mm1_dual")
    model.Params.OutputFlag = 0

    # variables x[0..m], y[0..m-1] (unbounded)
    x = model.addVars(m+1, lb=-GRB.INFINITY, name="x")
    y = model.addVars(m,   lb=-GRB.INFINITY, name="y")

    # objective: q0*lmd*x0 + sum_{i=1}^m qi*(lmd*xi - mu*x[i-1])
    expr = q[0] * lmd * x[0]
    for i in range(1, m+1):
        expr += q[i] * (lmd * x[i] - mu * x[i-1])
    model.setObjective(expr, GRB.MAXIMIZE)

    # |lmd*y_i - mu*y_{i-1}| <= 1 for i=1..m-1
    for i in range(1, m):
        model.addConstr( lmd*y[i] - mu*y[i-1] <= 1 )
        model.addConstr(-lmd*y[i] + mu*y[i-1] <= 1 )

    # |lmd*y_0 - mu*x_0| <= 1
    model.addConstr( lmd*y[0] - mu*x[0] <= 1 )
    model.addConstr(-lmd*y[0] + mu*x[0] <= 1 )

    # y_i = x_{i+1} - x_i for i=0..m-1
    for i in range(m):
        model.addConstr(y[i] == x[i+1] - x[i])

    # anchor to eliminate translation invariance
    model.addConstr(x[0] == 0)

    # solve
    model.optimize()

    # map Gurobi status code to string
    status_code = model.Status
    status_map = {
        GRB.OPTIMAL:          "OPTIMAL",
        GRB.UNBOUNDED:        "UNBOUNDED",
        GRB.INFEASIBLE:       "INFEASIBLE",
        GRB.INF_OR_UNBD:      "INF_OR_UNBOUNDED",
        GRB.TIME_LIMIT:       "TIME_LIMIT",
        GRB.SUBOPTIMAL:       "SUBOPTIMAL",
        GRB.INTERRUPTED:      "INTERRUPTED"
    }
    status = status_map.get(status_code, f"STATUS_{status_code}")

    obj_val = model.ObjVal if status_code == GRB.OPTIMAL else None
    return obj_val, status







