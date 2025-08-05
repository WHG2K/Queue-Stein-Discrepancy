import numpy as np
import argparse
from dotenv import load_dotenv
from scipy.stats import wasserstein_distance
# from src.discrepancy import SD_MM1_easy
import gurobipy as gp
from gurobipy import GRB


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with customizable parameters.")
    
    parser.add_argument('--node', type=str, default=None, help='which node to run on')
    parser.add_argument("--data", type=str, default="", help='data file name')

    return parser.parse_args()


def compute_w1_easy(q, rho):
    """
    Compute the 1-Wasserstein distance between an empirical PMF q SUPPORTED ON {0,1,...,m}
    and the M/M/1 steady-state distribution with load rho, whose theoretical PMF is truncated 
    at k = m + 10 and tail mass placed at that point.
    """
    # determine empirical support size
    m = len(q) - 1
    max_k = m + 100

    # build full support [0,1,...,max_k]
    support = np.arange(max_k + 1)

    # empirical weights: q[0..m] plus zeros up to max_k
    emp_weights = np.concatenate([q, np.zeros(max_k + 1 - len(q))])

    # theoretical PMF for k=0..max_k-1
    th_pmf = [(1 - rho) * rho**k for k in range(max_k)]
    # append tail mass at k = max_k
    tail_mass = rho**(max_k)
    th_pmf.append(tail_mass)
    th_weights = np.array(th_pmf)

    # normalize just in case
    emp_weights = emp_weights / emp_weights.sum()
    th_weights  = th_weights  / th_weights.sum()

    # compute and return 1-Wasserstein distance
    return wasserstein_distance(support, support, emp_weights, th_weights)


def SD_MM1_easy(q, lmd=0.07, mu=0.10, verbose=False):
    """
    Compute the graphic stein discrepancy of the samples Q and the steady state M/M/1, by solving the LP:
      max_{x[0..m], y[0..m-1]} q0*lmd*x0 + sum_{i=1}^m qi*(lmd*xi - mu*x[i-1])
      s.t. |lmd*y_i - mu*y_{i-1}| <= 1   (i=1..m-1)
           |lmd*y_0 - mu*x_0| <= 1
           y_i = x_{i+1} - x_i           (i=0..m-1)
           |y_{m-1}| <= 1 / (mu - lmd)
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

    # stability constraint
    model.addConstr(y[m-1] <= 1.0 / (mu - lmd))
    model.addConstr(y[m-1] >= 1.0 / (mu - lmd))

    # # anchor to eliminate translation invariance
    # model.addConstr(x[0] == 0)

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



if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # loading gurobi environment
    if not args.node:
        load_dotenv(override=True)
    else:
        load_dotenv(override=True, dotenv_path=f'{args.node}.env')



    # # example empirical PMF q0..qm
    # q = [0.2, 0.3, 0.5]        # replace with your data
    # lmd, mu = 0.07, 0.10
    # rho = lmd/mu

    # obj_lp, status = SD_MM1_easy(q, lmd=lmd, mu=mu, verbose=False)
    # print(f"Status: {status}")
    # w1    = compute_w1_easy(q, rho)

    # print(f"LP objective = {obj_lp:.8f}")
    # print(f"Wasserstein-1 = {w1:.8f}")
    # print("difference    =", abs(obj_lp - w1))


    # np.random.seed(0)
    num_tests = 10
    lmd, mu = 0.07, 0.10
    rho = lmd / mu

    for i in range(1, num_tests+1):
        # random dimension m between 5 and 20
        m = np.random.randint(5, 21)
        # sample q from Dirichlet to get a random PMF of length m+1
        q = np.random.dirichlet(alpha=np.ones(m+1))

        obj_lp, status = SD_MM1_easy(q, lmd=lmd, mu=mu)
        w1_value = compute_w1_easy(q, rho)

        diff = None
        if obj_lp is not None:
            diff = abs(obj_lp - w1_value)

        print(f"Test {i}: m={m}, status={status}, "
              f"LP={obj_lp:.6f}, W1={w1_value:.6f}, diff={diff}")
    