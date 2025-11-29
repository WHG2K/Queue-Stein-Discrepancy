# estimate_ksd_mtmm1_fast.py
import numpy as np
import matplotlib.pyplot as plt
from src.sim_mm1 import mm1_sim

# ---------- logistic lambda + bound ----------
def logistic_lambda(low=0.02, high=0.09, t0=1600.0, speed=120.0):
    def lam(t):
        z = (np.asarray(t) - t0) / speed
        return low + (high - low) / (1. + np.exp(-z))
    return lam

def lmax_over_window(lambda_t, t_min, t_max, num=1024):
    ts = np.linspace(t_min, t_max, num=num)
    return float(np.max(lambda_t(ts)))

# ---------- piecewise-linear basis ----------
def make_pwlinear_basis(knots):
    knots = np.asarray(knots, float)
    K = len(knots)
    def B(t):
        t = float(t)
        phi = np.zeros(K)
        if t <= knots[0]:  phi[0]  = 1.; return phi
        if t >= knots[-1]: phi[-1] = 1.; return phi
        j = np.searchsorted(knots, t)
        tL, tR = knots[j-1], knots[j]
        wR = (t - tL) / (tR - tL)
        phi[j-1] = 1. - wR; phi[j] = wR
        return phi
    return B, knots

def lam_from_theta(Bfun, theta, t):
    return float(np.exp(Bfun(t).dot(theta)))

# ---------- forward Kolmogorov (RK4) ----------
def _apply_Q_row(p_row, lam, mu, Kcap):
    out = np.zeros_like(p_row)
    out[0]     += -lam * p_row[0] + mu * p_row[1]
    for k in range(1, Kcap):
        out[k] += lam * p_row[k-1] - (lam + mu) * p_row[k] + mu * p_row[k+1]
    out[Kcap]  += lam * p_row[Kcap-1] - mu * p_row[Kcap]
    return out

def forward_mtmm1_rk4_param(t_eval, Kcap, theta, Bfun, mu, p0, dt_max):
    t_eval = np.asarray(t_eval, float)
    P = np.zeros((len(t_eval), Kcap+1))
    p = np.maximum(p0.copy(), 1e-14); p /= p.sum()
    P[0] = p
    for m in range(len(t_eval)-1):
        t0, t1 = t_eval[m], t_eval[m+1]
        dt = t1 - t0
        steps = max(1, int(np.ceil(dt / dt_max)))
        h = dt / steps
        t = t0
        for _ in range(steps):
            lam1 = lam_from_theta(Bfun, theta, t)
            k1 = _apply_Q_row(p, lam1, mu, Kcap)

            lam2 = lam_from_theta(Bfun, theta, t + 0.5*h)
            k2 = _apply_Q_row(p + 0.5*h*k1, lam2, mu, Kcap)

            k3 = _apply_Q_row(p + 0.5*h*k2, lam2, mu, Kcap)

            lam4 = lam_from_theta(Bfun, theta, t + h)
            k4 = _apply_Q_row(p + h*k3, lam4, mu, Kcap)

            p = p + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
            p = np.maximum(p, 1e-14); p /= p.sum()
            t += h
        P[m+1] = p
    return P

# ---------- discrete-score KSD ----------
def base_kernel_matrix(Kcap, gamma=0.5):
    xs = np.arange(Kcap+1)
    return np.exp(-gamma * np.abs(xs[:,None] - xs[None,:]))

def discrete_score_row(p_row, eps=1e-14):
    p = np.maximum(p_row, eps)
    psi = np.zeros_like(p)
    logp = np.log(p)
    psi[1:] = logp[1:] - logp[:-1]
    psi[0]  = 0.0
    return psi

def steinized_kernel_from_psi(K_base, psi):
    K = K_base
    Kx = np.zeros_like(K); Kx[:-1,:] = K[1:,:] - K[:-1,:]
    first = Kx + K * psi[:,None]
    KStein = np.zeros_like(K)
    KStein[:,:-1] = first[:,1:] - first[:,:-1]
    KStein += first * psi[None,:]
    return KStein

def window_indices(t_obs, t_center, h):
    t_obs = np.asarray(t_obs)
    return np.where(np.abs(t_obs - t_center) <= h)[0]

def ksd2_window(q_obs, idxs, K_Stein):
    if len(idxs) < 2: return 0.0
    Kcap = K_Stein.shape[0] - 1
    q = np.minimum(np.asarray(q_obs, int)[idxs], Kcap)
    M = len(q)
    return float(np.sum(K_Stein[q[:,None], q[None,:]]) / (M*M))

def second_diff_penalty(theta, alpha):
    if alpha <= 0 or len(theta) < 3: return 0.0
    d2 = theta[:-2] - 2*theta[1:-1] + theta[2:]
    return alpha * float(d2 @ d2)

def objective(theta, eta, t_obs, q_obs, t_grid, h, Bfun, Kcap, K_base,
              lambda_max_bound, p0, alpha_rough=5e-2):
    mu = float(np.exp(eta))
    dt_max = 0.35 / (lambda_max_bound + mu)
    P = forward_mtmm1_rk4_param(t_grid, Kcap, theta, Bfun, mu, p0, dt_max)
    total = 0.0
    for m, tm in enumerate(t_grid):
        idxs = window_indices(t_obs, tm, h)
        if len(idxs) < 2: 
            continue
        psi = discrete_score_row(P[m])
        K_Stein = steinized_kernel_from_psi(K_base, psi)
        total += ksd2_window(q_obs, idxs, K_Stein)
    total += second_diff_penalty(theta, alpha_rough)
    return float(total)

# ---------- optimizer (Adam + finite-diff grad) ----------
def fd_grad(fun, z, eps=1e-4):
    g = np.zeros_like(z, float)
    f0 = fun(z)
    for i in range(len(z)):
        old = z[i]
        z[i] = old + eps; f1 = fun(z)
        z[i] = old - eps; f2 = fun(z)
        z[i] = old
        g[i] = (f1 - f2) / (2*eps)
    return g, f0

def adam_minimize(fun, z0, lr=8e-2, steps=160, beta1=0.9, beta2=0.999, eps=1e-8,
                  callback=None, tol=1e-6):
    z = z0.copy()
    m = np.zeros_like(z); v = np.zeros_like(z)
    fprev = None
    for t in range(1, steps+1):
        g, fval = fd_grad(fun, z)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g*g)
        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)
        z -= lr * mhat / (np.sqrt(vhat) + eps)
        if callback and (t % 20 == 0):
            callback(t, fval, z, g)
        if fprev is not None and abs(fprev - fval) < tol:
            break
        fprev = fval
    return z, fval

# ===================== MAIN (fast settings) =====================
if __name__ == "__main__":
    # truth
    T = 4000.0
    lam_true = logistic_lambda(low=0.02, high=0.085, t0=1600.0, speed=120.0)
    mu_true  = 0.10
    mu_t     = lambda t: mu_true

    # simulate
    lmax = 1.05 * lmax_over_window(lam_true, 0.0, T)
    times, Q = mm1_sim(T, lam_true, mu_t, lmax=lmax, mumax=mu_true, init=0)
    t_obs = np.array(times, float)
    q_obs = np.array(Q, int)

    # evaluation grid & windows
    M = 120
    t_grid = np.linspace(t_obs.min(), t_obs.max(), M)
    h = 0.04 * (t_obs.max() - t_obs.min())   # wider window for stability

    # basis
    KNOTS = np.linspace(t_obs.min(), t_obs.max(), 12)
    Bfun, _ = make_pwlinear_basis(KNOTS)

    # state cap & kernel
    Kcap = int(q_obs.max()) + 10
    K_base = base_kernel_matrix(Kcap, gamma=0.5)

    # initial law
    p0 = np.zeros(Kcap+1); p0[min(q_obs[0], Kcap)] = 1.0

    # pack params
    theta0 = np.zeros(len(KNOTS))
    eta0   = np.log(0.09)  # close to truth but not exact
    z0 = np.concatenate([theta0, np.array([eta0])])

    lam_bound = lmax
    def fun(z):
        theta = z[:-1]; eta = z[-1]
        return objective(theta, eta, t_obs, q_obs, t_grid, h, Bfun,
                         Kcap, K_base, lam_bound, p0, alpha_rough=5e-2)

    def cb(step, fval, z, g):
        theta, mu_est = z[:-1], float(np.exp(z[-1]))
        med_lam = np.median(np.exp([Bfun(t).dot(theta) for t in t_grid]))
        print(f"[{step:3d}] obj={fval:.4e}  mu={mu_est:.4f}  median λ={med_lam:.4f}")

    z_hat, f_hat = adam_minimize(fun, z0, lr=8e-2, steps=160, callback=cb)
    theta_hat, mu_hat = z_hat[:-1], float(np.exp(z_hat[-1]))
    print(f"\nTrue mu={mu_true:.4f} | Estimated mu={mu_hat:.4f}")

    # curves
    lam_true_grid = lam_true(t_grid)
    lam_hat_grid  = np.exp(np.vstack([Bfun(t) for t in t_grid]) @ theta_hat)

    # plots
    plt.figure(figsize=(9,3.8))
    plt.plot(t_grid, lam_true_grid, lw=2, label="true λ(t)")
    plt.plot(t_grid, lam_hat_grid, lw=2, ls="--", label="estimated λ(t)")
    plt.xlabel("time"); plt.ylabel("rate"); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9,3.0))
    plt.step(t_obs, q_obs, where="post", lw=0.8)
    plt.xlabel("time"); plt.ylabel("Q(t)"); plt.tight_layout()
    plt.savefig("mm1_queue_length_trajectory.png", dpi=1300)