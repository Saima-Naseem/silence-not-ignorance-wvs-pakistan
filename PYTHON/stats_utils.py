"""
Shared estimation tools for Phase B (written with numpy/scipy only).

Models (maximum likelihood / quasi-likelihood):
  logit        binary outcome, or a proportion with weights (fractional / quasi-binomial logit)
  poisson      count outcome
  negbin       NB2 count outcome, log(alpha) estimated jointly
Standard errors: model-based (inverse Hessian) and cluster-robust sandwich,
V = H^-1 (sum_g s_g s_g') H^-1 * G/(G-1), the same formula used in the R scripts.
Validated against R in 04_validate_estimator.py.
"""
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

PROVINCES = ["Punjab", "Sindh", "KP", "Balochistan"]


def make_X(df, numeric, factors=None, interactions=None, intercept=True):
    """Design matrix as a DataFrame. factors = {col: [reference, level2, ...]};
    interactions = list of (a, b) numeric column pairs or (numeric, factor) pairs."""
    X = pd.DataFrame(index=df.index)
    if intercept:
        X["(Intercept)"] = 1.0
    for c in numeric:
        X[c] = df[c].astype(float)
    factors = factors or {}
    for f, levels in factors.items():
        for lv in levels[1:]:
            X[f"{f}[{lv}]"] = (df[f] == lv).astype(float)
    for a, b in interactions or []:
        if b in factors:
            for lv in factors[b][1:]:
                X[f"{a}:{b}[{lv}]"] = df[a].astype(float) * (df[b] == lv).astype(float)
        else:
            X[f"{a}:{b}"] = df[a].astype(float) * df[b].astype(float)
    return X


# ---------------------------------------------------------------- per-observation log-likelihood and scores
def _ll_scores(model, theta, y, X, w):
    if model in ("logit", "poisson"):
        eta = X @ theta
        if model == "logit":
            p = special.expit(eta)
            ll = w * (y * eta - np.logaddexp(0, eta))
            sc = (w * (y - p))[:, None] * X
        else:
            mu = np.exp(eta)
            ll = w * (y * eta - mu - special.gammaln(y + 1))
            sc = (w * (y - mu))[:, None] * X
        return ll, sc
    if model == "negbin":
        b, lna = theta[:-1], theta[-1]
        eta = X @ b; mu = np.exp(eta); a = np.exp(lna); r = 1.0 / a
        ll = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
              + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
        sb = ((y - mu) / (1 + a * mu))[:, None] * X
        dll_dr = special.digamma(y + r) - special.digamma(r) + np.log(r / (r + mu)) + 1 - (r + y) / (r + mu)
        sa = (-r * dll_dr)[:, None]
        return w * ll, w[:, None] * np.hstack([sb, sa])
    raise ValueError(model)


def fit(model, y, X, w=None, cluster=None, start=None):
    """Fit a model. y, X as pandas/np; returns a dict of results."""
    names = list(X.columns) + (["ln_alpha"] if model == "negbin" else [])
    y = np.asarray(y, float); Xm = np.asarray(X, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if start is None:
        if model == "negbin":
            start = np.append(fit("poisson", y, X, w)["coef"], 0.0)
        else:
            start = np.zeros(Xm.shape[1])
    f = lambda t: -_ll_scores(model, t, y, Xm, w)[0].sum()
    g = lambda t: -_ll_scores(model, t, y, Xm, w)[1].sum(axis=0)
    res = optimize.minimize(f, start, jac=g, method="BFGS", options={"gtol": 1e-9, "maxiter": 5000})
    th = res.x
    # numerical Hessian from analytic scores (central differences)
    k = len(th); H = np.zeros((k, k)); h = 1e-5
    for j in range(k):
        e = np.zeros(k); e[j] = h
        H[:, j] = (_ll_scores(model, th + e, y, Xm, w)[1].sum(0) - _ll_scores(model, th - e, y, Xm, w)[1].sum(0)) / (2 * h)
    H = (H + H.T) / 2
    Hinv = np.linalg.inv(-H)
    out = {"model": model, "names": names, "coef": th, "vcov_model": Hinv, "se_model": np.sqrt(np.diag(Hinv)),
           "ll": -res.fun, "n": len(y), "converged": bool(res.success) or np.abs(g(th)).max() < 1e-4,
           "max_grad": float(np.abs(g(th)).max())}
    if cluster is not None:
        sc = _ll_scores(model, th, y, Xm, w)[1]
        S = pd.DataFrame(sc).groupby(np.asarray(cluster)).sum().values
        G = S.shape[0]
        V = Hinv @ (S.T @ S) @ Hinv * G / (G - 1)
        out.update(vcov=V, se=np.sqrt(np.diag(V)), n_clusters=G)
    else:
        out.update(vcov=Hinv, se=out["se_model"])
    return out


def table(r, exp=True, digits=4):
    """Coefficient table with cluster-robust SE (if available), z, p, 95% CI and exp(b)."""
    b, se = r["coef"], r["se"]
    z = b / se; p = 2 * stats.norm.sf(np.abs(z))
    t = pd.DataFrame({"estimate": b, "se": se, "z": z, "p": p,
                      "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se}, index=r["names"])
    if exp:
        t["exp_b"] = np.exp(b); t["exp_ci_low"] = np.exp(t.ci_low); t["exp_ci_high"] = np.exp(t.ci_high)
    return t.round(digits)


def predict_mean(r, X):
    """Mean predicted outcome (count for poisson/negbin, probability for logit)."""
    b = r["coef"][:X.shape[1]]
    eta = np.asarray(X, float) @ b
    return special.expit(eta) if r["model"] == "logit" else np.exp(eta)


def ame_binary(r, df, build, var="female", ndraw=2000, seed=1):
    """Average marginal effect of a 0/1 variable: mean prediction at var=1 minus var=0,
    with a 95% interval from simulated draws of the (cluster-robust) coefficients."""
    d0, d1 = df.copy(), df.copy(); d0[var] = 0; d1[var] = 1
    X0, X1 = np.asarray(build(d0), float), np.asarray(build(d1), float)
    k = X0.shape[1]
    est = predict_mean(r, X1).mean() - predict_mean(r, X0).mean()
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(r["coef"], r["vcov"], ndraw)[:, :k]
    link = special.expit if r["model"] == "logit" else np.exp
    sims = link(X1 @ draws.T).mean(0) - link(X0 @ draws.T).mean(0)
    return {"ame": est, "ci_low": np.quantile(sims, .025), "ci_high": np.quantile(sims, .975),
            "pred0": predict_mean(r, X0).mean(), "pred1": predict_mean(r, X1).mean()}


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1, running)
    return adj


def smd(x_t, x_c):
    """Standardized mean difference (treated vs control), pooled SD."""
    s = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
    return (np.mean(x_t) - np.mean(x_c)) / s if s > 0 else 0.0
