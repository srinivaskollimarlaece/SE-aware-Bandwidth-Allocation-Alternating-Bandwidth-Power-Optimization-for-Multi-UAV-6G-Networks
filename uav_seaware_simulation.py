# =============================================================================
# SE-aware Bandwidth Allocation + Alternating Bandwidth-Power Optimization
# for Multi-UAV 6G Networks  —  complete simulation (Google Colab ready)
#
# Base paper: M. A. Alnakhli, E. M. Mohamed, M. M. Fouda, "Bandwidth Allocation
# and Power Control Optimization for Multi-UAVs Enabled 6G Network",
# IEEE Access, vol. 12, pp. 67405-67415, 2024.
#
# HOW TO RUN IN COLAB
#   1. Open https://colab.research.google.com -> New notebook
#   2. Paste this whole file into one cell and press Run (Shift+Enter)
#      (quick test first: set TRIALS = 20; final paper run: TRIALS = 200)
#   3. When finished, a file "results.zip" downloads automatically
#      (CSV tables + PNG figures for the paper).
# =============================================================================
TRIALS       = 200          # Monte Carlo trials per configuration (use 20 to test)
UAV_COUNTS   = [3, 4, 5, 6, 7, 8]
LARGE_COUNTS = [9, 10]      # scalability study
DENSITIES    = [0.5, 0.8, 1.0]
BETA         = 0.8          # QoS: each link must keep >= BETA of its stage-1 rate
MAX_ALT_IT   = 20           # max iterations of the alternating stage
SEED         = 2026
OUT          = "results"

import os, time, itertools, zipfile, warnings
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
BTOT, PMAX = 10e6, 1.0                      # Hz, W            (base Table 1)
PL0, D_V   = 46.4, 50.0                     # dB, m            (base Table 1)
RHO0 = 10 ** ((-147 + 10*np.log10(BTOT) + 10 - 30) / 10)   # W (base Table 1)
CHANNELS = {"LoS (base)":            dict(alpha=2.03, shadow=0.0),
            "LoS + 4 dB shadowing":  dict(alpha=2.03, shadow=4.0),
            "Mixed LoS/NLoS":        dict(alpha=2.03, shadow=6.0, nlos_alpha=3.0, p_los=0.7)}

# ---------------------------------------------------------------- scenario
def make_scenario(M, density, rng, alpha=2.03, shadow=0.0, nlos_alpha=None, p_los=1.0):
    """Random multi-UAV network exactly as in Sec. V of the base paper."""
    dH = np.zeros((M, M)); iu = np.triu_indices(M, 1)
    dH[iu] = rng.uniform(1, 100, len(iu[0])); dH = dH + dH.T
    d = np.sqrt(dH**2 + D_V**2); np.fill_diagonal(d, 1.0)
    a = np.full((M, M), alpha)
    if nlos_alpha is not None:
        los = rng.uniform(size=(M, M)) < p_los; los = np.triu(los, 1); los = los | los.T
        a = np.where(los, alpha, nlos_alpha)
    pl = PL0 + 10*a*np.log10(d)
    if shadow > 0:
        sh = rng.normal(0, shadow, (M, M)); sh = np.triu(sh, 1); pl = pl + sh + sh.T
    g = 10 ** (-pl / 10)
    pairs = list(itertools.combinations(range(M), 2))
    n_e = max(1, int(round(density * len(pairs))))
    links = [pairs[i] for i in sorted(rng.choice(len(pairs), n_e, replace=False))]
    G = nx.Graph(); G.add_nodes_from(range(M)); G.add_edges_from(links)
    active = np.array([G.degree(v) > 0 for v in range(M)])
    L = nx.line_graph(G)                                   # link graph (Fig. 3)
    mis = [set(c) for c in nx.find_cliques(nx.complement(L))] if len(links) > 1 else [set(L.nodes)]
    pos = {e: k for k, e in enumerate(links)}
    S = np.zeros((len(mis), len(links)))                   # MIS matrix, Eq. (9)
    for q, c in enumerate(mis):
        for e in c: S[q, pos[tuple(sorted(e))]] = 1
    return dict(M=M, g=g, links=links, S=S, active=active)

def dir_se(P, sc):
    """Per-direction spectral efficiency log2(1+SINR), Eq. (4)-(5); shape (K,2)."""
    g, act = sc["g"], sc["active"]; out = np.zeros((len(sc["links"]), 2))
    for k, (i, j) in enumerate(sc["links"]):
        for c, (t, r) in enumerate(((i, j), (j, i))):
            m = act.copy(); m[t] = m[r] = False
            out[k, c] = np.log2(1 + P[t]*g[t, r] / ((P[m]*g[m, r]).sum() + RHO0))
    return out

def total_rate(sc, B, P):            # bit/s
    return float((B[:, None] * dir_se(P, sc)).sum())

# ---------------------------------------------------------------- stage 1
def fairness_tau(S, gk):
    """Largest tau<=1 for which constraint (11.f) B_k >= tau*Btot*g_k is feasible."""
    Q, K = S.shape
    r = linprog(np.r_[np.zeros(Q), -1], A_ub=np.hstack([-S.T, (BTOT*gk)[:, None]]),
                b_ub=np.zeros(K), A_eq=np.r_[np.ones(Q), 0][None], b_eq=[BTOT],
                bounds=[(0, None)]*Q + [(0, 1)], method="highs")
    return min(1.0, r.x[-1]) * (1 - 1e-9)

def bw_base(sc, R):
    """Base paper, problem (11): max sum_k B_k  (spectral efficiency NOT in objective).
    Solved exactly (it is an LP); SQP of the base paper converges to the same point."""
    S = sc["S"]; Q = S.shape[0]; gk = R / R.sum(); tau = fairness_tau(S, gk)
    r = linprog(-S.sum(1), A_ub=-S.T, b_ub=-tau*BTOT*gk, A_eq=np.ones((1, Q)),
                b_eq=[BTOT], bounds=[(0, None)]*Q, method="highs")
    return r.x

def bw_se_aware(sc, R):
    """PROPOSED stage 1: max sum_k B_k R_k  (original objective (6)) under the same
    constraints (11.a)-(11.f). Exact LP -> global optimum."""
    S = sc["S"]; Q = S.shape[0]; gk = R / R.sum(); tau = fairness_tau(S, gk)
    r = linprog(-(S @ R), A_ub=-S.T, b_ub=-tau*BTOT*gk, A_eq=np.ones((1, Q)),
                b_eq=[BTOT], bounds=[(0, None)]*Q, method="highs")
    return r.x

def bw_equal(sc, R):  return np.full(sc["S"].shape[0], BTOT / sc["S"].shape[0])

def bw_sqp_random(sc, R, rng):
    """SLSQP on (11) from a random start (Algorithm 1 of the base paper)."""
    S = sc["S"]; Q = S.shape[0]; gk = R / R.sum(); tau = fairness_tau(S, gk)
    res = minimize(lambda x: -S.sum(1) @ x, rng.dirichlet(np.ones(Q)), jac=lambda x: -S.sum(1),
                   method="SLSQP", bounds=[(0, 1)]*Q,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1},
                                {"type": "ineq", "fun": lambda x: S.T @ x - tau*gk}],
                   options=dict(maxiter=300))
    return np.clip(res.x, 0, None) * BTOT

# ---------------------------------------------------------------- stage 2
def min_power(sc, B, dem):
    """min sum Pt s.t. per-direction rate >= dem, 0<=Pt<=Pmax  (problem (19)).
    With phi fixed this is linear in Pt; solved exactly as an LP (identical to the
    M-matrix / Sherman-Morrison solution when each UAV has one link)."""
    g, act, M = sc["g"], sc["active"], sc["M"]; rows = []
    for k, (i, j) in enumerate(sc["links"]):
        for c, (t, r) in enumerate(((i, j), (j, i))):
            if dem[k, c] <= 0: continue
            if B[k] <= 1.0: return None
            phi = 2 ** min(dem[k, c] / B[k], 60) - 1
            row = np.zeros(M); m = act.copy(); m[t] = m[r] = False
            row[t] = -g[t, r] / RHO0; row[m] = phi * g[m, r] / RHO0
            rows.append(row / phi)
    if not rows: return np.zeros(M)
    r = linprog(np.ones(M), A_ub=np.array(rows), b_ub=-np.ones(len(rows)),
                bounds=[(0, PMAX) if a else (0, 0) for a in act], method="highs")
    return r.x if r.status == 0 else None

def bw_headroom(sc, P, dem):
    """PROPOSED alternating step: for fixed powers, re-allocate MIS bandwidth to
    maximise the common QoS headroom t:  B_kd * SE_kd(P) >= t * dem_kd  (LP)."""
    S = sc["S"]; Q, K = S.shape; se = dir_se(P, sc); rows = []
    for k in range(K):
        for c in range(2):
            if dem[k, c] > 0:
                row = np.zeros(Q + 1); row[:Q] = -S[:, k]*se[k, c]/dem[k, c]; row[Q] = 1
                rows.append(row)
    r = linprog(np.r_[np.zeros(Q), -1], A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                A_eq=np.r_[np.ones(Q), 0][None], b_eq=[BTOT],
                bounds=[(0, None)]*(Q + 1), method="highs")
    return r.x[:Q]

def alternating(sc, X, P, dem):
    """Block-coordinate descent; total power is non-increasing at every step
    (the previous P stays feasible for the new bandwidths), so it converges."""
    hist = [P.sum()]
    for _ in range(MAX_ALT_IT):
        Xn = bw_headroom(sc, P, dem); Pn = min_power(sc, sc["S"].T @ Xn, dem)
        if Pn is None or Pn.sum() >= P.sum() * (1 - 1e-4): break
        X, P = Xn, Pn; hist.append(P.sum())
    return X, P, hist

# ---------------------------------------------------------------- one trial
def run_trial(sc, rng):
    out = {}
    P0 = np.where(sc["active"], PMAX, 0.0)
    se0 = dir_se(P0, sc); R = se0.sum(1)                      # R_k, Eq. (7)
    S = sc["S"]

    def pipeline(name, X, alt=False, pc=True):
        t0 = time.perf_counter()
        B = S.T @ X
        dem = BETA * B[:, None] * se0
        P = min_power(sc, B, dem) if pc else P0
        it = 0
        if P is None: P = P0
        if alt and pc:
            X, P, h = alternating(sc, X, P, dem); B = S.T @ X; it = len(h) - 1
        el = (time.perf_counter() - t0) * 1e3
        rl = (B[:, None] * dir_se(P, sc)).sum(1)               # per-link rate
        dr = rl.sum(); pw = P.sum()
        out[name] = dict(DR=dr/1e6, EE=dr/pw/1e9 if pw > 0 else np.nan, Ptot=pw,
                         Jain=rl.sum()**2/(len(rl)*(rl**2).sum()), MinRate=rl.min()/1e6,
                         alt_iters=it, t_ms=el)

    t0 = time.perf_counter(); Xb = bw_base(sc, R); tb = (time.perf_counter()-t0)*1e3
    t0 = time.perf_counter(); Xs = bw_se_aware(sc, R); ts = (time.perf_counter()-t0)*1e3
    Xq = bw_sqp_random(sc, R, rng)
    pipeline("Base SQP (Alnakhli 2024)", Xb)
    pipeline("Base SQP-WoPC", Xb, pc=False)
    pipeline("Base SLSQP random start", Xq)
    pipeline("Equal MIS bandwidth", bw_equal(sc, R))
    pipeline("Proposed: SE-aware only", Xs)
    pipeline("Proposed: alternating only", Xb, alt=True)
    pipeline("Proposed (SE-aware + alternating)", Xs, alt=True)
    pipeline("Proposed-WoPC", Xs, pc=False)
    # Random Selection: random bandwidth and random power (base paper)
    Br = rng.uniform(0, 1, len(R)); Br *= BTOT / Br.sum(); Pr = rng.uniform(0, PMAX, sc["M"]) * sc["active"]
    rl = (Br[:, None] * dir_se(Pr, sc)).sum(1); dr = rl.sum()
    out["Random Selection"] = dict(DR=dr/1e6, EE=dr/Pr.sum()/1e9, Ptot=Pr.sum(),
                                   Jain=rl.sum()**2/(len(rl)*(rl**2).sum()),
                                   MinRate=rl.min()/1e6, alt_iters=0, t_ms=0)
    out["Base SQP (Alnakhli 2024)"]["t_stage1_ms"] = tb
    out["Proposed (SE-aware + alternating)"]["t_stage1_ms"] = ts
    return out

def run_config(M, dens, trials, chan="LoS (base)", tag="main"):
    rows = []
    for t in range(trials):
        rng = np.random.default_rng(SEED + 100000*M + int(1000*dens) + t)
        sc = make_scenario(M, dens, rng, **CHANNELS[chan])
        for meth, v in run_trial(sc, rng).items():
            rows.append(dict(exp=tag, channel=chan, N=M, density=dens, trial=t, method=meth, **v))
    return rows

# ---------------------------------------------------------------- statistics
PROP = "Proposed (SE-aware + alternating)"
def summarise(df):
    res = []
    for (exp, ch, N, d), g in df.groupby(["exp", "channel", "N", "density"]):
        ref = g[g.method == PROP].sort_values("trial")
        for m, gm in g.groupby("method"):
            gm = gm.sort_values("trial"); row = dict(exp=exp, channel=ch, N=N, density=d, method=m)
            for met in ("DR", "EE", "Jain", "MinRate", "Ptot", "t_ms", "alt_iters"):
                x = gm[met].dropna().values
                row[f"{met}_mean"] = x.mean(); row[f"{met}_std"] = x.std(ddof=1)
                row[f"{met}_ci95"] = 1.96*x.std(ddof=1)/np.sqrt(len(x)); row[f"{met}_median"] = np.median(x)
            if m != PROP:
                for met in ("DR", "EE"):
                    a, b = ref[met].values, gm[met].values
                    row[f"gain_{met}_%"] = 100*(a.mean()/b.mean() - 1)
                    if np.allclose(a, b):            # identical allocations -> no test
                        row[f"p_{met}"] = np.nan; continue
                    try: p = wilcoxon(a, b, alternative="greater").pvalue
                    except ValueError: p = 1.0
                    row[f"p_{met}"] = 1.0 if np.isnan(p) else p
            res.append(row)
    s = pd.DataFrame(res)
    for met in ("DR", "EE"):                       # Holm correction within each config
        for key, g in s.groupby(["exp", "channel", "N", "density"]):
            idx = g[f"p_{met}"].dropna().sort_values().index; m = len(idx); run = 0
            for r_, i in enumerate(idx):
                run = max(run, min(1, (m - r_) * s.at[i, f"p_{met}"])); s.at[i, f"p_{met}_holm"] = run
    return s

# ---------------------------------------------------------------- run everything
all_rows = []
T0 = time.time()
for d in DENSITIES:
    for M in UAV_COUNTS:
        all_rows += run_config(M, d, TRIALS); print(f"main  density={d}  N={M}  done  ({time.time()-T0:.0f}s)")
for M in LARGE_COUNTS:
    all_rows += run_config(M, 0.5, max(20, TRIALS//4), tag="scale"); print(f"scale N={M} done")
for ch in list(CHANNELS)[1:]:
    for M in (4, 6, 8):
        all_rows += run_config(M, 0.8, TRIALS, chan=ch, tag="channel"); print(f"channel {ch} N={M} done")
raw = pd.DataFrame(all_rows); raw.to_csv(f"{OUT}/raw_all_trials.csv", index=False)
summ = summarise(raw); summ.to_csv(f"{OUT}/summary_stats.csv", index=False)

# beta sweep (QoS level vs EE trade-off)
beta_rows = []
for b in (0.5, 0.6, 0.7, 0.8, 0.9):
    BETA = b
    for r in run_config(6, 0.8, max(20, TRIALS//2), tag=f"beta={b}"):
        if r["method"] in (PROP, "Base SQP (Alnakhli 2024)"): r["beta"] = b; beta_rows.append(r)
BETA = 0.8
bdf = pd.DataFrame(beta_rows)
bdf.groupby(["beta", "method"])[["DR", "EE", "Jain"]].agg(["mean", "std"]).to_csv(f"{OUT}/beta_sweep.csv")

# convergence curve of the alternating stage (N=5, density 0.8)
conv = []
for t in range(30):
    rng = np.random.default_rng(777 + t); sc = make_scenario(5, 0.8, rng)
    P0 = np.where(sc["active"], PMAX, 0.0); se0 = dir_se(P0, sc); R = se0.sum(1)
    X = bw_se_aware(sc, R); B = sc["S"].T @ X; dem = BETA * B[:, None] * se0
    P = min_power(sc, B, dem)
    if P is None: continue
    _, _, h = alternating(sc, X, P, dem); conv.append(np.array(h) / h[0])

# ---------------------------------------------------------------- figures
main = summ[summ.exp == "main"]
show = ["Proposed (SE-aware + alternating)", "Base SQP (Alnakhli 2024)",
        "Equal MIS bandwidth", "Random Selection"]
style = {show[0]: ("o", "-"), show[1]: ("s", "--"), show[2]: ("^", ":"), show[3]: ("x", "-.")}
for met, lab, log in (("DR", "Sum data rate (Mbps)", False), ("EE", "Energy efficiency (Gbit/J)", True)):
    for d in DENSITIES:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for m in show:
            g = main[(main.density == d) & (main.method == m)].sort_values("N")
            mk, ls = style[m]
            ax.errorbar(g.N, g[f"{met}_mean"], yerr=g[f"{met}_ci95"], marker=mk, ls=ls,
                        capsize=3, ms=5, label=m)
        ax.set_xlabel("Number of UAVs"); ax.set_ylabel(lab); ax.grid(alpha=.3)
        if log: ax.set_yscale("log")
        ax.set_title(f"Network density = {d}  (mean ± 95% CI, {TRIALS} trials)", fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_{met}_density_{d}.png", dpi=300)
        plt.close(fig)
# ablation bar chart (N=6, density 0.8)
abl = ["Base SQP (Alnakhli 2024)", "Proposed: alternating only", "Proposed: SE-aware only",
       "Proposed (SE-aware + alternating)"]
g = main[(main.N == 6) & (main.density == 0.8)].set_index("method").loc[abl]
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
for a, met, lab in zip(ax, ("DR", "EE"), ("Sum data rate (Mbps)", "EE (Gbit/J)")):
    a.bar(range(4), g[f"{met}_mean"], yerr=g[f"{met}_ci95"], capsize=3, color=["#888", "#9ab", "#69c", "#247"])
    a.set_xticks(range(4)); a.set_xticklabels(["Base", "+Alt.", "+SE-aware", "Full"], fontsize=8)
    a.set_ylabel(lab); a.grid(axis="y", alpha=.3)
fig.suptitle("Ablation (N = 6, density = 0.8)", fontsize=9); fig.tight_layout()
fig.savefig(f"{OUT}/fig_ablation.png", dpi=300); plt.close(fig)
# convergence
fig, ax = plt.subplots(figsize=(5, 3.5))
L = max(len(c) for c in conv)
C = np.array([np.r_[c, np.full(L - len(c), c[-1])] for c in conv])
for c in C: ax.plot(c, color="#ccc", lw=.8)
ax.plot(C.mean(0), "o-", color="#247", label="mean")
ax.set_xlabel("Alternating iteration"); ax.set_ylabel("Total power / initial")
ax.set_title("Monotone convergence of the alternating stage", fontsize=9)
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_convergence.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- print paper tables
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
cols = ["N", "density", "method", "DR_mean", "DR_ci95", "EE_mean", "EE_ci95", "Jain_mean",
        "gain_DR_%", "gain_EE_%", "p_DR_holm", "p_EE_holm"]
key = ["Proposed (SE-aware + alternating)", "Base SQP (Alnakhli 2024)", "Equal MIS bandwidth", "Random Selection"]
print("\n=== TABLE A: main results (mean ± 95% CI); gains/p-values are Proposed vs. method ===")
print(main[main.method.isin(key)][cols].round(4).to_string(index=False))
print("\n=== TABLE B: ablation ===")
print(main[main.method.isin(abl + ["Proposed-WoPC", "Base SQP-WoPC", "Base SLSQP random start"])
           & main.N.isin([4, 6, 8]) & (main.density == 0.8)][cols].round(4).to_string(index=False))
print("\n=== TABLE C: scalability and channel robustness ===")
print(summ[summ.exp.isin(["scale", "channel"]) & summ.method.isin(key[:2])]
      [["exp", "channel"] + cols].round(4).to_string(index=False))
print("\n=== TABLE D: runtime (ms per network) ===")
print(summ[summ.method.isin(key[:2]) & (summ.exp != "channel")].groupby(["N", "method"])["t_ms_mean"].mean().round(2).to_string())
print("\n=== TABLE E: QoS factor beta sweep (N=6, density 0.8) ===")
print(bdf.groupby(["beta", "method"])[["DR", "EE", "Jain"]].mean().round(4).to_string())
print(f"\nTotal time {time.time()-T0:.0f} s")

with zipfile.ZipFile("results.zip", "w") as z:
    for f in os.listdir(OUT): z.write(os.path.join(OUT, f))
try:
    from google.colab import files; files.download("results.zip")
except Exception:
    print("results.zip written (download it from the Colab file panel if it did not start)")
