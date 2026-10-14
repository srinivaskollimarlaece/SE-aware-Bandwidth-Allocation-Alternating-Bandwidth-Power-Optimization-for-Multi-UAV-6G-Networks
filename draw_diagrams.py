import itertools, numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, ACC, ACC2, WARN = "#1f2933", "#7b8794", "#1f5fa8", "#2f855a", "#b7791f"

# ---------------------------------------------------------------- Fig. 1: system model
rng = np.random.default_rng(7)
M = 6
xy = np.array([[10, 70], [40, 92], [78, 80], [92, 40], [55, 18], [18, 30]], float)
z = np.full(M, 1.0)
pairs = list(itertools.combinations(range(M), 2))
links = [(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4), (2, 5), (0, 3)]
G = nx.Graph(); G.add_edges_from(links)
L = nx.line_graph(G)
mis = [sorted(tuple(sorted(e)) for e in c) for c in nx.find_cliques(nx.complement(L))]
mis.sort(key=lambda c: (-len(c), c))
colors = ["#1f5fa8", "#2f855a", "#b7791f", "#9b2c2c", "#6b46c1", "#2c7a7b", "#975a16", "#4a5568"]

fig = plt.figure(figsize=(11, 4.4))
# (a) network
ax = fig.add_axes([0.02, 0.06, 0.40, 0.84]); ax.set_xlim(-5, 108); ax.set_ylim(-8, 110); ax.axis("off")
ax.set_title("(a) Multi-UAV network G(V, E)", fontsize=11, color=INK, loc="left")
ax.add_patch(FancyBboxPatch((-2, -6), 108, 112, boxstyle="round,pad=0,rounding_size=4",
                            fc="#f5f7fa", ec="#d9e2ec", lw=1))
show = mis[:1]
lk_col = {}
for qi, q in enumerate(show):
    for e in q: lk_col.setdefault(e, colors[qi])
for (i, j) in links:
    c = lk_col.get(tuple(sorted((i, j))), MUTED)
    ax.plot(*zip(xy[i], xy[j]), color=c, lw=2.4, zorder=1, solid_capstyle="round")
for v in range(M):
    x, y = xy[v]
    ax.add_patch(plt.Circle((x, y), 5.2, fc="white", ec=INK, lw=1.6, zorder=3))
    for dx, dy in ((-4.2, 4.2), (4.2, 4.2), (-4.2, -4.2), (4.2, -4.2)):
        ax.add_patch(plt.Circle((x + dx, y + dy), 1.7, fc="none", ec=INK, lw=1, zorder=3))
    ax.text(x, y, f"v{v+1}", ha="center", va="center", fontsize=9, color=INK, zorder=4, weight="bold")
ax.text(2, -3, "UAVs at altitude h; horizontal spacing U[1,100] m; vertical spacing 50 m",
        fontsize=7.5, color=MUTED)
# (b) MIS / bandwidth
ax2 = fig.add_axes([0.46, 0.06, 0.52, 0.84]); ax2.axis("off"); ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
ax2.set_title("(b) Link graph Ĝ → maximal independent sets → shared bands", fontsize=11, color=INK, loc="left")
y = 88
for qi, q in enumerate(mis[:5]):
    c = colors[0] if qi == 0 else MUTED
    ax2.add_patch(FancyBboxPatch((1, y - 6), 50, 11, boxstyle="round,pad=0,rounding_size=1.8",
                                 fc="white", ec=c, lw=1.6))
    ax2.text(4, y - 0.5, f"MIS q{qi+1}:  " + ",  ".join(f"(v{a+1},v{b+1})" for a, b in q),
             fontsize=8.6, va="center", color=INK)
    ax2.add_patch(FancyArrowPatch((52, y - 0.5), (60, y - 0.5), arrowstyle="-|>", mutation_scale=10, color=MUTED))
    y -= 15
ax2.text(4, y + 4, f"... {len(mis)} MISs in total. Links in one MIS share no UAV,\nso they can reuse the same band (q1 highlighted in (a)).",
         fontsize=7.8, color=MUTED, va="top")
# band bar
widths = np.array([0.30, 0.24, 0.18, 0.16, 0.12])
x0 = 62; total_h = 66; yb = 94
ax2.text(x0, 97, "B_tot = 10 MHz split into X_q", fontsize=8.6, color=INK)
for qi, w in enumerate(widths):
    h = w * total_h
    c = colors[0] if qi == 0 else MUTED
    ax2.add_patch(plt.Rectangle((x0, yb - h - 1), 14, h, fc=c, alpha=0.85 if qi == 0 else 0.3+0.1*(qi%2), ec="white", lw=1.5))
    ax2.text(x0 + 16, yb - h / 2 - 1, f"X_q{qi+1}", fontsize=8.4, va="center", color=INK)
    yb -= h
ax2.text(x0, yb - 3, "Base [2]:  max Σ_q |q|·X_q\nProposed:  max Σ_q (Σ_k s_qk R_k)·X_q",
         fontsize=8.2, color=INK, va="top", linespacing=1.6)
fig.savefig("fig_system_model.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: block diagram
fig, ax = plt.subplots(figsize=(11, 6)); ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, title, body, ec=INK, fc="white", tc=INK, fs=8.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=9.6, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.35)

def arrow(p, q, c=INK, txt=None, off=(0, 1.3), cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=c, lw=1.5, connectionstyle=cs))
    if txt: ax.text((p[0]+q[0])/2+off[0], (p[1]+q[1])/2+off[1], txt, ha="center", fontsize=7.8, color=c)

# top row
box(2, 40, 28, 19, "Inputs", "UAV positions, connection graph G\nB_tot, Pt_max, QoS factor β\nA2A path-loss model")
box(40, 40, 28, 19, "Graph model", "link graph Ĝ, enumerate MISs\nMIS matrix [s_qk]\nR_k and g_k at Pt = Pt_max")
box(78, 40, 30, 19, "Stage 1 (proposed)", "SE-aware LP (P1)\nmax Σ_q X_q Σ_k s_qk R_k\ns.t. Σ_q X_q = B_tot,  B_k ≥ τ B_tot g_k",
    ec=ACC, fc="#eaf2fb", tc=ACC)
arrow((30, 49.5), (40, 49.5)); arrow((68, 49.5), (78, 49.5))
# bottom row
box(78, 6, 30, 22, "Stage 2", "QoS targets c_kd = β B_k SE_kd(Pt_max)\nminimum-power LP\n(M-matrix solution for one link per UAV)")
arrow((93, 40), (93, 28), txt="B_k", off=(3, 0))
ax.add_patch(FancyBboxPatch((30, 3), 40, 28, boxstyle="round,pad=0,rounding_size=1.6", fc="#ebf7f0", ec=ACC2, lw=1.6))
ax.text(50, 29.3, "Stage 3 (proposed): alternating loop", ha="center", va="top", fontsize=9.6, weight="bold", color=ACC2)
box(32, 6, 17, 17, "(a) bandwidth", "fix Pt;\nLP (P2): re-allocate\nX_q to maximise\nheadroom t", ec=ACC2, fs=7.8)
box(51, 6, 17, 17, "(b) power", "fix B;\nminimum-power\nLP", ec=ACC2, fs=7.8)
arrow((49, 17), (51, 17), c=ACC2); arrow((51, 10), (49, 10), c=ACC2)
arrow((78, 17), (70, 17), txt="Pt", c=INK)
box(2, 6, 20, 22, "Output", "bandwidths B*\npowers Pt*\ndata rate, EE")
arrow((30, 17), (22, 17), txt="Σ Pt drop\n< 0.01 %", off=(0, 1.6), c=ACC2)
ax.text(2, 37.5, "Prop. 1: (P1) is an LP over the same feasible set as the base problem, so its rate at Pt_max is never lower.",
        fontsize=8, color=ACC, va="top")
ax.text(2, 34.2, "Prop. 2: Σ Pt never increases in Stage 3 and every rate target c_kd stays met, so the loop converges.",
        fontsize=8, color=ACC2, va="top")
fig.savefig("fig_block_diagram.png", dpi=300, bbox_inches="tight"); plt.close(fig)
