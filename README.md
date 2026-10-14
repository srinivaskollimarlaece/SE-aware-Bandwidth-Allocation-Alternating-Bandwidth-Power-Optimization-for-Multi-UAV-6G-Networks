# SE-aware Bandwidth Allocation and Alternating Bandwidth–Power Optimization for Multi-UAV 6G Networks

Simulation code for the paper *"Spectral-Efficiency-Aware Graph-Based Bandwidth Allocation and Alternating Power Control for Energy-Efficient Multi-UAV 6G Networks"* by K. Srinivasa Rao, P. Bharat Kumar, S. Kannappan and G. Jyothi.

The system model follows M. A. Alnakhli, E. M. Mohamed and M. M. Fouda, "Bandwidth allocation and power control optimization for multi-UAVs enabled 6G network," *IEEE Access*, vol. 12, pp. 67405–67415, 2024.

## Method

1. **Stage 1, SE-aware bandwidth allocation:** maximises Σ_k B_k R_k over the maximal independent sets (MISs) of the link graph. This is an exact linear program.
2. **Stage 2, minimum-power control:** finds the lowest transmit powers that meet each link's rate target (LP / M-matrix).
3. **Stage 3, alternating loop:** re-allocates bandwidth at fixed power, then minimises power at fixed bandwidth, and repeats. Total power never increases.

## Files

| File | Content |
|---|---|
| `UAV_SE_aware.ipynb` | Google Colab notebook (Runtime → Run all) |
| `uav_seaware_simulation.py` | The same code as a Python script |
| `draw_diagrams.py` | Draws the system-model and block diagrams (Figs. 1–2) |
| `results/` | `raw_all_trials.csv`, `summary_stats.csv`, `beta_sweep.csv` |
| `figures/` | All figures used in the paper |

## How to run

```
pip install numpy scipy pandas networkx matplotlib
python uav_seaware_simulation.py
```

Set `TRIALS = 20` at the top of the script for a quick test, or `TRIALS = 200` to reproduce the paper (about 5–8 minutes). The seed is fixed (`SEED = 2026`), so the results are reproducible.
