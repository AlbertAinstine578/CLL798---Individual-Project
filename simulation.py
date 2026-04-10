"""
Self-Organized Criticality in Neuronal Networks
================================================
Simulation of the Bak-Tang-Wiesenfeld (BTW) sandpile model adapted to
model neuronal avalanches in a 2D cortical network.

Reference system: Human cortical neural network exhibiting SOC.
CLL798 Individual Project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.optimize import curve_fit
import networkx as nx
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1.  GRID-BASED BTW SANDPILE / NEURAL MODEL
# ─────────────────────────────────────────────

class NeuronalSOCModel:
    """
    2-D lattice of integrate-and-fire neurons driven by random (Poisson) noise.

    Parameters
    ----------
    N          : int   – lattice side length  (N×N neurons)
    threshold  : int   – firing threshold  (analogous to sandpile critical slope)
    connectivity: str  – 'nearest4' | 'nearest8' | 'random_k'
    k          : int   – number of random neighbours (only for 'random_k')
    seed       : int   – RNG seed for reproducibility
    """

    def __init__(self, N=50, threshold=4, connectivity='nearest4', k=4, seed=42):
        self.N = N
        self.threshold = threshold
        self.connectivity = connectivity
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((N, N), dtype=int)
        self._build_adjacency()

    # ── adjacency ──────────────────────────────
    def _build_adjacency(self):
        """Pre-compute neighbour lists for every cell."""
        N = self.N
        self.neighbours = {}
        for i in range(N):
            for j in range(N):
                if self.connectivity == 'nearest4':
                    nb = []
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < N and 0 <= nj < N:
                            nb.append((ni, nj))
                    self.neighbours[(i,j)] = nb

                elif self.connectivity == 'nearest8':
                    nb = []
                    for di in [-1,0,1]:
                        for dj in [-1,0,1]:
                            if di==0 and dj==0:
                                continue
                            ni, nj = i+di, j+dj
                            if 0 <= ni < N and 0 <= nj < N:
                                nb.append((ni, nj))
                    self.neighbours[(i,j)] = nb

                elif self.connectivity == 'random_k':
                    # random k distinct neighbours (with replacement allowed at boundary)
                    all_cells = [(r,c) for r in range(N) for c in range(N)
                                 if not (r==i and c==j)]
                    chosen = [tuple(x) for x in
                              self.rng.choice(all_cells, size=min(self.k, len(all_cells)),
                                              replace=False)]
                    self.neighbours[(i,j)] = chosen

    # ── single drive step ──────────────────────
    def add_grain(self, i=None, j=None):
        """Add one unit of potential to a (random) neuron."""
        if i is None:
            i = self.rng.integers(0, self.N)
        if j is None:
            j = self.rng.integers(0, self.N)
        self.grid[i, j] += 1

    # ── topple / relax ─────────────────────────
    def relax(self):
        """
        Fire all super-threshold neurons iteratively until stable.
        Returns avalanche size (number of toppling events) and duration.
        """
        size = 0
        duration = 0
        unstable = True
        while unstable:
            unstable = False
            duration += 1
            fired_this_step = 0
            # find unstable neurons
            unstable_cells = list(zip(*np.where(self.grid >= self.threshold)))
            if not unstable_cells:
                break
            for (i, j) in unstable_cells:
                if self.grid[i, j] >= self.threshold:
                    self.grid[i, j] -= self.threshold
                    nb = self.neighbours[(i, j)]
                    for (ni, nj) in nb:
                        self.grid[ni, nj] += 1
                    fired_this_step += 1
                    unstable = True
            size += fired_this_step
        return size, max(duration - 1, 0)

    # ── run simulation ─────────────────────────
    def run(self, n_steps=50000, burn_in=5000):
        """
        Drive system with n_steps grain additions.
        Returns arrays of avalanche sizes and durations (excluding trivial size-0 events).
        """
        sizes = []
        durations = []
        for step in range(n_steps + burn_in):
            self.add_grain()
            s, d = self.relax()
            if step >= burn_in and s > 0:
                sizes.append(s)
                durations.append(d)
        return np.array(sizes), np.array(durations)


# ─────────────────────────────────────────────
# 2.  ANALYSIS UTILITIES
# ─────────────────────────────────────────────

def power_law_fit(data, x_min=1):
    """MLE fit of power-law exponent using Clauset et al. method."""
    data = data[data >= x_min]
    if len(data) < 10:
        return np.nan, np.nan, np.nan
    # MLE for discrete power law
    alpha = 1 + len(data) / np.sum(np.log(data / (x_min - 0.5)))
    # KS test
    x = np.sort(data)
    cdf_emp = np.arange(1, len(x)+1) / len(x)
    cdf_the = 1 - (x_min / x) ** (alpha - 1)
    ks = np.max(np.abs(cdf_emp - cdf_the))
    return alpha, ks, len(data)


def log_binned_histogram(data, bins=40):
    """Logarithmically binned frequency histogram."""
    data = data[data > 0]
    bins_arr = np.logspace(np.log10(data.min()), np.log10(data.max()), bins)
    counts, edges = np.histogram(data, bins=bins_arr)
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    return centres[mask], density[mask]


# ─────────────────────────────────────────────
# 3.  CONNECTIVITY SWEEP
# ─────────────────────────────────────────────

def connectivity_sweep(N=40, n_steps=30000, burn_in=3000):
    """
    Run simulations for nearest-4, nearest-8, and several random-k values.
    Returns dict of results.
    """
    configs = [
        ('nearest4 (k=4)',   'nearest4', 4),
        ('nearest8 (k=8)',   'nearest8', 8),
        ('random k=2',       'random_k', 2),
        ('random k=6',       'random_k', 6),
        ('random k=12',      'random_k', 12),
    ]
    results = {}
    for label, conn, k in configs:
        print(f"  Running: {label} ...")
        model = NeuronalSOCModel(N=N, threshold=4, connectivity=conn, k=k, seed=99)
        sizes, durations = model.run(n_steps=n_steps, burn_in=burn_in)
        alpha, ks, n_fit = power_law_fit(sizes, x_min=2)
        results[label] = {
            'sizes': sizes,
            'durations': durations,
            'alpha': alpha,
            'ks': ks,
            'mean_size': sizes.mean(),
            'k_eff': k,
            'connectivity': conn,
        }
        print(f"    → alpha={alpha:.3f}, mean_size={sizes.mean():.2f}, n_events={len(sizes)}")
    return results


# ─────────────────────────────────────────────
# 4.  FIGURE GENERATION
# ─────────────────────────────────────────────

def make_figures(results, save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

    COLORS = plt.cm.tab10(np.linspace(0, 0.8, len(results)))

    # ── Fig 1: Power-law distributions ─────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for (label, res), c in zip(results.items(), COLORS):
        x, y = log_binned_histogram(res['sizes'])
        axes[0].loglog(x, y, 'o-', markersize=4, color=c,
                       label=f"{label} (α={res['alpha']:.2f})", linewidth=1.5)
        x2, y2 = log_binned_histogram(res['durations'])
        axes[1].loglog(x2, y2, 'o-', markersize=4, color=c,
                       label=label, linewidth=1.5)

    # Reference slope
    xr = np.array([1, 1000])
    axes[0].loglog(xr, 3e-1 * xr**(-1.5), 'k--', linewidth=1.5, label='slope −1.5')
    axes[0].set_xlabel('Avalanche Size $s$')
    axes[0].set_ylabel('Probability Density $P(s)$')
    axes[0].set_title('(a) Avalanche Size Distribution')
    axes[0].legend(fontsize=8, loc='lower left')
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].set_xlabel('Avalanche Duration $T$')
    axes[1].set_ylabel('Probability Density $P(T)$')
    axes[1].set_title('(b) Avalanche Duration Distribution')
    axes[1].legend(fontsize=8, loc='lower left')
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig1_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig1_distributions.png")

    # ── Fig 2: Exponent vs connectivity ────────
    labels_ordered = list(results.keys())
    k_vals  = [results[l]['k_eff'] for l in labels_ordered]
    alphas  = [results[l]['alpha'] for l in labels_ordered]
    means   = [results[l]['mean_size'] for l in labels_ordered]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(k_vals, alphas, 'bs-', markersize=9, linewidth=2)
    axes[0].axhline(1.5, color='r', linestyle='--', label='Theoretical SOC exponent (1.5)')
    axes[0].set_xlabel('Effective Connectivity $k$')
    axes[0].set_ylabel('Power-law exponent $\\alpha$')
    axes[0].set_title('(a) Exponent vs. Connectivity')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for i, lbl in enumerate(labels_ordered):
        axes[0].annotate(lbl.split('(')[0].strip(), (k_vals[i], alphas[i]),
                         textcoords='offset points', xytext=(5, 5), fontsize=8)

    axes[1].plot(k_vals, means, 'r^-', markersize=9, linewidth=2)
    axes[1].set_xlabel('Effective Connectivity $k$')
    axes[1].set_ylabel('Mean Avalanche Size $\\langle s \\rangle$')
    axes[1].set_title('(b) Mean Avalanche Size vs. Connectivity')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig2_connectivity.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig2_connectivity.png")

    # ── Fig 3: Single run spatial snapshot ─────
    print("  Generating spatial snapshots...")
    model_snap = NeuronalSOCModel(N=60, threshold=4, connectivity='nearest4', seed=7)
    # warm up
    for _ in range(5000):
        model_snap.add_grain()
        model_snap.relax()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    snapshots = []
    aval_masks = []
    for frame in range(3):
        for _ in range(50):
            model_snap.add_grain()
        grid_before = model_snap.grid.copy()
        model_snap.add_grain(i=30, j=30)
        fired_map = np.zeros((60, 60))
        # simulate and track which cells fired
        old = model_snap.grid.copy()
        s, d = model_snap.relax()
        diff = (model_snap.grid - old)
        # use absolute difference as proxy for activity
        fired_map = np.abs(model_snap.grid - grid_before)
        snapshots.append(model_snap.grid.copy())
        aval_masks.append(fired_map)

    for ax, grid, mask, fr in zip(axes, snapshots, aval_masks, range(3)):
        im = ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=0, vmax=5)
        ax.set_title(f'Frame {fr+1}: Potential Field')
        ax.set_xlabel('Neuron column')
        ax.set_ylabel('Neuron row')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Potential')
    plt.suptitle('Spatiotemporal Snapshots of Neural Potential Field', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig3_spatial.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3_spatial.png")

    # ── Fig 4: Time series + cumulative ────────
    # use the nearest4 result
    res4 = results['nearest4 (k=4)']
    sizes_ts = res4['sizes'][:500]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].plot(sizes_ts, color='steelblue', alpha=0.7, linewidth=0.8)
    axes[0].set_ylabel('Avalanche Size')
    axes[0].set_xlabel('Event Index')
    axes[0].set_title('(a) Time Series of Avalanche Sizes (nearest-4 network)')
    axes[0].grid(alpha=0.3)

    # CCDF
    s_sorted = np.sort(res4['sizes'])
    ccdf = 1 - np.arange(1, len(s_sorted)+1)/len(s_sorted)
    axes[1].loglog(s_sorted, ccdf, color='darkorange', linewidth=1.5, label='CCDF data')
    xfit = np.logspace(0, np.log10(s_sorted.max()), 200)
    alpha_fit = res4['alpha']
    # normalise
    C = s_sorted.min()**(alpha_fit-1)
    axes[1].loglog(xfit, C * xfit**(-(alpha_fit-1)), 'k--',
                   linewidth=2, label=f'Power law fit α={alpha_fit:.2f}')
    axes[1].set_xlabel('Avalanche Size $s$')
    axes[1].set_ylabel('$P(S > s)$ (CCDF)')
    axes[1].set_title('(b) Complementary CDF of Avalanche Sizes')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig4_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig4_timeseries.png")

    # ── Fig 5: Network visualization ───────────
    print("  Building network visualisation...")
    G = nx.grid_2d_graph(10, 10)   # 10×10 regular grid as proxy
    pos = {(r,c): (c, -r) for r, c in G.nodes()}
    fig, ax = plt.subplots(figsize=(7, 7))
    # color nodes by random "potential" level
    rng_vis = np.random.default_rng(0)
    potentials = rng_vis.integers(0, 5, size=len(G.nodes()))
    node_colors = [plt.cm.YlOrRd(p/4) for p in potentials]
    nx.draw_networkx(G, pos=pos, node_color=node_colors, node_size=200,
                     with_labels=False, edge_color='grey', width=0.5, ax=ax)
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                norm=mcolors.Normalize(vmin=0, vmax=4))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Neuron Potential', shrink=0.7)
    ax.set_title('Neural Network Connectivity\n(10×10 nearest-4 lattice, colours = potential)')
    ax.axis('off')
    fig.savefig(os.path.join(save_dir, 'fig5_network.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig5_network.png")

    return save_dir


# ─────────────────────────────────────────────
# 5.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    SAVE_DIR = 'figures'
    N_STEPS  = 40000
    BURN_IN  = 5000

    print("=" * 60)
    print("  Neuronal Avalanches — SOC Simulation")
    print("=" * 60)

    print("\n[1/3] Running connectivity sweep ...")
    results = connectivity_sweep(N=40, n_steps=N_STEPS, burn_in=BURN_IN)

    print("\n[2/3] Generating figures ...")
    make_figures(results, save_dir=SAVE_DIR)

    print("\n[3/3] Summary table")
    print(f"\n{'Config':<25} {'alpha':>8} {'mean_s':>10} {'KS':>8}")
    print("-" * 55)
    for label, res in results.items():
        print(f"{label:<25} {res['alpha']:>8.3f} {res['mean_size']:>10.2f} {res['ks']:>8.4f}")

    print("\nDone. Figures saved to ./figures/")
