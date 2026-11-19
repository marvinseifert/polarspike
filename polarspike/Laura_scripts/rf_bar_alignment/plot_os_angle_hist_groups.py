"""Angle between RF long axis and preferred bar orientation (bar long edge = motion axis + 90),
orientation-selective cells (OSI > OSI_MIN) of the final clustering (12 + 32 px), per elongation group.
Four 22.5 deg bins: with no relation each holds 25 % of cells.
Bars: fraction of cells; lines: 95 % Wilson confidence interval.
p (aligned bin): bar orientations shuffled among the cells of that group 10,000x; fraction of shuffles
with at least as many cells within 22.5 deg (one-sided).
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OSI_MIN, N = 0.10, 10000
o = pd.read_csv(sys.argv[1]); out = sys.argv[2]
o['bar_orient'] = (o.motion_axis + 90) % 180
t = o[o.osi > OSI_MIN].reset_index(drop=True)
diff = lambda a, b: np.abs((a - b + 90) % 180 - 90)
t['delta'] = diff(t.rf_axis, t.bar_orient)
rng = np.random.default_rng(0)
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
GROUPS = [('Round (1.00-1.05)', t[t.elong < 1.05], '#8fb8de'),
          ('Slightly elongated (1.05-1.20)', t[(t.elong >= 1.05) & (t.elong < 1.2)], '#3d85c6'),
          ('Elongated (≥ 1.20)', t[t.elong >= 1.2], '#0b3d79')]
BINS = [0, 22.5, 45, 67.5, 90]
NAMES = ['0-22.5°\naligned', '22.5-45°', '45-67.5°', '67.5-90°\nacross']


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pstr(p): return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'


fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (name, g, col) in zip(axes, GROUPS):
    n = len(g); h, _ = np.histogram(g.delta, BINS)
    for i, k in enumerate(h):
        lo, hi = wilson(k, n)
        ax.bar(i, k / n, width=0.72, color=col if i == 0 else col, alpha=1 if i == 0 else 0.45)
        ax.plot([i, i], [lo, hi], color=INK, lw=1.3)
        ax.text(i, hi + 0.012, f'{100 * k / n:.0f}%', ha='center', fontsize=8.5, color=INK)
    real = np.mean(g.delta <= 22.5)
    a, b = g.rf_axis.values, g.bar_orient.values
    null = np.array([np.mean(diff(a, rng.permutation(b)) <= 22.5) for _ in range(N)])
    p = (np.sum(null >= real) + 1) / (N + 1)
    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(4)); ax.set_xticklabels(NAMES, fontsize=8.5)
    ax.set_title(f'{name}\nn = {n};  aligned {100 * real:.0f}% vs 25%, {pstr(p)}', loc='left', fontsize=10, color=INK)
    ax.set_ylim(0, 0.62); ax.set_xlim(-0.5, 3.5)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel('Angle between RF long axis and preferred bar', fontsize=9, color=INK)
axes[0].set_ylabel('Fraction of cells (± 95% CI)', fontsize=9.5, color=INK)
axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
fig.suptitle(f'Orientation-selective cells (OSI > {OSI_MIN}, n = {len(t)}): how well does the preferred bar line up with the RF?',
             x=0.07, y=0.99, ha='left', fontsize=13, color=INK)
fig.text(0.07, 0.885, 'Final clustering cells, 9 recordings (12 px at 5 µm/px and 32 px). Preferred bar = its long edge '
         '(perpendicular to the preferred motion). Door view. 0° = bar lies along the RF long axis, 90° = across it.\n'
         'Dashed line: chance (25% per bin). Lines: 95% confidence interval. p: one-sided shuffle test of the aligned bin (bar orientations re-assigned among the cells of that group, 10,000×).',
         fontsize=8, color=MUTED, va='bottom')
fig.subplots_adjust(top=0.76)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
for name, g, _ in GROUPS: print(name, len(g), np.histogram(g.delta, BINS)[0])
