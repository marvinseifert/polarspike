"""Alignment of RF long axis and preferred bar orientation (bar long edge = motion axis + 90) in 0.05
elongation steps, orientation-selective cells (OSI > OSI_MIN) of the final clustering (12 + 32 px).
Bars: alignment score = mean cos 2(angle), +1 aligned, 0 unrelated; lines: 95% bootstrap CI.
p per bin: bar orientations shuffled among the cells of that bin (one-sided, aligned).
Trend: Spearman rho between elongation and each cell's cos 2(angle); p by shuffling bar orientations.
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OSI_MIN, N = 0.10, 10000
o = pd.read_csv(sys.argv[1]); out = sys.argv[2]
o['bar_orient'] = (o.motion_axis + 90) % 180
t = o[o.osi > OSI_MIN].reset_index(drop=True)
t['c2'] = np.cos(2 * np.deg2rad(t.rf_axis - t.bar_orient))
rng = np.random.default_rng(0)
EDGES = list(np.round(np.arange(1.0, 1.401, 0.05), 2)) + [9]
LABELS = [f'{a:.2f}-{b:.2f}' for a, b in zip(EDGES[:-2], EDGES[1:-1])] + ['≥ 1.40']
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
cmap = plt.get_cmap('Blues')


def pstr(p): return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'


cut = pd.cut(t.elong, EDGES, labels=LABELS, right=False)
fig, ax = plt.subplots(figsize=(11, 5.6))
for j, lab in enumerate(LABELS):
    g = t[cut == lab]
    a, b = g.rf_axis.values, g.bar_orient.values
    real = g.c2.mean()
    null = np.array([np.mean(np.cos(2 * np.deg2rad(a - rng.permutation(b)))) for _ in range(N)])
    p = (np.sum(null >= real) + 1) / (N + 1)
    lo, hi = np.percentile(rng.choice(g.c2.values, (4000, len(g))).mean(1), [2.5, 97.5])
    ax.bar(j, real, width=0.7, color=cmap(0.35 + 0.6 * j / (len(LABELS) - 1)))
    ax.plot([j, j], [lo, hi], color=INK, lw=1.2)
    ax.text(j, 0.02, f'n={len(g)}\n{pstr(p)}', ha='center', va='bottom', fontsize=7.5,
            color=INK if p < 0.05 else MUTED, transform=ax.get_xaxis_transform())

rho = spearmanr(t.elong, t.c2)[0]
a, b = t.rf_axis.values, t.bar_orient.values
null = np.array([spearmanr(t.elong, np.cos(2 * np.deg2rad(a - rng.permutation(b))))[0] for _ in range(N)])
p_tr = (np.sum(np.abs(null) >= abs(rho)) + 1) / (N + 1)
ax.axhline(0, color=MUTED, lw=0.9)
ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=8.5)
ax.set_ylim(-0.85, 1.05)
ax.set_xlabel('RF elongation (long / short axis)', fontsize=10, color=INK)
ax.set_ylabel('Alignment score', fontsize=10, color=INK)
ax.text(-0.075, 0.97, '↑ bar along RF', transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=MUTED)
ax.text(-0.075, 0.03, '↓ bar across RF', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color=MUTED)
ax.text(0.01, 0.97, f'Does alignment rise with elongation? Spearman rho = {rho:+.2f}, {pstr(p_tr)}\n'
        'Score = mean cos 2(angle between RF long axis and preferred bar): +1 aligned, 0 unrelated, −1 across.\n'
        'Lines = 95% bootstrap CI. Dark labels: p < 0.05 within that bin.',
        transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=8); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.suptitle(f'Orientation-selective cells (OSI > {OSI_MIN}, n = {len(t)}): RF axis vs preferred bar, in 0.05 elongation steps',
             x=0.07, y=1.0, ha='left', fontsize=12.5, color=INK)
fig.text(0.07, 0.935, 'Final clustering cells, 9 recordings (12 px at 5 µm/px and 32 px). Preferred bar = its long edge '
         '(perpendicular to the preferred motion). Door view. p: one-sided shuffle test, 10,000 shuffles.', fontsize=8, color=MUTED)
fig.subplots_adjust(top=0.86)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('rho', rho, p_tr)
