"""Orientation-selective cells (OSI > 0.15) of the final clustering, 12 px and 32 px together:
does RF elongation predict the preferred bar orientation (bar long edge = motion axis + 90)?
A: alignment score per elongation bin (bars, 95% bootstrap CI).
B: histogram of the angle between RF long axis and preferred bar, for three elongation groups.
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OSI = 0.15
o = pd.read_csv(sys.argv[1]); out = sys.argv[2]
o['bar_orient'] = (o.motion_axis + 90) % 180
o['delta'] = np.abs((o.rf_axis - o.bar_orient + 90) % 180 - 90)
o['c2'] = np.cos(2 * np.deg2rad(o.delta))
t = o[o.osi > OSI].reset_index(drop=True)
rng = np.random.default_rng(0)
INK, MUTED, GRID, BLUE = '#1f2328', '#6e7781', '#e6e8eb', '#2f6fb3'
EDGES = [1.0, 1.05, 1.2, 3.5]
LABELS = ['round\n1.0-1.05', 'slightly elongated\n1.05-1.2', 'elongated\n> 1.2']


def shuffle_p(rf, bo, stat, n=10000):
    real = stat(rf, bo)
    null = np.array([stat(rf, rng.permutation(bo)) for _ in range(n)])
    return real, (np.sum(np.abs(null - null.mean()) >= abs(real - null.mean())) + 1) / (n + 1)


def c2(rf, bo): return np.cos(2 * np.deg2rad(np.abs((rf - bo + 90) % 180 - 90)))
def pstr(p): return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'
def style(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(wspace=0.25))

# A: score per elongation bin
cut = pd.cut(t.elong, EDGES, labels=LABELS, right=False)
for j, lab in enumerate(LABELS):
    v = t.c2[cut == lab].values
    m = rng.choice(v, (4000, len(v))).mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    a1.bar(j, v.mean(), width=0.6, color=['#b8bfc7', '#7fa7d6', '#16457a'][j])
    a1.plot([j, j], [lo, hi], color=INK, lw=1.3)
    a1.text(j, -0.95, f'n={len(v)}', ha='center', fontsize=8, color=MUTED)
rho, p = shuffle_p(t.rf_axis.values, t.bar_orient.values, lambda rf, bo: spearmanr(t.elong, c2(rf, bo))[0])
a1.axhline(0, color=MUTED, lw=0.9)
a1.set_xticks(range(len(LABELS))); a1.set_xticklabels(LABELS, fontsize=9)
a1.set_ylim(-1, 1)
a1.set_xlabel('RF elongation (long / short axis), round → elongated', fontsize=9, color=INK)
a1.set_ylabel('Alignment score: mean cos 2(angle)\n+1 bar edge along RF long axis · −1 across', fontsize=9, color=INK)
a1.set_title('A  Alignment score by elongation group', loc='left', fontsize=11, color=INK)
a1.text(0.02, 0.97, f'Rises with elongation? rho = {rho:+.2f}, {pstr(p)}\nLines = 95% bootstrap CI',
        transform=a1.transAxes, va='top', fontsize=8.5, color=INK)
style(a1)

# B: angle histograms for three elongation groups
groups = [('round (1.0-1.05)', t[t.elong < 1.05], '#b8bfc7'),
          ('slightly elongated (1.05-1.2)', t[(t.elong >= 1.05) & (t.elong < 1.2)], '#7fa7d6'),
          ('elongated (≥ 1.2)', t[t.elong >= 1.2], '#16457a')]
bins = np.arange(0, 91, 15)
for i, (name, g, c) in enumerate(groups):
    h, _ = np.histogram(g.delta, bins); f = h / h.sum()
    s, pp = shuffle_p(g.rf_axis.values, g.bar_orient.values, lambda rf, bo: c2(rf, bo).mean())
    a2.bar(bins[:-1] + 7.5 + (i - 1) * 4.4, f, width=4.1, color=c, label=f'{name}, n={len(g)}: score {s:+.2f}, {pstr(pp)}')
a2.axhline(1 / 6, color=MUTED, lw=0.8, ls=(0, (3, 3)))
a2.text(89, 1 / 6 + 0.006, 'no relation', ha='right', fontsize=7.5, color=MUTED)
a2.set_xticks(bins); a2.set_xlim(0, 90); a2.set_ylim(0, 0.42)
a2.set_xlabel('Angle between RF long axis and preferred bar (°)\n0 = bar edge along the RF, 90 = across it', fontsize=9, color=INK)
a2.set_ylabel('Fraction of cells', fontsize=9, color=INK)
a2.set_title('B  Angle distribution by elongation', loc='left', fontsize=11, color=INK)
a2.legend(frameon=False, fontsize=8, loc='upper right')
style(a2)

fig.suptitle(f'Orientation-selective cells (OSI > {OSI}, n = {len(t)}): does RF elongation predict the preferred bar orientation?',
             x=0.07, y=1.0, ha='left', fontsize=13, color=INK)
fig.text(0.07, 0.925, 'Final clustering cells, 9 recordings (12 px at 5 µm/px and 32 px together). Preferred bar = long edge of the bar, '
         'perpendicular to its preferred motion axis. Door view. p: bar orientations shuffled among the cells.',
         fontsize=8, color=MUTED)
fig.subplots_adjust(top=0.85)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
for name, g, _ in groups: print(name, len(g), round(g.c2.mean(), 2))
print('rho', rho, p)
