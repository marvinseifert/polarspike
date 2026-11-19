"""RF long axis vs preferred bar orientation (bar edge = motion axis + 90) across the whole
range of elongation, from round to elongated. All cells of the final clustering (rf_vs_bar_axis.csv).
Score per cell = cos 2(angle): +1 bar edge along the RF long axis, -1 across it, 0 no relation.
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
rng = np.random.default_rng(0)
EDGES = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.45, 3.5]
LABELS = ['1.00-1.05', '1.05-1.10', '1.10-1.15', '1.15-1.20', '1.20-1.30', '1.30-1.45', '>1.45']
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'


def boot_ci(v, n=4000):
    if len(v) < 3: return np.nan, np.nan
    m = rng.choice(v, (n, len(v))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def trend_p(t, n=10000):
    """Does the score rise with elongation? Spearman rho(elongation, cos 2angle); p by shuffling
    the bar orientations among cells (two-sided)."""
    rho = spearmanr(t.elong, t.c2)[0]
    rf, bo, el = t.rf_axis.values, t.bar_orient.values, t.elong.values
    null = np.array([spearmanr(el, np.cos(2 * np.deg2rad(np.abs((rf - rng.permutation(bo) + 90) % 180 - 90))))[0]
                     for _ in range(n)])
    return rho, (np.sum(np.abs(null) >= abs(rho)) + 1) / (n + 1)


def pstr(p): return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'


fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, gridspec_kw=dict(wspace=0.08))
sets = [('All cells', o), (f'Orientation-selective cells (OSI > {OSI})', o[o.osi > OSI])]
for ax, (title, t) in zip(axes, sets):
    cut = pd.cut(t.elong, EDGES, labels=LABELS)
    lines = []
    for i, (px, c) in enumerate((('all', '#1f2328'), ('12px', '#2f6fb3'), ('32px', '#e8683f'))):
        tt = t if px == 'all' else t[t.px == px]
        cc = cut if px == 'all' else cut[t.px == px]
        x = np.arange(len(LABELS)) + (i - 1) * 0.22; done = False
        for j, lab in enumerate(LABELS):
            v = tt.c2[cc == lab].values
            if len(v) < 3: continue
            lo, hi = boot_ci(v)
            ax.plot([x[j], x[j]], [lo, hi], color=c, lw=2 if px == 'all' else 1.3, alpha=1 if px == 'all' else 0.75)
            ax.plot(x[j], v.mean(), 'o', color=c, ms=7 if px == 'all' else 5, mec='white', mew=0.7,
                    label=None if done else ('12 px + 32 px' if px == 'all' else px + ' noise')); done = True
            if px == 'all': ax.text(x[j] + 0.22, -0.93, str(len(v)), ha='center', fontsize=7.5, color=MUTED)
        rho, p = trend_p(tt)
        lines.append(f"{'all' if px == 'all' else px}: rho = {rho:+.2f}, {pstr(p)}")
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_xlabel('RF elongation (long / short axis), round → elongated', fontsize=9, color=INK)
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.text(0.99, 0.09, 'Does alignment rise with elongation?\n' + '\n'.join(lines), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=INK, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=GRID))
    ax.text(-0.45, -0.93, 'n:', fontsize=7.5, color=MUTED)
    ax.set_ylim(-1, 1.08); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
axes[0].set_ylabel('Alignment score: mean cos 2(angle)\n+1 bar edge along RF long axis · −1 across', fontsize=9, color=INK)
fig.suptitle('Does a little elongation already predict the preferred bar orientation?', x=0.07, ha='left', fontsize=13, color=INK)
fig.text(0.07, 0.905, 'Final clustering cells (9 recordings, 12 px at 5 µm/px). Angle between RF long axis and the preferred bar\'s long edge '
         '(motion axis + 90°), door view. Dots = mean, lines = 95% bootstrap CI (bins with ≥ 3 cells).',
         fontsize=8, color=MUTED)
fig.subplots_adjust(top=0.83)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(o.groupby(pd.cut(o.elong, EDGES, labels=LABELS), observed=False).c2.agg(['size', 'mean']).round(2))
