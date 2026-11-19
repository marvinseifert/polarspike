import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
QI = 0.3
d = d[d.osi.notna() & ~d.at_edge]
d['good'] = d.bar_qi > QI
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
ALL_C, GOOD_C = '#b8bfc7', '#2f6fb3'
recs = ['14_05_2026', '15_05_2026', '22_07_2026', '14_08_2026', '15_10_2026']
px = {r: '12 px' if r.endswith('05_2026') else '32 px' for r in recs}


def pstr(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.2f}' if p >= 0.01 else f'p={p:.3f}'


def stat(t):
    if len(t) < 5: return f'n={len(t)}'
    r, p = spearmanr(t.elongation, t.osi)
    return f'n={len(t)}, rho={r:+.2f}, {pstr(p)}'


def panel(ax, t, title):
    a, g = t[~t.good], t[t.good]
    ax.scatter(a.elongation, a.osi, s=16, color=ALL_C, lw=0, label=f'weak bar response (QI <= {QI})')
    ax.scatter(g.elongation, g.osi, s=22, color=GOOD_C, edgecolor='white', lw=0.6, label=f'reliable bar response (QI > {QI})')
    ax.set_xlim(0.95, 3.25); ax.set_ylim(-0.02, 1.02)
    ax.axvline(1.2, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.text(0.98, 0.97, 'all: ' + stat(t) + '\nreliable: ' + stat(g), transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRID))


fig, axes = plt.subplots(2, 4, figsize=(16, 8), gridspec_kw=dict(hspace=0.38, wspace=0.22))
panel(axes[0, 0], d, 'All 5 recordings pooled')
for ax, r in zip(axes.flat[1:], recs):
    panel(ax, d[d.rec == r], f'{r[:5].replace("_", "/")}  ({px[r]} noise)')

# binned summary: OSI of reliable responders, round vs elongated
ax = axes.flat[6]
bins = [1, 1.1, 1.2, 1.4, 1.7, 3.5]; labels = ['1.0-1.1', '1.1-1.2', '1.2-1.4', '1.4-1.7', '>1.7']
for i, (name, t, c) in enumerate((('all cells', d, ALL_C), ('reliable', d[d.good], GOOD_C))):
    cut = pd.cut(t.elongation, bins)
    med = t.groupby(cut, observed=False).osi.median(); n = t.groupby(cut, observed=False).osi.size()
    q1 = t.groupby(cut, observed=False).osi.quantile(.25); q3 = t.groupby(cut, observed=False).osi.quantile(.75)
    x = np.arange(len(labels)) + (i - 0.5) * 0.3
    ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt='o', color=c if i else '#8c959f', ms=6, capsize=0, lw=2, label=name)
    for xi, ni in zip(x, n): ax.text(xi, -0.08, f'{ni}', ha='center', fontsize=7, color=MUTED)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(-0.12, 0.8); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_title('Median OSI by elongation (line = middle 50%)', loc='left', fontsize=10, color=INK)
ax.set_xlabel('RF elongation (long / short axis)', fontsize=9, color=INK)
ax.text(-0.4, -0.08, 'n:', fontsize=7, color=MUTED, ha='right')
ax.legend(frameon=False, fontsize=8, loc='upper left')

ax = axes.flat[7]; ax.axis('off')
r_all, p_all = spearmanr(d.elongation, d.osi)
r_sp, p_sp = spearmanr(d.bar_spikes, d.osi)
txt = (f'Cells: RF quality > 10, RF not cut off by the stimulus edge,\nmoving bar shown (n={len(d)}).\n\n'
       f'Blue: moving-bar response repeats reliably\n(bar QI > {QI}, Baden et al. 2016 method).\n\n'
       f'Dashed line: elongation 1.2 (round | elongated).\n\n'
       f'Weak responders have inflated OSI: OSI vs bar\nspike count rho={r_sp:+.2f} ({pstr(p_sp)}).\n'
       f'Their slight negative trend with elongation\ndisappears in the reliable cells.\n\n'
       'OSI and elongation do not depend on the\nstimulus/door alignment (rotating or mirroring\nthe 8 bar directions leaves OSI unchanged).')
ax.text(0, 0.80, txt, va='top', fontsize=9, color=INK, linespacing=1.35)
h, l = axes[0, 0].get_legend_handles_labels()
ax.legend(h, l, loc='upper left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8.5)

for a in axes[1, :3]: a.set_xlabel('RF elongation (long / short axis)', fontsize=9, color=INK)
for a in axes[:, 0]: a.set_ylabel('Orientation selectivity (OSI, moving bar)', fontsize=9, color=INK)
fig.suptitle('Moving-bar orientation selectivity vs RF elongation: recordings with cone-image alignment',
             x=0.06, ha='left', fontsize=13, color=INK)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('saved', out)
