import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import axes_rf_bar_row as A

m = A.load(sys.argv[1]); out = sys.argv[2]
r = m[m.reliable].reset_index(drop=True)
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
COL = {'round': '#9aa4ae', 'elongated': '#2f6fb3'}
recs = ['14_05_2026', '15_05_2026', '22_07_2026', '14_08_2026']  # 15/10: only 2 reliable cells


def c2(t, a, b): return np.cos(2 * np.deg2rad(A.diff(t[a], t[b])))


def pstr(p): return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'


def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1.0], hspace=0.5, wspace=0.28))
bins = np.arange(0, 91, 15)
for row, (a, b, name) in enumerate(A.COMPARISONS):
    ax = axes[row, 0]
    lines = []
    for i, shape in enumerate(('round', 'elongated')):
        t = r[r['shape'] == shape].reset_index(drop=True)
        h, _ = np.histogram(A.diff(t[a], t[b]), bins); f = h / max(h.sum(), 1)
        ax.bar(bins[:-1] + 7.5 + (i - 0.5) * 5.6, f, width=5.2, color=COL[shape], label=f'{shape} (n={len(t)})')
        v = c2(t, a, b); s, p_shuf = A.score(t, a, b)
        lines.append(f'{shape}: score {v.mean():+.2f}  vs 0 {pstr(ttest_1samp(v, 0).pvalue)}  '
                     f'within-rec. shuffle {pstr(p_shuf)}')
    ax.axhline(1 / 6, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(bins); ax.set_xlim(0, 90); ax.set_ylim(0, 0.5)
    ax.set_xlabel('Angle between the two axes (0 = parallel, 90 = perpendicular)', fontsize=8.5, color=INK)
    ax.set_ylabel('Fraction of cells', fontsize=9, color=INK)
    ax.set_title(f'{"ABC"[row]}  {name}', loc='left', fontsize=11, color=INK, x=-0.12)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    style(ax)
    axes[row, 2].axis('off')
    axes[row, 2].text(0, 0.95, 'Reliable bar responders (bar QI > 0.3)\n\n' + '\n'.join(lines).replace('  vs', '\n   vs').replace('  within', '\n   within'),
                      va='top', fontsize=8.5, color=INK, linespacing=1.4, transform=axes[row, 2].transAxes)

    ax = axes[row, 1]
    for i, shape in enumerate(('round', 'elongated')):
        for j, rec in enumerate(recs):
            t = r[(r['shape'] == shape) & (r.rec == rec)]
            if len(t) < 3: continue
            v = c2(t, a, b); se = v.std(ddof=1) / np.sqrt(len(v))
            x = j + (i - 0.5) * 0.28
            ax.errorbar(x, v.mean(), yerr=1.96 * se, fmt='o', color=COL[shape], ms=4 + np.sqrt(len(t)), lw=1.6)
            ax.text(x, -1.12, str(len(t)), ha='center', fontsize=7, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(recs))); ax.set_xticklabels([x[:5].replace('_', '/') for x in recs], fontsize=8.5)
    ax.set_ylim(-1.2, 1.05); ax.set_xlim(-0.6, len(recs) - 0.4)
    ax.set_ylabel('Score: mean cos 2(angle)\n+1 parallel, -1 perpendicular', fontsize=8.5, color=INK)
    ax.set_title('Each recording (mean ± 95% CI; numbers = n; 15/10 left out: 2 reliable cells)', loc='left', fontsize=9, color=MUTED)
    style(ax)

fig.text(0.06, 0.955, 'RF long axis, preferred moving-bar axis and radiating cone row: round vs elongated RFs',
         fontsize=14, color=INK)
fig.text(0.06, 0.935,
         f'Recordings with cone-image alignment and moving bar (07/05 and 07/08 not extracted in polarspike). Cells: RF quality > 10, RF centre inside the '
         f'displayed stimulus and > 50 µm from its edge, bar QI > {A.QI}. Round: elongation < {A.ELONG}; elongated: ≥ {A.ELONG}.\n'
         'All axes in the aligned cone-image frame (door view). Bar axis = axis the bar moves along (its edge is perpendicular to it). Cone row = line from '
         'the clicked optic nerve head to the RF centre.\n'
         '"vs 0" asks whether the angles lean parallel or perpendicular at all; "within-rec. shuffle" asks whether it holds CELL BY CELL, beyond a '
         'direction shared by the whole recording (second axis shuffled among cells of the same recording).',
         fontsize=8.5, color=MUTED, va='top', linespacing=1.4)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('saved', out)
