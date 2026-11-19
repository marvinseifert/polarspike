"""RF long axis vs preferred bar orientation (bar long edge = motion axis + 90), final clustering
cells (12 + 32 px), across OSI thresholds and elongation groupings.

Figure 1 (scatter grid): rows = OSI threshold, columns = round / slightly elongated / elongated.
Figure 2 (robustness): alignment score vs OSI threshold, one panel per elongation grouping.
Score = mean cos 2(angle between the axes): +1 bar edge along the RF axis, -1 across, 0 none.
p = bar orientations shuffled among the cells in that group (two-sided).
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

o = pd.read_csv(sys.argv[1]); out1, out2 = sys.argv[2], sys.argv[3]
o['bar_orient'] = (o.motion_axis + 90) % 180
o['delta'] = np.abs((o.rf_axis - o.bar_orient + 90) % 180 - 90)
rng = np.random.default_rng(0)
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
GC = ['#8c959f', '#5b8fd0', '#16457a']
OSIS = [0.05, 0.10, 0.15, 0.20]


def score_p(t, n=3000):
    if len(t) < 3: return np.nan, np.nan
    rf, bo = t.rf_axis.values, t.bar_orient.values
    f = lambda b: np.mean(np.cos(2 * np.deg2rad(np.abs((rf - b + 90) % 180 - 90))))
    real = f(bo); null = np.array([f(rng.permutation(bo)) for _ in range(n)])
    return real, (np.sum(np.abs(null - null.mean()) >= abs(real - null.mean())) + 1) / (n + 1)


def pstr(p): return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'


def groups(t, lo, hi):
    return [(f'round (< {lo})', t[t.elong < lo]), (f'slightly elongated ({lo}-{hi})', t[(t.elong >= lo) & (t.elong < hi)]),
            (f'elongated (≥ {hi})', t[t.elong >= hi])]


# ---------- Figure 1: scatter grid (round < 1.05, slight 1.05-1.2, elongated >= 1.2)
fig, axes = plt.subplots(len(OSIS), 3, figsize=(12.5, 4.1 * len(OSIS)), gridspec_kw=dict(hspace=0.42, wspace=0.18))
for r, th in enumerate(OSIS):
    for c, (name, g) in enumerate(groups(o[o.osi > th], 1.05, 1.2)):
        ax = axes[r, c]
        for dy in (-180, 0, 180):
            ax.fill_between([0, 180], [dy - 22.5, 180 + dy - 22.5], [dy + 22.5, 180 + dy + 22.5], color='#dfe9f6', lw=0)
        ax.plot([0, 180], [0, 180], color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.scatter(g.rf_axis, g.bar_orient, s=10 + 60 * np.clip(g.osi, 0, 0.6), color=GC[c], edgecolor='white', lw=0.5, alpha=0.9)
        s, p = score_p(g)
        ax.set_xlim(0, 180); ax.set_ylim(0, 180); ax.set_aspect('equal')
        ax.set_xticks(range(0, 181, 45)); ax.set_yticks(range(0, 181, 45))
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)
        ax.set_title(f'{name}\nOSI > {th:.2f}:  n={len(g)}, score {s:+.2f}, {pstr(p)}', loc='left', fontsize=8.5, color=INK)
        if c == 0: ax.set_ylabel('Preferred bar orientation (°)', fontsize=8.5, color=INK)
        if r == len(OSIS) - 1: ax.set_xlabel('RF long axis (°)', fontsize=8.5, color=INK)
fig.suptitle('RF long axis vs preferred bar orientation: orientation-selective cells at four OSI thresholds',
             x=0.07, y=0.985, ha='left', fontsize=13, color=INK)
fig.text(0.07, 0.968, 'Final clustering cells, 9 recordings, 12 + 32 px. Diagonal = bar long edge parallel to the RF long axis '
         '(shaded ±22.5°, wraps at 0/180°). Dot size = OSI. Door view.', fontsize=8, color=MUTED)
fig.subplots_adjust(top=0.93)
fig.savefig(out1, dpi=130, bbox_inches='tight', facecolor='white')

# ---------- Figure 2: robustness over thresholds
SCHEMES = [(1.05, 1.2), (1.1, 1.2), (1.05, 1.3), (1.1, 1.3)]
THS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
fig, axes = plt.subplots(1, len(SCHEMES), figsize=(17, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
rows = []
for ax, (lo, hi) in zip(axes, SCHEMES):
    for c in range(3):
        xs, ys, ps, ns = [], [], [], []
        for th in THS:
            name, g = groups(o[o.osi > th], lo, hi)[c]
            s, p = score_p(g, 2000)
            xs.append(th); ys.append(s); ps.append(p); ns.append(len(g))
            rows.append(dict(round_below=lo, elongated_from=hi, osi_above=th, group=name, n=len(g), score=s, p=p))
        ax.plot(xs, ys, color=GC[c], lw=2, label=name)
        for x, y, p, n in zip(xs, ys, ps, ns):
            if np.isnan(y): continue
            ax.plot(x, y, 'o', ms=7 if p < 0.05 else 5, color=GC[c] if p < 0.05 else 'white', mec=GC[c], mew=1.5)
            ax.text(x, y + 0.035, str(n), ha='center', fontsize=6.5, color=GC[c])
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.set_xticks(THS); ax.set_ylim(-0.6, 0.8)
    ax.set_title(f'round < {lo} · elongated ≥ {hi}', loc='left', fontsize=10, color=INK)
    ax.set_xlabel('Only cells with OSI above', fontsize=8.5, color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
axes[0].set_ylabel('Alignment score (mean cos 2 angle)\n+1 bar edge along RF axis', fontsize=8.5, color=INK)
fig.suptitle('How robust is it? Alignment score across OSI thresholds and elongation cut-offs',
             x=0.07, y=1.02, ha='left', fontsize=13, color=INK)
fig.text(0.07, 0.955, 'Filled dots: p < 0.05 (bar orientations shuffled among the cells of that group); open: not significant. '
         'Small numbers = cells. OSI 0 = all cells.', fontsize=8, color=MUTED)
fig.savefig(out2, dpi=140, bbox_inches='tight', facecolor='white')
pd.DataFrame(rows).to_csv(out2.replace('.png', '.csv'), index=False)
print(pd.DataFrame(rows).round(3).to_string())
