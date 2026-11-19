"""Elongated AND orientation-selective cells: is the preferred bar orientation parallel to the RF long axis?

Input: rf_vs_bar_axis.csv (cells of the final clustering 90faade5: rec, elong, osi, rf_axis = RF long
axis, motion_axis = preferred bar motion axis, both 0-180 deg in the door-view polar-plot frame).
Preferred bar orientation = the bar's edge = motion axis + 90.
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ELONG, OSI, OSI_STRICT, NPERM = 1.2, 0.15, 0.2, 10000
o = pd.read_csv(sys.argv[1]); out = sys.argv[2]
o['bar_orient'] = (o.motion_axis + 90) % 180
o['delta'] = np.abs((o.rf_axis - o.bar_orient + 90) % 180 - 90)  # 0 = RF along the preferred bar
rng = np.random.default_rng(0)


def test(t):
    """score = mean cos 2*delta (+1 aligned, -1 perpendicular). p vs no relation: bar orientations
    shuffled among the selected cells (10000x), two-sided."""
    real = np.mean(np.cos(2 * np.deg2rad(t.delta)))
    rf, bo = t.rf_axis.values, t.bar_orient.values
    null = np.array([np.mean(np.cos(2 * np.deg2rad(np.abs((rf - rng.permutation(bo) + 90) % 180 - 90))))
                     for _ in range(NPERM)])
    p = (np.sum(np.abs(null - null.mean()) >= abs(real - null.mean())) + 1) / (NPERM + 1)
    return real, p


def pstr(p): return 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'


sel = o[(o.elong > ELONG) & (o.osi > OSI)]
strict = o[(o.elong > ELONG) & (o.osi > OSI_STRICT)]
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
C = {'12px': '#2f6fb3', '32px': '#e8683f'}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.3))
# A: scatter, with wrap-around copies so the diagonal band is continuous
for dy in (-180, 0, 180):
    a1.fill_between([0, 180], [0 + dy - 22.5, 180 + dy - 22.5], [0 + dy + 22.5, 180 + dy + 22.5], color='#d9e6f5', lw=0)
a1.plot([0, 180], [0, 180], color=MUTED, lw=0.8, ls=(0, (3, 3)))
for px, g in sel.groupby('px'):
    big = g.osi > OSI_STRICT
    a1.scatter(g.rf_axis[~big], g.bar_orient[~big], s=26, color=C[px], alpha=0.55, lw=0)
    a1.scatter(g.rf_axis[big], g.bar_orient[big], s=46, color=C[px], edgecolor='white', lw=0.7, label=f'{px} noise')
a1.set_xlim(0, 180); a1.set_ylim(0, 180); a1.set_xticks(range(0, 181, 45)); a1.set_yticks(range(0, 181, 45))
a1.set_aspect('equal')
a1.set_xlabel('RF long axis (°, 0 = up-down, 90 = left-right)', fontsize=9, color=INK)
a1.set_ylabel('Preferred bar orientation (° , edge of the bar)', fontsize=9, color=INK)
a1.set_title('A  Each elongated, orientation-selective cell', loc='left', fontsize=11, color=INK)
a1.text(4, 172, 'shaded = within 22.5° of aligned\nlarge dots: OSI > %.1f' % OSI_STRICT, fontsize=7.5, color=MUTED, va='top')
a1.legend(frameon=False, fontsize=8, loc='lower right')

# B: histogram of the angle between the two
bins = np.arange(0, 91, 15)
groups = [(f'elongated + OSI > {OSI}', sel, '#2f6fb3'), (f'elongated + OSI > {OSI_STRICT}', strict, '#16457a'),
          ('all other cells', o.drop(sel.index), '#b8bfc7')]
for i, (name, t, c) in enumerate(groups):
    h, _ = np.histogram(t.delta, bins); f = h / h.sum()
    s, p = test(t)
    a2.bar(bins[:-1] + 7.5 + (i - 1) * 4.4, f, width=4.1, color=c, label=f'{name} (n={len(t)}): score {s:+.2f}, {pstr(p)}')
a2.axhline(1 / 6, color=MUTED, lw=0.8, ls=(0, (3, 3)))
a2.text(89, 1 / 6 + 0.006, 'no relation', ha='right', fontsize=7.5, color=MUTED)
a2.set_xticks(bins); a2.set_xlim(0, 90); a2.set_ylim(0, 0.42)
a2.set_xlabel('Angle between RF long axis and preferred bar (°)\n0 = bar lies along the RF, 90 = across it', fontsize=9, color=INK)
a2.set_ylabel('Fraction of cells', fontsize=9, color=INK)
a2.set_title('B  How far apart the two axes are', loc='left', fontsize=11, color=INK)
a2.legend(frameon=False, fontsize=8, loc='upper left')
for ax in (a1, a2):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.suptitle('Do orientation-selective elongated cells prefer bars parallel to their RF?', x=0.07, ha='left', fontsize=13, color=INK)
fig.text(0.07, 0.905, f'Cells of the final clustering (9 recordings, 12 px at 5 µm/px). Elongated: elongation > {ELONG}. '
         f'Orientation-selective: moving-bar OSI > {OSI} (top quarter of all cells). Door view. '
         'Score = mean cos 2(angle): +1 aligned, -1 perpendicular; p from shuffling bar orientations among the cells.',
         fontsize=8, color=MUTED, wrap=True)
fig.subplots_adjust(top=0.8)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
for name, t, _ in groups: print(name, len(t), test(t))
for px, g in sel.groupby('px'): print(px, len(g), test(g))
