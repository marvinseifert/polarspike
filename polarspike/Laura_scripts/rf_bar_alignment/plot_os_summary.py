"""Summary: does the RF long axis predict the preferred bar orientation (bar long edge = motion axis + 90)?
Final clustering cells (9 recordings, 12 + 32 px), orientation-selective (OSI > OSI_MIN).

Statistics (axes repeat every 180 deg, so all angles are doubled first):
  r   circular correlation between RF axis and bar orientation (Jammalamadaka & SenGupta 2001,
      on doubled angles); r^2 = the circular analogue of variance explained.
  score  mean cos 2(angle between them): +1 all aligned, 0 unrelated, -1 all perpendicular.
  band   fraction of cells within 22.5 deg of aligned (25 % expected if unrelated).
  p   shuffle test: bar orientations randomly re-assigned to the cells N times; p = fraction of
      shuffles giving a value at least as large as the real one. 'within recording' shuffles only
      among cells of the same recording (stricter: a direction shared by a whole retina can't count).
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OSI_MIN, ELONG_MIN, N = 0.10, 1.05, 10000
o = pd.read_csv(sys.argv[1]); out = sys.argv[2]
o['bar_orient'] = (o.motion_axis + 90) % 180
o['delta'] = np.abs((o.rf_axis - o.bar_orient + 90) % 180 - 90)
rng = np.random.default_rng(0)
INK, MUTED, GRID = '#1f2328', '#6e7781', '#e6e8eb'
C_SLIGHT, C_ELONG, C_ROUND = '#5b8fd0', '#16457a', '#9aa4ae'


def circ_r(a, b):
    a, b = np.deg2rad(2 * np.asarray(a)), np.deg2rad(2 * np.asarray(b))
    ma, mb = np.angle(np.exp(1j * a).mean()), np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - ma), np.sin(b - mb)
    return np.sum(sa * sb) / np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))


def score(a, b): return np.mean(np.cos(2 * np.deg2rad(a - b)))
def band(a, b): return np.mean(np.abs((a - b + 90) % 180 - 90) <= 22.5)


def perm_p(t, stat, within=False):
    a, b = t.rf_axis.values, t.bar_orient.values
    real = stat(a, b); null = np.empty(N)
    groups = list(t.reset_index(drop=True).groupby('rec').indices.values())
    for i in range(N):
        if within:
            bb = b.copy()
            for g in groups: bb[g] = rng.permutation(b[g])
        else:
            bb = rng.permutation(b)
        null[i] = stat(a, bb)
    return real, (np.sum(null >= real) + 1) / (N + 1)  # one-sided: aligned


def pstr(p): return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'


t = o[(o.osi > OSI_MIN) & (o.elong >= ELONG_MIN)].reset_index(drop=True)
r, p_r = perm_p(t, circ_r); _, p_rw = perm_p(t, circ_r, within=True)
s, p_s = perm_p(t, score); _, p_sw = perm_p(t, score, within=True)
bf, p_b = perm_p(t, band)

fig = plt.figure(figsize=(15, 5.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.8], wspace=0.38)
a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))

# A scatter
for dy in (-180, 0, 180):
    a1.fill_between([0, 180], [dy - 22.5, 180 + dy - 22.5], [dy + 22.5, 180 + dy + 22.5], color='#dfe9f6', lw=0)
a1.plot([0, 180], [0, 180], color=MUTED, lw=0.8, ls=(0, (3, 3)))
for name, g, c in ((f'slightly elongated ({ELONG_MIN}-1.2)', t[t.elong < 1.2], C_SLIGHT), ('elongated (≥ 1.2)', t[t.elong >= 1.2], C_ELONG)):
    a1.scatter(g.rf_axis, g.bar_orient, s=12 + 70 * np.clip(g.osi, 0, 0.6), color=c, edgecolor='white', lw=0.5, label=f'{name}, n={len(g)}')
a1.set_xlim(0, 180); a1.set_ylim(0, 180); a1.set_aspect('equal')
a1.set_xticks(range(0, 181, 45)); a1.set_yticks(range(0, 181, 45))
a1.set_xlabel('RF long axis (°, 0 = up-down, 90 = left-right)', fontsize=9, color=INK)
a1.set_ylabel('Preferred bar orientation (°, long edge of the bar)', fontsize=9, color=INK)
a1.set_title(f'A  Each cell (OSI > {OSI_MIN}, elongation ≥ {ELONG_MIN}; n = {len(t)})', loc='left', fontsize=10.5, color=INK, pad=32)
a1.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0, -0.13), ncol=2)
a1.text(0, 1.03, f'Circular correlation r = {r:+.2f} (r² = {r * r:.3f}), {pstr(p_r)}\nwithin recording {pstr(p_rw)}',
        transform=a1.transAxes, fontsize=8.5, color=INK)

# B histogram of the difference
bins = np.arange(0, 91, 7.5)
a2.hist(t.delta, bins=bins, weights=np.full(len(t), 1 / len(t)), color=C_SLIGHT, edgecolor='white', lw=1)
a2.axhline(1 / 12, color=MUTED, lw=0.9, ls=(0, (3, 3)))
a2.text(89, 1 / 12 + 0.003, 'no relation', ha='right', fontsize=7.5, color=MUTED)
a2.axvspan(0, 22.5, color='#dfe9f6', zorder=0)
a2.set_xlim(0, 90); a2.set_xticks(range(0, 91, 15)); a2.set_ylim(0, 0.17)
a2.set_xlabel('Angle between RF long axis and preferred bar (°)\n0 = bar edge along the RF, 90 = across it', fontsize=9, color=INK)
a2.set_ylabel('Fraction of cells', fontsize=9, color=INK)
a2.set_title('B  How far apart the two axes are', loc='left', fontsize=10.5, color=INK)
a2.text(0.98, 0.97, f'Within 22.5° (shaded): {100 * bf:.0f} % of cells\n(25 % if unrelated), {pstr(p_b)}\n'
        f'Alignment score {s:+.2f}, {pstr(p_s)}\nwithin recording {pstr(p_sw)}',
        transform=a2.transAxes, ha='right', va='top', fontsize=8.5, color=INK, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=GRID))

# C comparison by elongation
rows = [('round\n< 1.05', o[(o.osi > OSI_MIN) & (o.elong < 1.05)], C_ROUND),
        ('slightly\n1.05-1.2', o[(o.osi > OSI_MIN) & (o.elong >= 1.05) & (o.elong < 1.2)], C_SLIGHT),
        ('elongated\n≥ 1.2', o[(o.osi > OSI_MIN) & (o.elong >= 1.2)], C_ELONG)]
for j, (name, g, c) in enumerate(rows):
    g = g.reset_index(drop=True)
    rr, pp = perm_p(g, score)
    c2 = np.cos(2 * np.deg2rad(g.rf_axis.values - g.bar_orient.values))
    boot = rng.choice(c2, (4000, len(c2))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    a3.bar(j, rr, width=0.6, color=c); a3.plot([j, j], [lo, hi], color=INK, lw=1.3)
    a3.text(j, 0.02, f'n={len(g)}\n{pstr(pp)}', ha='center', va='bottom', fontsize=7.5, color=MUTED, transform=a3.get_xaxis_transform())
a3.axhline(0, color=MUTED, lw=0.9)
a3.set_xticks(range(3)); a3.set_xticklabels([x[0] for x in rows], fontsize=8.5)
a3.set_ylim(-0.6, 0.6)
a3.set_ylabel('Alignment score (mean cos 2 angle, 95% CI)', fontsize=9, color=INK)
a3.set_title(f'C  By elongation (OSI > {OSI_MIN})', loc='left', fontsize=10.5, color=INK)

for ax in (a1, a2, a3):
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
for ax in (a2, a3): ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.suptitle('Orientation-selective RGCs prefer bars whose long edge lies along their RF long axis (weak but consistent)',
             x=0.07, y=1.0, ha='left', fontsize=13, color=INK)
fig.text(0.07, 0.935, 'Final clustering cells, 9 recordings (12 px at 5 µm/px and 32 px). Door view. Dot size = OSI. '
         'Shaded: within ±22.5° of aligned (wraps at 0/180°). p: one-sided shuffle test (bar orientations re-assigned to cells 10,000×).',
         fontsize=8, color=MUTED)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'n={len(t)} r={r:.3f} r2={r*r:.3f} p={p_r} pw={p_rw} score={s:.3f} p={p_s} pw={p_sw} band={bf:.3f} p={p_b}')
