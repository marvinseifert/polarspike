"""RF long axis vs preferred moving-bar axis vs radiating cone row, round vs elongated cells.

Inputs: aligned_osi_elong.csv (polarspike: OSI, bar QI, bar axis) and the rf_analysis
cone-row table (RF axis, cone-row axis, edge distance; projected frame, 0 deg = right,
increasing clockwise on screen). Bar axis (door compass, 0 = up, clockwise) -> projected: -90.
Score per comparison: mean cos(2 x difference): +1 = parallel, -1 = perpendicular, 0 = unrelated.
p: the second axis shuffled among cells WITHIN each recording (5000 x), two-sided. This keeps
each recording's own spread of axes, so a direction shared by all cells cannot make p small.
"""
import sys
import numpy as np, pandas as pd

ROW_CSV = '/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction/rf_vs_cone_row_angle_all_recordings_cells.csv'
QI, ELONG, NPERM = 0.3, 1.2, 5000
rng = np.random.default_rng(0)


def load(bar_csv):
    a = pd.read_csv(bar_csv); b = pd.read_csv(ROW_CSV)
    m = a.merge(b, left_on=['rec', 'cell'], right_on=['recording', 'cell_index'])
    m['bar_axis_proj'] = (m.bar_axis_deg - 90) % 180
    m = m[m.inside_displayed_stimulus & ~m.near_edge & ~m.at_edge & m.osi.notna()]
    m['reliable'] = m.bar_qi > QI
    m['shape'] = np.where(m.elongation >= ELONG, 'elongated', 'round')
    return m


def diff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 90) % 180 - 90)


def score(t, a, b):
    if len(t) < 3: return np.nan, np.nan
    real = np.mean(np.cos(2 * np.deg2rad(diff(t[a], t[b]))))
    groups = [g.index.values for _, g in t.groupby('rec')]
    bb = t[b].copy(); null = np.empty(NPERM)
    for i in range(NPERM):
        for idx in groups: bb.loc[idx] = rng.permutation(t.loc[idx, b].values)
        null[i] = np.mean(np.cos(2 * np.deg2rad(diff(t[a], bb))))
    p = (np.sum(np.abs(null - null.mean()) >= abs(real - null.mean())) + 1) / (NPERM + 1)
    return real, p


COMPARISONS = [('rf_axis_projected_deg', 'bar_axis_proj', 'RF axis vs bar motion axis'),
               ('bar_axis_proj', 'cone_row_axis_projected_deg', 'Bar motion axis vs radiating cone row'),
               ('rf_axis_projected_deg', 'cone_row_axis_projected_deg', 'RF axis vs radiating cone row')]

if __name__ == '__main__':
    m = load(sys.argv[1])
    print('cells', len(m), 'reliable', int(m.reliable.sum()))
    for a, b, name in COMPARISONS:
        print('\n' + name)
        for sel_name, sel in (('reliable', m[m.reliable]), ('all', m)):
            for shape in ('round', 'elongated'):
                t = sel[sel['shape'] == shape].reset_index(drop=True)
                s, p = score(t, a, b)
                print(f'  {sel_name:8s} {shape:9s} n={len(t):3d} score={s:+.2f} p={p:.3f}')
            t = sel[sel.elongation >= 1.4].reset_index(drop=True); s, p = score(t, a, b)
            print(f'  {sel_name:8s} elong>=1.4 n={len(t):3d} score={s:+.2f} p={p:.3f}')
