"""OSI / DSI vs RF elongation for the recordings that have cone-image alignment in rf_analysis.

Per cell (RF quality > 10 in the recording's only RF noise channel):
  elongation, RF angle, edge margin  - from retina_master.sqlite (v2 RF values)
  OSI, DSI                           - moving bar, same code as the clustering (door view, 100 ms bins)
  bar QI                             - repeat reliability of the moving-bar response (Baden et al. 2016)
Output: CSV, one row per cell.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

LS = Path('/home/laura/PycharmProjects/polarspike/polarspike/Laura_scripts')
sys.path.insert(0, str(LS))
import clustering_data_extraction_all_functions as storage
import clustering_master_functions as F
import clustering_master_main as M

RECS = ['14_05_2026', '15_05_2026', '22_07_2026', '14_08_2026', '15_10_2026']
OUT = Path(sys.argv[1])
p = M.config()['processing']


def bar_qi(db, rec, cell, index):
    raw = db.spikes(rec, cell, index)
    meta = db.stimulus_metadata(rec, index)
    rel = np.asarray(meta['trigger_fr_relative'], np.int64); logic = int(meta['stimulus_repeat_logic'])
    count = min(int(meta['nr_repeats']), (len(rel) - 1) // logic)
    absolute = int(meta['begin_fr']) + rel
    if not meta.get('inferred_final_trigger_count', 0):
        while count and absolute[count * logic] > int(meta['end_fr']): count -= 1
    grid = np.linspace(0, p['durations']['moving_bar'], p['samples'])
    traces = []
    for r in range(count):
        b, e = absolute[r * logic], absolute[(r + 1) * logic]
        ev = raw[(raw >= b) & (raw < e)]
        edges = absolute[r * logic:(r + 1) * logic + 1:p['frames_per_direction']]
        d = np.searchsorted(edges, ev, side='right') - 1
        t = (d + (ev - edges[d]) / np.diff(edges)[d]) * p['direction_duration_s']
        traces.append(F.smooth(t, grid, p['bandwidth_s']))
    traces = np.array(traces)
    if count < 2 or traces.var(axis=1).mean() == 0: return np.nan, count
    return float(traces.mean(0).var() / traces.var(axis=1).mean()), count


db = storage.MasterDataset(LS / 'retina_master.sqlite')
rf = db.rf_scalars(); stim = db.table('stimuli')
rows = []
for rec in RECS:
    sub = rf[rf.recording == rec]
    assert sub.context.nunique() == 1, sub.context.unique()
    bar = int(stim[(stim.recording == rec) & (stim.role == 'moving_bar')].stimulus_index.iloc[0])
    for r in sub.itertuples():
        if not (r.rf_quality > 10): continue
        cell = int(r.cell_index)
        try:
            res = F.response(db, rec, cell, bar, 'moving_bar', p)
            dsi, osi = F.selectivity(res['direction_response'], p['directions_deg'])
            qi, n = bar_qi(db, rec, cell, bar)
            resp = res['direction_response']
            pref = p['direction_names'][int(np.argmax(resp))]
            vec = np.sum(resp * np.exp(2j * np.deg2rad(p['directions_deg'])))
            bar_axis = (np.rad2deg(np.angle(vec)) / 2) % 180  # preferred motion axis (door compass, x right y up)
            spikes = res['spikes_used']
        except (ValueError, KeyError) as exc:
            dsi = osi = qi = bar_axis = np.nan; pref = str(exc); spikes = 0
        rows.append(dict(rec=rec, cell=cell, context=r.context, rf_quality=r.rf_quality,
                         elongation=r.elongation, rf_angle_deg=r.orientation_deg,
                         edge_margin_um=r.rf_edge_margin_um, at_edge=bool(r.rf_edge_margin_um <= 0),
                         osi=osi, dsi=dsi, bar_qi=qi, bar_spikes=spikes, bar_pref=pref, bar_axis_deg=bar_axis))
    print(rec, sum(1 for x in rows if x['rec'] == rec), flush=True)
db.close()
pd.DataFrame(rows).to_csv(OUT, index=False)
print('wrote', OUT)
