"""Per-cell table for the RF axis vs moving-bar orientation analyses: rf_vs_bar_axis.csv.

Source: the final clustering cache (fit 90faade5, data e1768a028c3f83da5c3a) in
master_clustering_cache/ and its moving-bar response bank.

Columns
  rec, cell       recording, cell index
  elong           RF elongation (long / short axis), as used in the clustering
  osi, dsi        moving-bar OSI / DSI (clustering values, door view)
  rf_axis         RF long axis, 0-180 deg, polar-plot frame (door view, 0 = up, clockwise):
                  stimulus-frame angle -> before 8 Oct 2026 (180 - angle) % 180, from 8 Oct angle % 180
                  (= rf_analysis alignment_geometry.rf_angle_to_polar_orientation_deg)
  motion_axis     preferred bar MOTION axis, 0-180 deg, same frame: axis of the vector sum of the
                  8 direction responses on doubled angles. The preferred bar's long edge is motion_axis + 90.
  diff            angle between rf_axis and motion_axis (0-90)
  cluster         cluster in fit 90faade5
  px              RF noise pixel size (12px / 32px)

Run from Laura_scripts:
    python rf_bar_alignment/build_rf_vs_bar_axis_table.py [output.csv]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LS = HERE.parent
sys.path.insert(0, str(LS))
import clustering_data_extraction_all_functions as storage  # noqa: E402
import clustering_master_functions as F  # noqa: E402
import clustering_master_main as M  # noqa: E402

CACHE = LS / 'master_clustering_cache'
DATA = CACHE / 'data_e1768a028c3f83da5c3a.pkl'
ASSIGNMENTS = CACHE / 'assignments_90faade5297b83f0f898.csv'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'rf_vs_bar_axis.csv'


def main():
    data = F.read_cache(DATA)
    bank = F.read_cache(next(CACHE.glob('response_bank_*.pkl')))
    assign = pd.read_csv(ASSIGNMENTS).set_index(['recording', 'cell_index']).cluster
    p = M.config()['processing']
    angles = np.deg2rad(2 * np.asarray(p['directions_deg'], float))
    change = storage.recording_date(storage.PROJECTION_CHANGE_DATE)
    by_cell = {}
    for key, item in bank.items():
        if item.get('status') == 'ok' and 'direction_response' in item:
            by_cell.setdefault(key[:2], []).append(item['direction_response'])
    rows, E = data['rows'], data['extras']
    out = []
    for i, r in enumerate(rows.itertuples()):
        responses = by_cell.get((r.recording, int(r.cell_index)))
        o, el, osi, dsi = E['orientation'][i, 0], E['elongation'][i, 0], E['OSI'][i, 0], E['DSI'][i, 0]
        if not responses or not np.isfinite([o, el, osi]).all():
            continue
        resp = np.mean(responses, axis=0)
        motion = (np.rad2deg(np.angle(np.sum(resp * np.exp(1j * angles)))) / 2) % 180
        rf = (180 - o) % 180 if storage.recording_date(r.recording) < change else o % 180
        px = re.search(r'(\d+)px', str(r.rf_context))
        out.append(dict(rec=r.recording, cell=int(r.cell_index), elong=el, osi=osi, dsi=dsi, rf_axis=rf,
                        motion_axis=motion, diff=abs((rf - motion + 90) % 180 - 90),
                        cluster=assign.get((r.recording, int(r.cell_index))),
                        px=f'{px.group(1)}px' if px else None))
    table = pd.DataFrame(out)
    table.to_csv(OUT, index=False)
    print(f'{len(table)} cells -> {OUT}')


if __name__ == '__main__':
    main()
