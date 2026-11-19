#!/usr/bin/env python3
"""
build_cluster_explorer.py
=========================
Turn one clustering run from ``master_clustering_cache`` into a single
interactive HTML page and open it in your browser.

What it does
------------
1. Loads the clustering you point at (``fit_<key>.pkl``) and the prepared data it
   was fitted on (``data_<signature>.pkl``, found automatically).
2. Works out, from the saved pass/fail flags, which cells were selected and why.
3. Optionally rebuilds every individual repeat of every cell from the spike times
   in ``retina_master.sqlite`` (checked against the stored averages).
4. Optionally reads the noise recordings (``noise_data.nc``) and cuts a
   ``CROP_UM`` wide window around each cell's most important pixel, for the
   covariance map (``cm_most_important``) and/or the RMS map (``rms``).
5. Writes ONE self-contained HTML file (no internet needed) and opens it.

If the clustering did not use receptive field features, every receptive field
element (traces, centre size, maps, text in the dataset and method tabs) is
removed from the page. The same goes for the moving bar and for DSI/OSI.

Files needed in the same folder as this script:
    explorer_template.html      (the page; contains d3 so it works offline)

Needs: numpy, pandas. xarray (+ netCDF4 or h5netcdf) only when RF maps are on.

Usage
-----
Edit the SETTINGS below, then run it from PyCharm, or from a terminal:

    python build_cluster_explorer.py                       # uses FIT below
    python build_cluster_explorer.py fit_458539eb5bbe16d2e86f.pkl
    python build_cluster_explorer.py 458539eb5bbe16d2e86f --rf-map cm
    python build_cluster_explorer.py latest --rf-map none --no-repeats
    python build_cluster_explorer.py latest --noise 14_08_2026=F:\\Laura\\...\\noise_data.nc
"""
import argparse
import base64
import collections
import datetime
import io
import json
import math
import pickle
import re
import sqlite3
import sys
import warnings
import webbrowser
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustering_data_extraction_all_functions as storage  # noqa: E402  (door view, RF pixel rule)

# =============================================================================
# SETTINGS  (everything here can also be set on the command line)
# =============================================================================
CACHE_DIR = Path(__file__).resolve().parent / "master_clustering_cache"

# Which clustering to show: a file name, a bare key, a full path, or 'latest'.
FIT = 'latest'

# retina_master.sqlite. None = look next to the cache folder (one level up).
MASTER_FILE = None

# Where the HTML goes. None = <CACHE_DIR>/explorer
OUT_DIR = None

# Receptive field maps taken from noise_data.nc: 'cm' (cm_most_important),
# 'rms', 'both' or 'none'. Only used if the clustering used RF features.
RF_MAP = 'both'

# Width of the square window cut out around the most important pixel (micrometres).
CROP_UM = 500

# Also show each cell's map from the other (finer) noise stimulus of its recording.
RF_OTHER = True

# STA videos made by make_sta_videos.py from the v2 noise data. None = no videos.
STA_VIDEO_DIR = Path('/mnt/mea_nas/Laura/RGC_clustering/sta_videos_v2')

# Feature space tab (cells plotted in the clustering space).
FEATURE_SPACE = True

# Rebuild individual repeats (rasters) from the spike times in the master file.
INCLUDE_REPEATS = True

OPEN_BROWSER = True

# Use these noise_data.nc files instead of the paths stored in the master file,
# e.g. {'14_08_2026': r'F:\Laura\zebrafish_14_08_2026\Phase_00\noise_analysis\noise_data.nc'}
NOISE_PATH_OVERRIDES = {}

TITLE = 'Retinal ganglion cell clusters'
# =============================================================================

STIMS_ALL = ('scf', 'csteps', 'chirp', 'moving_bar')
SHORT = {'scf': 'scf', 'csteps': 'csteps', 'chirp': 'chirp', 'moving_bar': 'bar'}
STIM_WORDS = {'scf': 'full-field flashes', 'csteps': 'contrast steps', 'chirp': 'a chirp', 'moving_bar': 'a moving bar'}
COMP_WORDS = {'scf': 'flashes', 'csteps': 'contrast steps', 'chirp': 'the chirp', 'moving_bar': 'the moving bar'}
RF_FEATURES = ('center_size_mm2', 'surround_size_mm2', 'elongation', 'orientation', 'temporal_sta')
EXTRA_FEATURES = ('DSI', 'OSI') + RF_FEATURES
MAX_MAP_PX = 64


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def npy(blob):
    return np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)


def sig(x, n=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return float(f'{x:.{n}g}') if math.isfinite(x) else None


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def fmt(n):
    return f'{int(n):,}'


def date_key(rec):
    d, m, y = rec.split('_')
    return f'{y}-{m}-{d}'


def join_words(items):
    items = list(items)
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


# ----------------------------------------------------------------------------
# loading the run
# ----------------------------------------------------------------------------
def find_fit(cache, spec):
    cache = Path(cache)
    if spec in (None, '', 'latest'):
        files = sorted(cache.glob('fit_*.pkl'), key=lambda p: p.stat().st_mtime)
        if not files:
            sys.exit(f'No fit_*.pkl files found in {cache}')
        return files[-1]
    p = Path(str(spec))
    if p.is_file():
        return p
    key = re.sub(r'^fit_|\.pkl$', '', p.name)
    cand = cache / f'fit_{key}.pkl'
    if not cand.is_file():
        sys.exit(f'Cannot find {cand}')
    return cand


def load_run(cache, spec):
    fit_path = find_fit(cache, spec)
    log(f'Clustering: {fit_path.name}')
    with open(fit_path, 'rb') as fh:
        fit = pickle.load(fh)
    data_path = fit_path.parent / f"data_{fit['data_signature']}.pkl"
    if not data_path.is_file():
        sys.exit(f'The prepared data this clustering used is missing: {data_path}')
    with open(data_path, 'rb') as fh:
        data = pickle.load(fh)
    return fit_path, fit, data


# ----------------------------------------------------------------------------
# working out how cells were selected
# ----------------------------------------------------------------------------
def infer_fff(ga):
    """Rule ('any'/'all') and flash-quality cutoff that reproduce passes_fff."""
    q = ga[['scf_qi', 'csteps_qi', 'chirp_qi']].to_numpy(float)
    q = np.where(np.isfinite(q), q, -np.inf)
    passed = ga['passes_fff'].to_numpy(bool)
    if not len(ga) or not passed.any() or passed.all():
        return 'any', None
    best = None
    for rule, stat in (('any', q.max(axis=1)), ('all', q.min(axis=1))):
        fail_max = stat[~passed].max()
        pass_min = stat[passed].min()
        if fail_max < pass_min:
            cands = [c / 100 for c in range(0, 100, 5) if fail_max <= c / 100 < pass_min]
            if cands:
                mid = (max(fail_max, 0) + pass_min) / 2
                thr = min(cands, key=lambda c: abs(c - mid))
            else:
                thr = float(fail_max) if np.isfinite(fail_max) else 0.0
            best = (rule, thr)
            break
    return best if best else ('any', None)


def infer_rf(sa):
    q = sa['rf_quality'].to_numpy(float)
    passed = sa['rf_pass'].to_numpy(bool)
    fin = np.isfinite(q)
    if not passed.any():
        return None
    pass_min = q[passed & fin].min()
    fails = q[~passed & fin]
    fail_max = fails.max() if len(fails) else 0.0
    cands = [float(c) for c in range(0, 1000, 5) if fail_max <= c < pass_min]
    if cands:
        mid = (fail_max + pass_min) / 2
        return min(cands, key=lambda c: abs(c - mid))
    return float(fail_max)


def infer_mode(sa, rf_used):
    ff, rf, inc = sa['fff_pass'].to_numpy(bool), sa['rf_pass'].to_numpy(bool), sa['included'].to_numpy(bool)
    rules = {'fff_or_rf': ff | rf, 'fff_and_rf': ff & rf, 'fff_only': ff, 'rf_only': rf, 'none': np.ones(len(sa), bool)}
    order = ['fff_or_rf', 'fff_and_rf', 'rf_only', 'fff_only', 'none'] if rf_used else \
            ['fff_only', 'fff_or_rf', 'fff_and_rf', 'rf_only', 'none']
    hits = [m for m in order if (rules[m] == inc).all()]
    return hits[0] if hits else ('fff_or_rf' if rf_used else 'fff_only')


# ----------------------------------------------------------------------------
# master file access (optional)
# ----------------------------------------------------------------------------
def open_master(path):
    if path and Path(path).is_file():
        return sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True)
    return None


def load_stimuli_table(con):
    return pd.read_sql_query(
        'select recording, stimulus_index, name, role, group_number, selected, status from stimuli', con)


def indices_for(st, rec, role, groups):
    sub = st[(st.recording == rec) & (st.role == role) & (st.selected == 1) & (st.status == 'complete')]
    if role == 'moving_bar':
        return [int(i) for i in sub.stimulus_index]
    gs = {int(g) for g in groups}
    return [int(i) for i, g in zip(sub.stimulus_index, sub.group_number) if pd.notna(g) and int(g) in gs]


_timing_cache = {}


def timing(con, rec, idx):
    key = (rec, idx)
    if key not in _timing_cache:
        meta = json.loads(con.execute('select metadata from stimuli where recording=? and stimulus_index=?',
                                      (rec, idx)).fetchone()[0])
        rel_blob, qmeta = con.execute(
            'select relative_triggers, metadata from qi_timing where recording=? and stimulus_index=?',
            (rec, idx)).fetchone()
        rel = npy(rel_blob).astype(np.int64)
        inferred = int(json.loads(qmeta).get('padding', {}).get('inferred_count', 0))
        _timing_cache[key] = (meta, rel, inferred)
    return _timing_cache[key]


def spikes(con, rec, idx, cell):
    row = con.execute('select spikes from observations where recording=? and stimulus_index=? and cell_index=?',
                      (rec, idx, int(cell))).fetchone()
    return npy(row[0]).astype(np.int64) if row else np.array([], np.int64)


def split_repeats(raw, meta, rel, inferred, role, duration, frames_per_direction=300, direction_s=5.0, recording=None):
    """Same logic as response() in clustering_master_functions.py, but per repeat.
    Moving-bar directions are put in door-view order (storage.BAR_DOOR_NAMES) if recording is given."""
    logic = int(meta['stimulus_repeat_logic'])
    fs = float(meta['sampling_freq'])
    count = min(int(meta['nr_repeats']), (len(rel) - 1) // logic)
    absolute = int(meta['begin_fr']) + rel
    if not inferred:
        while count and absolute[count * logic] > int(meta['end_fr']):
            count -= 1
    out = []
    for r in range(count):
        b, e = absolute[r * logic], absolute[(r + 1) * logic]
        ev = raw[(raw >= b) & (raw < e)]
        if role == 'moving_bar':
            edges = absolute[r * logic:(r + 1) * logic + 1:frames_per_direction]
            d = np.searchsorted(edges, ev, side='right') - 1
            t = (d + (ev - edges[d]) / np.diff(edges)[d]) * direction_s
            if recording is not None:
                t = storage.bar_times_to_door(t, recording, direction_s)
        else:
            t = (ev - b) / fs
        out.append(t[(t >= 0) & (t < duration)])
    return out


def _varint(n, buf):
    while True:
        b = n & 127
        n >>= 7
        if n:
            buf.append(b | 128)
        else:
            buf.append(b)
            return


def encode_repeats(reps):
    buf = bytearray()
    for t in reps:
        ms = np.sort(np.rint(t * 1000).astype(np.int64))
        _varint(len(ms), buf)
        prev = 0
        for v in ms:
            _varint(int(v) - prev, buf)
            prev = int(v)
    return base64.b64encode(bytes(buf)).decode()


def smooth(times, grid, bw):
    v = np.zeros(len(grid))
    for s in range(0, len(times), 1024):
        z = (grid[:, None] - times[None, s:s + 1024]) / bw
        v += np.exp(-.5 * z * z).sum(axis=1)
    return v / (bw * np.sqrt(2 * np.pi))


def build_repeats(rows, con, stims, data, processing):
    st = load_stimuli_table(con)
    durations = {s: float(data['responses'][s]['grid'][-1]) for s in stims}
    fpd = int(processing.get('frames_per_direction', 300))
    dsec = float(processing.get('direction_duration_s', 5.0))
    out = {s: [] for s in stims}
    per_cell = {}
    check = []
    by_rec = collections.defaultdict(list)
    for i, r in rows.iterrows():
        by_rec[r['recording']].append(i)
    for rec, ids in by_rec.items():
        for i in ids:
            r = rows.loc[i]
            for s in stims:
                idxs = indices_for(st, rec, s, r['groups'])
                presentations = []
                for ix in idxs:
                    meta, rel, inf = timing(con, rec, ix)
                    raw = spikes(con, rec, ix, r['cell_index'])
                    presentations.append(split_repeats(raw, meta, rel, inf, s, durations[s], fpd, dsec, recording=rec))
                reps = [t for p in presentations for t in p]
                out[s].append({'uid': r['uid'], 'reps': len(reps), 'd': encode_repeats(reps)})
                per_cell[(r['uid'], s)] = (len(reps), int(sum(len(t) for t in reps)))
                if s != 'moving_bar' and len(check) < 12 and reps and i % 97 == 0:
                    grid = np.linspace(0, durations[s], len(data['responses'][s]['grid']))
                    rate = np.mean([smooth(np.concatenate(p), grid, 0.05) / max(len(p), 1)
                                    for p in presentations if p], axis=0)
                    ref = data['responses'][s]['rate'][r['data_row']]
                    check.append(float(np.abs(rate - ref).max() / max(1.0, np.nanmax(ref))))
        log(f'  repeats: {rec} done')
    if check:
        worst = max(check)
        if worst > 1e-6:
            log(f'  WARNING: rebuilt repeats differ from the stored averages (worst relative difference {worst:.2g}). '
                'Check that the master file is the one this clustering was made from.')
        else:
            log(f'  repeats match the stored averages ({len(check)} spot checks, worst difference {worst:.1e})')
    return out, per_cell


# ----------------------------------------------------------------------------
# receptive field maps
# ----------------------------------------------------------------------------
def noise_path(con, rec, overrides):
    if rec in overrides:
        return Path(overrides[rec])
    if con is not None:
        row = con.execute("select provenance from rf_sources where recording=? and status='complete' limit 1",
                          (rec,)).fetchone()
        if row:
            try:
                return Path(json.loads(row[0])['file']['path'])
            except (KeyError, ValueError, TypeError):
                pass
    return None


def _crop(arr, xs, ys, cx, cy, step, n, dx, dy, recording):
    """Nearest-neighbour window of n x n samples spaced `step` um, centred on (cx, cy), cut in
    the stimulus frame (row = y down) and then turned into the door view (row 0 = up, col = right),
    the same orientation as the STA videos and the moving-bar polar plot."""
    offs = (np.arange(n) - (n - 1) / 2) * step
    xq, yq = cx + offs, cy + offs
    ix = np.abs(xs[None, :] - xq[:, None]).argmin(axis=1)
    iy = np.abs(ys[None, :] - yq[:, None]).argmin(axis=1)
    ok_x = np.abs(xs[ix] - xq) <= dx * 0.51 + 1e-9
    ok_y = np.abs(ys[iy] - yq) <= dy * 0.51 + 1e-9
    block = arr[np.ix_(iy, ix)].astype(float)
    block[~ok_y, :] = np.nan
    block[:, ~ok_x] = np.nan
    return storage.stimulus_image_to_door(block, recording)


def rf_scale_for(rows, data):
    """{recording: length factor} from the clustering setting RF_SCALE_BY_PIXEL, by the pixel size of the
    recording's clustering noise (e.g. 12 px noise really 5 um/px, not 4 -> 1.25). The factor belongs to the
    projector, so it is used for every map of that recording."""
    table = {int(k): float(v) for k, v in (data.get('rf_scale_by_pixel') or {}).items()}
    out = {}
    for rec, g in rows.groupby('recording'):
        ctx = next((c for c in g['rf_context'] if isinstance(c, str)), None)
        m = re.search(r'(\d+)px', ctx or '')
        out[rec] = table.get(int(m.group(1)), 1.0) if m else 1.0
    return out


def _cut_maps(ds, chan_name, todo, want_cm, want_rms, crop_um, recording):
    """Cut a square window around each cell's strongest pixel from one channel of noise_data.nc."""
    chans = [str(c) for c in ds['channel'].values] if 'channel' in ds.coords else []
    sel = {'channel': ds['channel'].values[chans.index(chan_name)]} if chans else {}

    def var(name):
        if name not in ds:
            return None
        v = ds[name]
        return v.sel(**sel) if ('channel' in v.dims and sel) else v

    v_rms, v_cm = var('rms'), var('cm_most_important')
    if v_rms is None and v_cm is None:
        return [], None
    xs, ys = ds['x'].values.astype(float), ds['y'].values.astype(float)
    dx = float(np.median(np.abs(np.diff(xs)))) if len(xs) > 1 else 1.0
    dy = float(np.median(np.abs(np.diff(ys)))) if len(ys) > 1 else 1.0
    step = max(dx, dy, crop_um / (MAX_MAP_PX - 1.0))
    n = int(round(crop_um / step))
    n = n + 1 if n % 2 == 0 else n
    out = []
    for k in range(0, len(todo), 40):
        chunk = todo[k:k + 40]
        cell_ids = [c for _, c in chunk]

        def load(v):
            if v is None:
                return None
            return v.sel(cell_index=cell_ids).transpose('cell_index', 'y', 'x').values.astype(float)

        a_rms, a_cm = load(v_rms), load(v_cm)
        for j, (uid, ci) in enumerate(chunk):
            base = a_rms[j] if a_rms is not None else None
            if base is None or not np.isfinite(base).any() or np.nanmax(base) <= 0:
                base = np.abs(a_cm[j]) if a_cm is not None else None
            if base is None or not np.isfinite(base).any():
                continue
            iy0, ix0 = np.unravel_index(np.nanargmax(base), base.shape)
            cx, cy = xs[ix0], ys[iy0]
            door_x, door_y = storage.stimulus_point_to_door(cx, cy, recording)
            # shown as door view, x = right, y = UP (so the axis numbers grow upwards)
            row = {'uid': uid, 'ch': chan_name, 'n': n, 'um': sig(step, 5), 'cx': sig(door_x, 6), 'cy': sig(-door_y, 6)}
            any_map = False
            if want_cm and a_cm is not None and np.isfinite(a_cm[j]).any():
                blk = _crop(a_cm[j], xs, ys, cx, cy, step, n, dx, dy, recording)
                s = np.nanmax(np.abs(blk)) if np.isfinite(blk).any() else 0
                if s and np.isfinite(s):
                    q = np.rint(np.nan_to_num(blk / s) * 127).astype(np.int8)
                    row['cm_d'] = base64.b64encode(q.tobytes()).decode()
                    row['cm_s'] = sig(s, 5)
                    any_map = True
            if want_rms and a_rms is not None and np.isfinite(a_rms[j]).any():
                blk = _crop(a_rms[j], xs, ys, cx, cy, step, n, dx, dy, recording)
                s = np.nanmax(blk) if np.isfinite(blk).any() else 0
                if s and np.isfinite(s) and s > 0:
                    q = np.rint(np.clip(np.nan_to_num(blk) / s, 0, 1) * 255).astype(np.uint8)
                    row['rms_d'] = base64.b64encode(q.tobytes()).decode()
                    row['rms_s'] = sig(s, 5)
                    any_map = True
            if any_map:
                out.append(row)
    return out, (n, step)


def build_rfmaps(rows, con, which, crop_um, overrides, other=True, scale=None):
    """Maps from the noise stimulus used for clustering ('rfmaps') and, if other=True, from the
    best other noise stimulus of the same recording ('rfmaps_other': for each cell the channel
    with the highest RF quality, e.g. the finer 2/4/6/8 px noise)."""
    import xarray as xr
    want_cm = which in ('cm', 'both')
    want_rms = which in ('rms', 'both')
    main_rows, other_rows = [], []
    for rec, g in rows.groupby('recording', sort=False):
        ctx = next((c for c in g['rf_context'] if isinstance(c, str)), None)
        path = noise_path(con, rec, overrides)
        if path is None or not path.exists():
            log(f'  RF maps: {rec}: noise file not found ({path}). Set NOISE_PATH_OVERRIDES to point at it.')
            continue
        try:
            ds = xr.open_dataset(path)
        except Exception as exc:  # noqa: BLE001
            log(f'  RF maps: {rec}: cannot open {path} ({exc})')
            continue
        try:
            chans = [str(c) for c in ds['channel'].values] if 'channel' in ds.coords else []
            chan_name = ctx.split(':', 2)[2] if ctx and ctx.count(':') >= 2 else None
            if chans and chan_name not in chans:
                if len(chans) == 1:
                    chan_name = chans[0]
                else:
                    log(f'  RF maps: {rec}: channel {chan_name!r} not in the file (has {chans}); skipped')
                    continue
            ids = set(int(c) for c in ds['cell_index'].values)
            todo = [(u, int(c)) for u, c in zip(g['uid'], g['cell_index']) if int(c) in ids]
            made, geo = _cut_maps(ds, chan_name, todo, want_cm, want_rms, crop_um, rec)
            f = (scale or {}).get(rec, 1.0)
            if f != 1.0:
                # same window, axes in corrected micrometres (RF_SCALE_BY_PIXEL)
                for row in made:
                    row.update(um=sig(float(row['um']) * f, 5), cx=sig(float(row['cx']) * f, 6), cy=sig(float(row['cy']) * f, 6))
            main_rows += made
            if geo:
                log(f'  RF maps: {rec}: {len(made)} maps ({geo[0]}x{geo[0]} samples, {geo[1]:.1f} um apart, channel {chan_name})')
            others = [c for c in chans if c != chan_name]
            if other and others and 'quality' in ds:
                quality = ds['quality'].sel(cell_index=[c for _, c in todo]).transpose('cell_index', 'channel')
                qv = quality.values.astype(float)
                order = [chans.index(c) for c in others]
                by_channel = {}
                for j, (uid, ci) in enumerate(todo):
                    vals = qv[j, order]
                    if np.isfinite(vals).any():
                        best = others[int(np.nanargmax(vals))]
                        by_channel.setdefault(best, []).append((uid, ci))
                for ch, cells in by_channel.items():
                    made, _ = _cut_maps(ds, ch, cells, want_cm, want_rms, crop_um, rec)
                    f = (scale or {}).get(rec, 1.0)
                    for row in made:
                        if f != 1.0:
                            row.update(um=sig(float(row['um']) * f, 5), cx=sig(float(row['cx']) * f, 6), cy=sig(float(row['cy']) * f, 6))
                    other_rows += made
                log(f'  RF maps: {rec}: other noise maps: ' + ', '.join(f'{ch} {len(c)}' for ch, c in by_channel.items()))
        finally:
            ds.close()
    return main_rows, other_rows


# ----------------------------------------------------------------------------
# moving bar direction tuning, feature space, STA videos
# ----------------------------------------------------------------------------
def build_bar_dirs(rows, con, processing):
    """Peak rate (Hz, 100 ms bins, mean over repeats) in each bar direction, as used for DSI/OSI."""
    st = load_stimuli_table(con)
    p = processing
    width, dur = p.get('selectivity_bin_s', 0.1), p['direction_duration_s']
    bins = np.r_[np.arange(0, dur, width), dur]
    out = {}
    for _, r in rows.iterrows():
        idxs = indices_for(st, r['recording'], 'moving_bar', r['groups'])
        per_stim = []
        for idx in idxs:
            meta, rel, inferred = timing(con, r['recording'], idx)
            reps = split_repeats(spikes(con, r['recording'], idx, r['cell_index']), meta, rel, inferred, 'moving_bar',
                                 p['durations']['moving_bar'], p['frames_per_direction'], dur, recording=r['recording'])
            if not reps:
                continue
            t = np.concatenate(reps)
            per_stim.append([np.max(np.histogram(t[(t >= d * dur) & (t < (d + 1) * dur)] - d * dur, bins)[0]
                                    / (np.diff(bins) * len(reps))) for d in range(p['directions'])])
        if per_stim:
            out[r['uid']] = [sig(v, 4) for v in np.mean(per_stim, axis=0)]
    return out


SPACE_NAMES = {'scf': 'Flashes', 'csteps': 'Contrast steps', 'chirp': 'Chirp', 'moving_bar': 'Moving bar',
               'DSI': 'DSI', 'OSI': 'OSI', 'center_size_mm2': 'RF centre size', 'surround_size_mm2': 'RF surround size',
               'elongation': 'RF elongation', 'orientation': 'RF orientation', 'temporal_sta': 'RF time course'}


def build_space(rows, fit, processing, n_axes=6):
    """The clustering space itself: per-column labels, cell coordinates, cluster Gaussians and
    two sets of 2-D views (combined PCA = largest spread; LDA = best separation of clusters).
    Missing values are shown at the column mean (display only; the clustering never fills them)."""
    feats = fit['features']
    x_all = np.asarray(feats['x'], float)
    names = processing.get('direction_names', [])
    labels, groups = [], []
    for name, seg, ncol, _ in feats['column_blocks']:
        group = SPACE_NAMES.get(name, name)
        if name in ('scf', 'csteps'):
            base = f'{group} {4 * seg}\u2013{4 * (seg + 1)} s'
        elif name == 'moving_bar':
            base = f'{group} {names[seg] if seg < len(names) else seg + 1}'
        else:
            base = group
        for c in range(ncol):
            labels.append(f'{base} PC{c + 1}' if (ncol > 1 or name in RESPONSE_BLOCKS) else base)
            groups.append(group)
    X = x_all[rows['data_row'].to_numpy()]
    col_mean = np.nanmean(X, axis=0)
    Xf = np.where(np.isfinite(X), X, col_mean)
    lab = rows['cluster'].to_numpy()
    axes = []
    Xc = Xf - Xf.mean(axis=0)
    _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    ratio = sv ** 2 / np.sum(sv ** 2)
    for i in range(min(n_axes, len(sv))):
        axes.append(dict(id=f'pc{i + 1}', kind='pca', label=f'Combined PC{i + 1} ({ratio[i]:.1%} of variance)',
                         w=[sig(v, 5) for v in vt[i]]))
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        k = len(np.unique(lab))
        if k > 2:
            lda = LinearDiscriminantAnalysis(n_components=min(n_axes, k - 1)).fit(Xf, lab)
            for i in range(lda.scalings_.shape[1] if lda.scalings_.ndim == 2 else 0):
                if i >= min(n_axes, k - 1):
                    break
                w = lda.scalings_[:, i] / np.linalg.norm(lda.scalings_[:, i])
                axes.append(dict(id=f'ld{i + 1}', kind='lda',
                                 label=f'Discriminant {i + 1} ({lda.explained_variance_ratio_[i]:.1%} of cluster separation)',
                                 w=[sig(v, 5) for v in w]))
    except ImportError:
        log('Feature space: scikit-learn not installed, only PCA views.')
    points = [dict(uid=u, cluster=int(c), x=[sig(v, 4) for v in xv]) for u, c, xv in zip(rows['uid'], lab, Xf)]
    model = fit['model']
    clusters = []
    for raw, disp in sorted(fit['label_mapping'].items(), key=lambda kv: kv[1]):
        clusters.append(dict(cluster=int(disp), weight=sig(model['weights'][raw], 4),
                             mean=[sig(v, 4) for v in model['means'][raw]],
                             var=[sig(v, 4) for v in model['variances'][raw]]))
    return dict(space_points=points, space_axes=axes, space_clusters=clusters), labels, groups


RESPONSE_BLOCKS = ('scf', 'csteps', 'chirp', 'moving_bar', 'temporal_sta')


def build_sta_videos(rows, video_dir, out_dir, scale=None):
    """Link each cell to its STA video (made by the lost video script; files named
    <rec>_<channel>_cell<N>_<um>um_<fps>fps_from<a>to<b>ms.mp4) and copy the files beside the page."""
    import shutil
    import subprocess
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        log(f'STA videos: folder not found ({video_dir}); skipped.')
        return []
    pat = re.compile(r'^(?P<rec>\d\d_\d\d_\d{4})_(?P<ch>.+)_cell(?P<cell>\d+)_(?P<um>\d+)um_(?P<fps>[\d.]+)fps_from(?P<a>-?\d+)to(?P<b>-?\d+)ms\.mp4$')
    found = {}
    for f in video_dir.iterdir():
        m = pat.match(f.name)
        if m:
            found[(m['rec'], int(m['cell']), m['ch'])] = (f, m)
    target = out_dir / 'sta_videos'
    target.mkdir(parents=True, exist_ok=True)
    out, frames_cache = [], {}
    for _, r in rows.iterrows():
        ch = r['rf_context'].split(':', 2)[2] if isinstance(r.get('rf_context'), str) else None
        hit = found.get((r['recording'], int(r['cell_index']), ch))
        if hit is None:
            hit = next((v for k, v in found.items() if k[0] == r['recording'] and k[1] == int(r['cell_index'])), None)
        if hit is None:
            continue
        f, m = hit
        a, b = int(m['a']), int(m['b'])
        key = (a, b, m['fps'])
        if key not in frames_cache:
            try:
                res = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                      '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(f)],
                                     capture_output=True, text=True, timeout=30)
                frames_cache[key] = int(res.stdout.strip().split(',')[0])
            except Exception:  # noqa: BLE001
                frames_cache[key] = None
        frames = frames_cache[key]
        dt = (b - a) / (frames - 1) if frames and frames > 1 else 10.0
        frames = frames or int(round((b - a) / dt)) + 1
        dst = target / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size or dst.stat().st_mtime < f.stat().st_mtime:
            shutil.copy2(f, dst)
        out.append(dict(uid=r['uid'], src='sta_videos/' + f.name, ch=m['ch'], frames=frames, fps=sig(float(m['fps']), 3),
                        dt_ms=sig(dt, 4), zero=int(round(-a / dt)),
                        win_um=sig(int(m['um']) * (scale or {}).get(r['recording'], 1.0), 5)))
    log(f'STA videos: {len(out)} of {len(rows)} cells have a video (copied to {target})')
    return out


# ----------------------------------------------------------------------------
# main build
# ----------------------------------------------------------------------------
FEATURE_WORDS = {'scf': 'full-field flash responses (8 colours)', 'csteps': 'contrast step responses',
                 'chirp': 'chirp response', 'moving_bar': 'moving bar response (8 directions)',
                 'DSI': 'direction selectivity (DSI)', 'OSI': 'orientation selectivity (OSI)',
                 'center_size_mm2': 'RF centre size', 'surround_size_mm2': 'RF surround size',
                 'elongation': 'RF elongation', 'orientation': 'RF orientation',
                 'temporal_sta': 'RF time course (temporal STA)'}
FEATURE_WHY_NOT = {'surround_size_mm2': 'dropped: the 32 px noise finds surrounds about a third the size the 12 px noise '
                                        'does, which no pixel-size correction explains'}


def data_used_lines(fit, data, rows, used_recs):
    """Plain-language list of exactly what went into this clustering, for the Dataset tab."""
    su = data.get('settings_used') or {}
    q = su.get('quality') or {}
    enabled = fit['enabled']
    settings = fit['settings']
    p = data['processing']
    out = []

    def add(item, detail):
        out.append(dict(item=item, detail=detail))

    add('Recordings', f"{len(used_recs)}: " + ', '.join(sorted(used_recs, key=lambda r: storage.recording_date(r))))
    rule = su.get('rf_pixel_rule')
    if rule:
        add('Noise pixel size rule', f"receptive fields only from {rule['before_px']} px noise before "
            f"{rule['change_date'].replace('_', '/')} and {rule['after_px']} px noise after it (projector pixel size changed)")
    add('Receptive field data', 'v2 noise analysis (noise_analysis_QI10/noise_data_v2.nc), one noise stimulus per recording '
        '(see the table)')
    mode = q.get('mode')
    words = {'fff_and_rf': 'both checks', 'fff_or_rf': 'either check', 'fff_only': 'the flash check only',
             'rf_only': 'the receptive field check only', 'none': 'no quality check'}
    if mode:
        groups = su.get('groups')
        gtext = 'all flash groups' if groups == 'all' else 'flash group ' + ', '.join(str(g) for g in (groups or []))
        special = [r for r in (su.get('groups_by_recording') or {}) if r in set(used_recs)]
        if special:
            gtext += ' (all groups for ' + ', '.join(special) + ')'
        add('Cells included', f"{len(rows)} cells, passing {words.get(mode, mode)}: flash quality index above "
            f"{q.get('fff_threshold'):g} in {'any' if q.get('fff_rule') == 'any' else 'all'} of flashes / steps / chirp "
            f"({gtext}), and receptive field quality above {q.get('rf_threshold'):g}")
    edge = q.get('rf_edge')
    audit = data.get('selection_audit')
    n_drop = int(audit['dropped_rf_edge'].sum()) if audit is not None and 'dropped_rf_edge' in audit else None
    n_edge = int(rows['rf_at_edge'].sum()) if 'rf_at_edge' in rows else None
    if edge == 'drop_cell':
        add('Receptive fields cut off by the stimulus edge', f"left out completely ({n_drop} cells). A cell counts as cut off "
            "if its RF (map above 30 % of its peak) or its centre outline "
            + (f"comes within {q.get('rf_edge_margin_um'):g} um of" if q.get('rf_edge_margin_um') else 'reaches')
            + " the edge of the noise stimulus")
    elif edge == 'drop_rf':
        add('Receptive fields cut off by the stimulus edge', f"{n_edge} cells kept, but their RF size and shape are not used")
    elif edge is None and 'rf_edge' in q:
        add('Receptive fields cut off by the stimulus edge', 'used as they are')
    on = [FEATURE_WORDS.get(k, k) for k, v in enabled.items() if v]
    add('Features used', ', '.join(on))
    off = [k for k, v in enabled.items() if not v]
    if off:
        add('Features not used', '; '.join(FEATURE_WORDS.get(k, k) + (f" ({FEATURE_WHY_NOT[k]})" if k in FEATURE_WHY_NOT else '')
                                         for k in off))
    scale = su.get('rf_scale_by_pixel') or data.get('rf_scale_by_pixel') or {}
    for px, f in scale.items():
        f = float(f)
        add(f'{px} px noise scale', f"taken as {4 * f:g} um per projector pixel instead of the 4 um the RF analysis assumed: "
            f"RF lengths x{f:g}, areas (centre size) x{f * f:g}. The RF map and STA video axes use the same scale")
    if not scale:
        add('RF size scale', 'sizes as stored in the RF analysis (no correction)')
    if enabled.get('surround_size_mm2') and su.get('rf_surround') not in (None, 'as_is'):
        add('Surround size', f"{su.get('rf_surround')}")
    if enabled.get('moving_bar') or enabled.get('DSI') or enabled.get('OSI'):
        add('Moving bar', 'all presentations averaged; 10 repeats each (dropped triggers re-inserted); directions put in the '
            'door-view order for every recording (the projection changed on 8 Oct 2026)'
            + (f"; DSI/OSI from the peak firing rate in each direction ({p.get('selectivity_bin_s', 0.1) * 1000:g} ms bins)"
               if enabled.get('DSI') or enabled.get('OSI') else '')
            + (f"; response curves: each direction shifted so its first clear peak is at {p.get('peak_target_s')} s"
               if enabled.get('moving_bar') and p.get('bar_peak_align') else ''))
    add('Orientation of maps and videos', 'door view: the stimulus as it lands on the retina, seen facing the MEA from '
        'the door (top = up)')
    if any(enabled.get(k) for k in ('scf', 'csteps', 'chirp', 'moving_bar')):
        add('Response curves', f"spike times smoothed with a {p['bandwidth_s'] * 1000:g} ms Gaussian; each part (flash colour, "
            f"contrast step, bar direction) compressed by PCA keeping {settings['pca_variance']:.0%} of its variance")
    if enabled.get('temporal_sta'):
        rng = p.get('sta_time_range_s', [-0.5, 0])
        add('RF time course', f"STA at the RF centre over the {abs(rng[0] - rng[1]) * 1000:g} ms before the spike, "
            f"compressed by PCA keeping {settings['pca_variance']:.0%} of its variance")
    share = settings.get('rf_share')
    fw = settings.get('feature_weights') or {}
    add('Weighting', (f"receptive field features {share:.0%} of the total weight, responses {1 - share:.0%}" if share is not None
                      else 'none: every column counts the same') + (f"; extra weights {fw}" if fw else ''))
    add('Clustering', f"Gaussian mixture ({' / '.join(settings['covariance_types'])}), number of clusters chosen by BIC "
        f"from {', '.join(str(k) for k in settings['cluster_counts'])}; {settings['n_initialisations']} starts each; "
        f"chosen: {fit['model']['k']} clusters ({fit['model']['covariance']})")
    add('Files', f"fit {fit['key']}, prepared data {fit['data_signature']}, master file revision {su.get('master_revision', '?')} "
        '(RGC_clustering/master_data_extraction/retina_master.sqlite on the NAS)')
    return out


def build(args):
    cache = Path(args.cache)
    fit_path, fit, data = load_run(cache, args.fit)
    key = fit['key']
    enabled = fit['enabled']
    A = fit['assignments'].reset_index(drop=True)
    n_all = len(A)
    group_mode = data.get('group_mode', 'average')
    stims = [s for s in STIMS_ALL if enabled.get(s)]
    if not stims:
        stims = [s for s in STIMS_ALL]
    sel_on = bool(enabled.get('DSI') or enabled.get('OSI'))
    rf_on = any(enabled.get(f) for f in RF_FEATURES)
    sta_on = bool(enabled.get('temporal_sta'))
    centre_on = bool(enabled.get('center_size_mm2') or enabled.get('elongation'))
    log(f'Features used: stimuli={stims}, selectivity={sel_on}, receptive field={rf_on}')

    master_path = Path(args.master) if args.master else (cache.parent / 'retina_master.sqlite')
    con = open_master(master_path)
    if con is None:
        log(f'Master file not found ({master_path}): repeats, noise paths and recording lists will be skipped.')
    regions = {}
    if con is not None:
        try:
            regions = {rec: (reg or None) for rec, reg in con.execute('select recording, region from recordings')}
        except sqlite3.Error:
            pass
    region_of = lambda rec: regions.get(rec) or 'unknown'

    # ---- rows
    A['uid'] = A['cell_uid'] if group_mode == 'average' else A['observation_uid']
    A['data_row'] = np.arange(n_all)
    ok = (A['status'] == 'clustered') & (A['cluster'] >= 1)
    n_unassigned = int((~ok).sum())
    rows = A[ok].copy().reset_index(drop=True)

    # ---- posterior in display numbering
    model = fit['model']
    K = len(fit['label_mapping'])
    post = np.zeros((n_all, K))
    fitted_idx = np.where(A['status'] == 'clustered')[0]
    for raw_label, disp in fit['label_mapping'].items():
        post[fitted_idx, disp - 1] = model['posterior'][:, int(raw_label)]

    # ---- selection inference
    ga, sa = data['group_audit'], data['selection_audit']
    fff_rule, fff_thr = infer_fff(ga)
    rf_thr = infer_rf(sa) if rf_on or sa['rf_pass'].any() else None
    mode = infer_mode(sa, rf_on)
    rfq = mode in ('fff_or_rf', 'fff_and_rf', 'rf_only')
    log(f'Selection (worked out from the saved flags): mode={mode}, flash cutoff={fff_thr} ({fff_rule}), RF cutoff={rf_thr}')

    # ---- per-row quality indices
    gidx = ga.set_index(['recording', 'cell_index', 'group_number'])
    qi = {'scf_qi': [], 'csteps_qi': [], 'chirp_qi': []}
    for _, r in rows.iterrows():
        vals = []
        for g in r['groups']:
            try:
                vals.append(gidx.loc[(r['recording'], r['cell_index'], int(g))][['scf_qi', 'csteps_qi', 'chirp_qi']].to_numpy(float))
            except KeyError:
                pass
        arr = np.nanmean(np.vstack(vals), axis=0) if vals else np.full(3, np.nan)
        for name, v in zip(qi, arr):
            qi[name].append(v)
    for name, v in qi.items():
        rows[name] = v

    ex = data['extras']
    resp = data['responses']
    di = rows['data_row'].to_numpy()
    sta_arr = ex['temporal_sta'] if sta_on else None
    rate_stim = stims[0]
    rows['mean_rate_hz'] = np.nanmean(resp[rate_stim]['rate'][di], axis=1)

    # ---- repeats
    repeats, per_cell = {}, {}
    use_repeats = bool(args.repeats and con is not None)
    if args.repeats and con is None:
        log('Repeats skipped: master file not available.')
    if use_repeats:
        log('Rebuilding individual repeats from the master file ...')
        try:
            repeats, per_cell = build_repeats(rows, con, stims, data, data['processing'])
        except Exception as exc:  # noqa: BLE001
            log(f'Repeats skipped: {exc!r}')
            use_repeats, repeats, per_cell = False, {}, {}

    # ---- RF maps
    rfmaps, rfmaps_other, rf_types = [], [], []
    if rf_on and args.rf_map != 'none':
        rf_try = True
        try:
            import xarray  # noqa: F401
        except ImportError:
            log('RF maps skipped: xarray is not installed in this environment.')
            rf_try = False
        if rf_try:
            log('Cutting receptive field maps from the noise recordings ...')
            rfmaps, rfmaps_other = build_rfmaps(rows, con, args.rf_map, args.crop_um, args.noise_overrides, args.rf_other,
                                                scale=rf_scale_for(rows, data))
            have_cm = any('cm_d' in r for r in rfmaps)
            have_rms = any('rms_d' in r for r in rfmaps)
            rf_types = (['cm'] if have_cm else []) + (['rms'] if have_rms else [])
    elif not rf_on:
        log('Clustering did not use receptive field features: receptive field elements are left out of the page.')

    out_dir = Path(args.out) if args.out else (cache / 'explorer')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- moving bar direction tuning (polar plot)
    bar_dirs = {}
    if 'moving_bar' in stims and con is not None:
        try:
            bar_dirs = build_bar_dirs(rows, con, data['processing'])
            log(f'Bar direction tuning: {len(bar_dirs)} cells')
        except Exception as exc:  # noqa: BLE001
            log(f'Bar direction tuning skipped: {exc!r}')

    # ---- feature space
    space, space_labels, space_groups = {}, [], []
    if args.space:
        try:
            space, space_labels, space_groups = build_space(rows, fit, data['processing'])
            log(f'Feature space: {len(space_labels)} dimensions, {len(space["space_axes"])} views')
        except Exception as exc:  # noqa: BLE001
            log(f'Feature space skipped: {exc!r}')

    # ---- STA videos
    sta_videos = []
    if rf_on and args.sta_videos and args.sta_videos.lower() != 'none':
        sta_videos = build_sta_videos(rows, args.sta_videos, out_dir, scale=rf_scale_for(rows, data))

    # ------------------------------------------------------------------ datasets
    D = {}
    D.update(space)
    if sta_videos:
        D['sta_videos'] = sta_videos
    cells = []
    for i, r in rows.iterrows():
        order = np.argsort(-post[r['data_row']])[:min(3, K)]
        d = dict(uid=r['uid'], recording=r['recording'], cell=int(r['cell_index']), cluster=int(r['cluster']),
                 conf=sig(r['posterior_max'], 6), dims=int(r['observed_dimensions']), ff_pass=bool(r['fff_pass']),
                 scf_qi=sig(r['scf_qi'], 3), csteps_qi=sig(r['csteps_qi'], 3), chirp_qi=sig(r['chirp_qi'], 3),
                 mean_rate_hz=sig(r['mean_rate_hz'], 4), region=region_of(r['recording']))
        if r['uid'] in bar_dirs:
            d['bar_dirs'] = bar_dirs[r['uid']]
        if rf_on:
            d.update(rf_pass=bool(r['rf_pass']), rf_quality=sig(r['rf_quality'], 4), rf_context=r['rf_context'])
            if 'rf_at_edge' in r and bool(r['rf_at_edge']):
                d['rf_edge'] = True
        if sel_on:
            d.update(dsi=sig(ex['DSI'][r['data_row'], 0], 3), osi=sig(ex['OSI'][r['data_row'], 0], 3))
        if centre_on:
            d.update(center_mm2=sig(ex['center_size_mm2'][r['data_row'], 0], 4),
                     elongation=sig(ex['elongation'][r['data_row'], 0], 4))
        if sta_on:
            d['has_sta'] = bool(np.isfinite(sta_arr[r['data_row']]).all())
        if use_repeats:
            for s in stims:
                n_rep, n_spk = per_cell.get((r['uid'], s), (None, None))
                d['rep_' + SHORT[s]] = n_rep
                d['spk_' + SHORT[s]] = n_spk
        for k, j in enumerate(order, 1):
            d[f'p{k}_cluster'] = int(j + 1)
            d[f'p{k}'] = sig(post[r['data_row'], j], 4)
        cells.append(d)
    D['cells'] = cells
    cdf = pd.DataFrame(cells)

    # traces (tenths of Hz)
    names = {'scf': 'traces_scf', 'csteps': 'traces_csteps', 'chirp': 'traces_chirp', 'moving_bar': 'traces_bar'}
    for s in stims:
        rate = resp[s]['rate']
        D[names[s]] = [dict(uid=u, dhz=np.rint(rate[dr] * 10).astype(int).tolist())
                       for u, dr in zip(rows['uid'], di) if np.isfinite(rate[dr]).all()]
    if use_repeats:
        for s in stims:
            D['repeats_' + SHORT[s]] = repeats[s]

    # stimulus axes
    axes = []
    for s in stims:
        g = resp[s]['grid']
        axes.append(dict(stim=s, label=STIM_WORDS[s], duration=float(g[-1]), samples=len(g), start=0.0, end=float(g[-1])))
    if sta_on:
        sg = data['sta_grid']
        axes.append(dict(stim='sta', label='Receptive field time course', duration=float(sg[-1] - sg[0]),
                         samples=len(sg), start=float(sg[0]), end=float(sg[-1])))
    D['stim_axes'] = axes

    def mean_sem(m):
        n = np.isfinite(m).sum(axis=0)
        mean = np.nanmean(m, axis=0)
        sem = np.nanstd(m, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if m.shape[0] > 1 else np.zeros(m.shape[1])
        return mean, np.nan_to_num(sem)

    cm_rows, cs_rows, sta_cells = [], [], []
    for k in sorted(cdf['cluster'].unique()):
        sel_rows = np.where(rows['cluster'].to_numpy() == k)[0]
        dr = di[sel_rows]
        for s in stims:
            mn, se = mean_sem(resp[s]['rate'][dr])
            cm_rows.append(dict(cluster=int(k), stim=s, n=len(sel_rows),
                                mean=[round(float(v), 2) for v in mn], sem=[round(float(v), 2) for v in se]))
        if sta_on:
            good = [d for d in dr if np.isfinite(sta_arr[d]).all()]
            if good:
                mn, se = mean_sem(sta_arr[good])
                cs_rows.append(dict(cluster=int(k), n=len(good), mean=[round(float(v), 4) for v in mn],
                                    sem=[round(float(v), 4) for v in se]))
            else:
                cs_rows.append(dict(cluster=int(k), n=0, mean=[], sem=[]))
    D['cluster_means'] = cm_rows
    if sta_on:
        D['cluster_sta'] = cs_rows
        D['cell_sta'] = [dict(uid=u, v=[round(float(v), 4) for v in sta_arr[dr]])
                         for u, dr in zip(rows['uid'], di) if np.isfinite(sta_arr[dr]).all()]
    if rfmaps:
        D['rfmaps'] = rfmaps
    if rfmaps_other:
        D['rfmaps_other'] = rfmaps_other

    # clusters + composition
    overall = cdf['recording'].value_counts(normalize=True)
    clusters, comp = [], []
    for k, g in cdf.groupby('cluster'):
        vc = g['recording'].value_counts()
        top = vc.index[0]
        for rec, n in vc.items():
            comp.append(dict(cluster=int(k), recording=rec, cells=int(n)))
        rc = g['recording'].map(region_of).value_counts()

        def med(c):
            return sig(g[c].median(skipna=True), 4) if c in g and g[c].notna().any() else None
        clusters.append(dict(
            cluster=int(k), cells=len(g), recordings=int(g['recording'].nunique()), top_recording=top,
            top_share=sig(vc.iloc[0] / len(g), 3), top_expected=sig(overall[top], 3),
            top_ratio=sig(vc.iloc[0] / len(g) / overall[top], 3), ff_share=sig(g['ff_pass'].mean(), 3),
            rf_cells=int(g['has_sta'].sum()) if sta_on else 0, rf_share=sig(g['has_sta'].mean(), 3) if sta_on else None,
            top_region=rc.index[0], top_region_share=sig(rc.iloc[0] / len(g), 3),
            conf_mean=sig(g['conf'].mean(), 4), dsi_med=med('dsi'), osi_med=med('osi'),
            center_med=med('center_mm2'), elong_med=med('elongation'), rate_med=med('mean_rate_hz'),
            size_class='large' if len(g) >= 20 else 'small'))
    D['clusters'], D['cluster_recording'] = clusters, comp

    # ---- recordings table
    rec_rows = []
    used_recs = sorted(cdf['recording'].unique(), key=date_key)
    all_recs = used_recs
    cells_in_master = {}
    if con is not None:
        all_recs = sorted([r[0] for r in con.execute('select recording from recordings')], key=date_key)
        cells_in_master = dict(con.execute('select recording, count(*) from cells group by recording').fetchall())
    rf_by_rec = {}
    rf_rule_broken = set()
    if con is not None:
        try:
            import clustering_data_extraction_all_functions as storage
            for rec, ch in con.execute("select recording, channel from rf_sources where status='complete'"):
                if not storage.rf_allowed(rec, ch):
                    rf_rule_broken.add(rec)
        except Exception:  # noqa: BLE001
            pass
    if con is not None and rf_on:
        for rec in all_recs:
            if rec in used_recs:
                continue
            best = np.nan
            for (blob,) in con.execute("select values_blob from features where recording=? and name='rf_quality'", (rec,)):
                try:
                    v = float(np.asarray(npy(blob)).ravel()[0])
                    if math.isfinite(v):
                        best = v if not math.isfinite(best) else max(best, v)
                except Exception:  # noqa: BLE001
                    pass
            rf_by_rec[rec] = best
    excl_rf_related = []
    for rec in all_recs:
        if rec in used_recs:
            a = sa[sa['recording'] == rec]
            g = cdf[cdf['recording'] == rec]
            inc = a[a['included'].astype(bool)]
            ff, rfp = inc['fff_pass'].astype(bool), inc['rf_pass'].astype(bool)
            if rfq:
                ffo, rfo, both = int((ff & ~rfp).sum()), int((~ff & rfp).sum()), int((ff & rfp).sum())
            else:
                ffo, rfo, both = len(inc), 0, 0
            row = dict(recording=rec, date=date_key(rec), status='Used', examined=len(a), used=len(g),
                       used_share=sig(len(g) / max(len(a), 1), 3), ff_only=ffo, rf_only=rfo, both=both,
                       neither=int(len(a) - len(g)),
                       with_sta=int(g['has_sta'].sum()) if sta_on else 0,
                       scf_qi_med=sig(g['scf_qi'].median(), 3), csteps_qi_med=sig(g['csteps_qi'].median(), 3),
                       chirp_qi_med=sig(g['chirp_qi'].median(), 3),
                       rf_quality_med=sig(g.loc[g['rf_pass'], 'rf_quality'].median(), 4) if rf_on and g['rf_pass'].any() else None,
                       rf_context=g['rf_context'].iloc[0] if rf_on else None, note='Used in this run')
            for s in STIMS_ALL:
                col = 'rep_' + SHORT[s]
                row[col] = int(g[col].median()) if use_repeats and s in stims and col in g and g[col].notna().any() else None
            if rf_on:
                row['note'] = 'Receptive fields available' if g['rf_pass'].any() else 'No cell above the receptive field cutoff'
        else:
            n = int(cells_in_master.get(rec, 0))
            if rf_on:
                best = rf_by_rec.get(rec, np.nan)
                if rec in rf_rule_broken:
                    note, rel = 'Receptive field noise breaks the 12 px / 32 px rule (pixel size change on 20/04/2026)', True
                elif not math.isfinite(best):
                    note, rel = 'No receptive field data', True
                elif rf_thr is not None and best <= rf_thr:
                    note, rel = f'No cell with receptive field quality above {rf_thr:g} (best {best:.1f})', True
                else:
                    note, rel = 'Not selected for this run', False
                excl_rf_related.append(rel)
            else:
                note = 'Not used in this run'
            row = dict(recording=rec, date=date_key(rec), status='Excluded', examined=n, used=0, used_share=0.0,
                       ff_only=None, rf_only=None, both=None, neither=None, with_sta=0, scf_qi_med=None,
                       csteps_qi_med=None, chirp_qi_med=None, rf_quality_med=None, rf_context=None,
                       rep_scf=None, rep_csteps=None, rep_chirp=None, rep_bar=None, note=note)
        row['region'] = region_of(rec)
        rec_rows.append(row)
    D['recordings'] = rec_rows

    # ---- BIC + blocks
    cand = fit['candidates'].copy()
    cand['selected'] = (cand['k'] == model['k']) & (cand['covariance'] == model['covariance'])
    D['bic_scores'] = [dict(k=int(r.k), covariance=r.covariance, bic=sig(r.bic, 8), selected=bool(r.selected))
                       for r in cand.itertuples()]
    report = fit['features']['report']
    # share of the clustering weight of each block (weighted likelihood; equal weights = share of columns)
    cw = np.asarray(fit['features'].get('column_weights', np.ones(int(sum(b[2] for b in fit['features']['column_blocks'])))), float)
    block_w, pos = {}, 0
    for name, seg, ncol, _ in fit['features']['column_blocks']:
        block_w[(name, int(seg))] = float(cw[pos:pos + ncol].sum())
        pos += ncol
    total_w = sum(block_w.values()) or 1.0
    D['feature_blocks'] = [dict(feature=r.feature, segment=int(r.segment), components=int(r.components),
                                variance_retained=sig(r.variance_retained, 3), observed_cells=int(r.observed_cells),
                                weight_share=sig(block_w.get((r.feature, int(r.segment) - 1), 0.0) / total_w, 3))
                           for r in report.itertuples()]

    # ---- run summary (text for the dataset and method tabs)
    sizes = cdf['cluster'].value_counts()
    large, small = sizes[sizes >= 20], sizes[sizes < 20]
    comps = {f: int(report.loc[report['feature'] == f, 'components'].sum()) for f in STIMS_ALL + EXTRA_FEATURES}
    extras_present = [f for f in EXTRA_FEATURES if enabled.get(f)]
    n_extras_comp = sum(comps[f] for f in extras_present)
    extra_names = []
    if sel_on:
        extra_names.append('direction and orientation selectivity')
    if centre_on:
        extra_names.append('receptive field centre size and elongation')
    if enabled.get('surround_size_mm2'):
        extra_names.append('receptive field surround size')
    if enabled.get('orientation'):
        extra_names.append('receptive field orientation')
    if sta_on:
        extra_names.append('the receptive field time course')
    extras_text = join_words(extra_names)
    extras_text = extras_text[:1].upper() + extras_text[1:] if extras_text else ''
    comp_parts = [f'{comps[s]} from {COMP_WORDS[s]}' for s in stims]
    if n_extras_comp:
        kind = 'selectivity and receptive field measures' if (sel_on and rf_on) else (
            'selectivity measures' if sel_on else 'receptive field measures')
        comp_parts.append(f'{n_extras_comp} from {kind}')
    ff_all, rfp_all = rows['fff_pass'].astype(bool), rows['rf_pass'].astype(bool)
    n_ff_only = int((ff_all & ~rfp_all).sum())
    n_rf_only = int((~ff_all & rfp_all).sum())
    n_both = int((ff_all & rfp_all).sum())
    n_pixel = sorted({m.group(1) + 'px' for c in (cdf['rf_context'].dropna().unique() if rf_on else [])
                      for m in [re.search(r'(\d+)px', str(c))] if m}, key=lambda s: int(s[:-2]))
    excl_list = [r['recording'] for r in rec_rows if r['status'] == 'Excluded']
    settings = fit['settings']
    fff_group_text = ', '.join(sorted({str(int(g)) for gs in rows['groups'] for g in gs})) or '1'
    S = {}
    S['n_cells'] = fmt(len(cdf))
    S['n_recordings'] = str(len(used_recs))
    S['n_recordings_all'] = str(len(all_recs))
    S['n_recordings_excluded'] = str(len(excl_list))
    S['n_ff_only'], S['n_rf_only'], S['n_both'] = fmt(n_ff_only), fmt(n_rf_only), fmt(n_both)
    S['n_sta'] = fmt(int(cdf['has_sta'].sum())) if sta_on else '0'
    S['n_center'] = fmt(int(cdf['center_mm2'].notna().sum())) if centre_on else '0'
    S['n_dsi'] = fmt(int(cdf['dsi'].notna().sum())) if sel_on else '0'
    S['fff_threshold'] = f'{fff_thr:g}' if fff_thr is not None else 'n/a'
    S['rf_threshold'] = f'{rf_thr:g}' if rf_thr is not None else 'n/a'
    S['fff_group'] = fff_group_text
    S['fff_rule_text'] = ('at least one of the three full-field stimuli (flashes, contrast steps, chirp)'
                          if fff_rule == 'any' else 'all three full-field stimuli (flashes, contrast steps, chirp)')
    S['n_features'] = str(model['means'].shape[1])
    S['pca_variance'] = f"{int(round(settings['pca_variance'] * 100))}%"
    share = settings.get('rf_share')
    cw = fit['features'].get('column_weights')
    if share is not None:
        S['weighting_text'] = (f"Receptive field features were given {share:.0%} of the total weight and the "
                               f"response features {1 - share:.0%}, so they count that much when cells are assigned to clusters.")
    elif cw is not None and not np.allclose(cw, 1):
        S['weighting_text'] = 'Some features were given extra weight (feature_weights in the clustering settings).'
    else:
        S['weighting_text'] = ('No extra weighting: every column counts the same, so features with more columns '
                               '(the response curves) count more than single numbers such as receptive field size.')
    S['k_selected'] = str(model['k'])
    S['bic'] = fmt(round(model['bic']))
    S['n_init'] = str(settings['n_initialisations'])
    S['max_iter'] = str(settings['max_iterations'])
    S['tolerance'] = f"{settings['tolerance']:g}"
    S['seed'] = str(settings['random_seed'])
    S['k_candidates'] = ', '.join(str(k) for k in settings['cluster_counts'])
    S['bandwidth_ms'] = f"{int(round(data['processing']['bandwidth_s'] * 1000))} ms"
    S['samples'] = fmt(data['processing']['samples'])
    S['n_large'], S['cells_large'] = str(len(large)), fmt(int(large.sum()))
    S['share_large'] = f'{large.sum() / len(cdf):.0%}'
    S['n_small'], S['cells_small'] = str(len(small)), fmt(int(small.sum()))
    S['conf_median'] = f"{cdf['conf'].median():.4f}"
    S['conf_above_99'] = f"{(cdf['conf'] > 0.99).mean():.0%}"
    S['components_text'] = join_words(comp_parts)
    S['extras_text'] = extras_text
    S['n_extras_comp'] = str(n_extras_comp)
    S['stim_list'] = join_words([STIM_WORDS[s] for s in stims])
    S['excluded_list'] = ', '.join(excl_list)
    S['used_list'] = ', '.join(used_recs)
    S['n_pixel_sizes'] = ', '.join(n_pixel)
    S['n_unassigned'] = fmt(n_unassigned)
    D['run_summary'] = [dict(key=k, label=k, value=v) for k, v in S.items()]
    D['data_used'] = data_used_lines(fit, data, rows, used_recs)

    missing = bool((report['observed_cells'] < n_all).any() or n_unassigned)
    flags = dict(
        rf=rf_on, sta=sta_on, centre=centre_on, sel=sel_on, bar='moving_bar' in stims, rfq=rfq, extras=bool(n_extras_comp),
        repeats=use_repeats, rfmap=bool(rfmaps), rfother=bool(rfmaps_other), stavideo=bool(sta_videos),
        polar=bool(bar_dirs), space=bool(space), master=con is not None, excl=bool(excl_list),
        exclrf=bool(rf_on and excl_list and excl_rf_related and all(excl_rf_related)),
        rfonly=bool(rfq and n_rf_only > 0), multipx=len(n_pixel) > 1, missing=missing,
        unassigned=n_unassigned > 0, qi=mode != 'none',
        mode_or=mode == 'fff_or_rf', mode_and=mode == 'fff_and_rf', mode_fff=mode == 'fff_only',
        mode_rf=mode == 'rf_only', mode_none=mode == 'none', used=True)
    config = dict(title=args.title, subtitle=f'{fit_path.name} \u00b7 {model["k"]} clusters \u00b7 built {datetime.date.today().isoformat()}',
                  stims=stims, flags=flags, rf_maps=rf_types, rf_crop_um=args.crop_um, key=key,
                  bar_deg=list(data['processing'].get('directions_deg', [])),
                  bar_names=list(data['processing'].get('direction_names', [])),
                  space_labels=space_labels, space_groups=space_groups)

    # ------------------------------------------------------------------ write the page
    template = Path(__file__).resolve().parent / 'explorer_template.html'
    if not template.is_file():
        sys.exit(f'explorer_template.html must sit next to this script ({template})')
    html = template.read_text(encoding='utf-8')
    payload = json.dumps(clean({'config': config, 'datasets': D}), separators=(',', ':'))
    payload = payload.replace('</', '<\\/').replace('<!--', '<\\!--')
    html = html.replace('"__EXPLORER_DATA__"', payload, 1).replace('__EXPLORER_TITLE__', args.title)
    out = out_dir / f'cluster_explorer_{key}.html'
    out.write_text(html, encoding='utf-8')
    log(f'\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB)')
    log(f'  {len(cdf)} cells, {model["k"]} clusters, repeats={"yes" if use_repeats else "no"}, '
        f'RF maps={"yes (" + ", ".join(rf_types) + ")" if rfmaps else "no"}, '
        f'receptive field content={"kept" if rf_on else "removed"}')
    if con is not None:
        con.close()
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Build the interactive cluster explorer page from a clustering run.')
    p.add_argument('fit', nargs='?', default=FIT, help="fit file, key, path or 'latest'")
    p.add_argument('--cache', default=str(CACHE_DIR))
    p.add_argument('--master', default=str(MASTER_FILE) if MASTER_FILE else None)
    p.add_argument('--out', default=str(OUT_DIR) if OUT_DIR else None)
    p.add_argument('--rf-map', choices=['cm', 'rms', 'both', 'none'], default=RF_MAP)
    p.add_argument('--crop-um', type=float, default=CROP_UM)
    p.add_argument('--no-rf-other', dest='rf_other', action='store_false', default=RF_OTHER)
    p.add_argument('--sta-videos', default=str(STA_VIDEO_DIR) if STA_VIDEO_DIR else None, help="folder of STA mp4 files, or 'none'")
    p.add_argument('--no-space', dest='space', action='store_false', default=FEATURE_SPACE)
    p.add_argument('--no-repeats', dest='repeats', action='store_false', default=INCLUDE_REPEATS)
    p.add_argument('--no-open', dest='open', action='store_false', default=OPEN_BROWSER)
    p.add_argument('--noise', action='append', default=[], metavar='REC=PATH', help='noise_data.nc for one recording')
    p.add_argument('--title', default=TITLE)
    a = p.parse_args(argv)
    a.noise_overrides = dict(NOISE_PATH_OVERRIDES)
    for item in a.noise:
        rec, _, path = item.partition('=')
        a.noise_overrides[rec.strip()] = path.strip()
    return a


def main(argv=None):
    args = parse_args(argv)
    out = build(args)
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == '__main__':
    main()
