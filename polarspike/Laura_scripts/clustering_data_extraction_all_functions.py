"""Master data file for RGC clustering: storage format, reader and extraction.

Rewritten 2026-11-19 (the original was lost with the old SSD). The reader part
(MasterDataset, dumps, fingerprint) provides exactly what
clustering_master_functions.py and build_cluster_explorer.py expect.

What the master file (SQLite) holds, per recording:
  recordings   one row per recording: phase folder, sampling rate, region, noise pixel size, notes
  cells        cell_index, cell_uid ('14_08_2026:cell_6'), electrode position
  stimuli      every stimulus: name, role (scf/csteps/chirp/moving_bar/noise_rf/other),
               flash group number, selected flag, status, timing metadata (JSON)
  qi_timing    trigger times relative to begin_fr (repaired for dropped moving-bar triggers)
  observations spike times (absolute sample numbers) of every cell for every stimulus,
               spike count, and flash QI (Baden et al. 2016, polarspike.quality_tests)
  features     per cell receptive-field values from noise_data_v2.nc (quality, centre size,
               elongation, orientation, temporal STA, ...)
  rf_sources   which noise file / channel the RF values came from
  meta         schema version and a revision id that changes whenever the file is written

Arrays are stored as zlib-compressed .npy blobs (see encode/decode).
Extraction needs polarspike (Overview.Recording) and xarray + netCDF4; use the
rf-analysis Poetry environment and run from the polarspike repository folder
(build_master_file.py takes care of the import path).
"""
from pathlib import Path
import datetime
import io
import json
import re
import sqlite3
import uuid
import zlib

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
FFF = ('scf', 'csteps', 'chirp')
RF_SCALARS = ('rf_quality', 'center_size_mm2', 'surround_size_mm2', 'elongation', 'orientation_deg',
              'tir', 'center_shift', 'center_std', 'surround_std', 'rf_x_um', 'rf_y_um',
              'rf_edge_margin_um', 'rf_outline_edge_um', 'rf_map_edge_um',
              'stimulus_half_width_x_um', 'stimulus_half_width_y_um')

SCHEMA = """
create table if not exists meta(key text primary key, value text);
create table if not exists recordings(
    recording text primary key, phase text, folder text, n_cells integer, sampling_freq real,
    region text, noise_pixel_size integer, notes text, selection text, extracted_at text);
create table if not exists cells(
    recording text, cell_index integer, cell_uid text, centre_x real, centre_y real,
    primary key(recording, cell_index));
create table if not exists stimuli(
    recording text, stimulus_index integer, name text, role text, group_number integer,
    selected integer, status text, kind text, nr_repeats integer, metadata text,
    primary key(recording, stimulus_index));
create table if not exists qi_timing(
    recording text, stimulus_index integer, relative_triggers blob, metadata text,
    primary key(recording, stimulus_index));
create table if not exists observations(
    recording text, stimulus_index integer, cell_index integer, n_spikes integer, qi real, spikes blob,
    primary key(recording, stimulus_index, cell_index));
create table if not exists features(
    recording text, cell_index integer, context text, name text, value real, values_blob blob, metadata text,
    primary key(recording, cell_index, context, name));
create table if not exists rf_sources(
    recording text, stimulus_index integer, channel text, status text, pixel_size integer, provenance text,
    primary key(recording, stimulus_index, channel));
"""
TABLES = ('meta', 'recordings', 'cells', 'stimuli', 'qi_timing', 'observations', 'features', 'rf_sources')

# Projector pixel size changed on 20/04/2026: only 12 px noise is comparable before that
# date and only 32 px noise after it (similar real checker size). RF data from any other
# combination is not used (Laura, 2026-11-19).
RF_PIXEL_RULE = dict(change_date='20_04_2026', before_px=12, after_px=32)


def recording_date(recording):
    day, month, year = (int(v) for v in recording.split('_')[:3])
    return year, month, day


# ----------------------------------------------------------------------------
# Door view: the stimulus as it lands on the retina, seen facing the MEA from the door
# (x = right, y = down). Same rule as rf_analysis Alignment_pipeline/alignment_geometry.py
# (confirmed there with the L-shaped noise, 14_08_2026 Phase_01):
#   before 8 Oct 2026: OpenGL up-down flip + projector 90 deg anticlockwise: (x, y) -> (-y, -x)
#   from 8 Oct 2026 (fps_py):           projector 90 deg anticlockwise only: (x, y) -> (y, -x)
# Stimulus frame = numpy layout of the noise/bar stimulus (x = column, y = row, + down);
# kernel.npy and the v2 rms / cm maps are in it.
# ----------------------------------------------------------------------------
PROJECTION_CHANGE_DATE = '08_10_2026'
# Moving bar: presentation order of the 8 directions (stimulus frame) and the door-view
# order every recording is put into (= the projected order before 8 Oct 2026, so the
# October explorer's names/angles stay valid).
BAR_STIMULUS_ORDER = ((1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (-1, -1), (1, 1), (-1, 1))
BAR_DOOR_NAMES = ('up', 'down', 'right', 'left', 'up-right', 'down-right', 'up-left', 'down-left')


def door_matrix(recording):
    """2x2 matrix taking stimulus (x, y) to door view (x right, y down) for this recording."""
    if recording_date(recording) >= recording_date(PROJECTION_CHANGE_DATE):
        return np.array([[0, 1], [-1, 0]])
    return np.array([[0, -1], [-1, 0]])


def _door_name(v):
    vertical = 'up' if v[1] < -0.5 else 'down' if v[1] > 0.5 else ''
    horizontal = 'right' if v[0] > 0.5 else 'left' if v[0] < -0.5 else ''
    return '-'.join(p for p in (vertical, horizontal) if p)


def bar_door_order(recording):
    """For each door-view direction (BAR_DOOR_NAMES order), the index of the presented block."""
    m = door_matrix(recording)
    shown = [_door_name(m @ np.array(v)) for v in BAR_STIMULUS_ORDER]
    return [shown.index(name) for name in BAR_DOOR_NAMES]


def bar_times_to_door(times, recording, direction_s):
    """Moving-bar spike times laid out as presented (block d = [d*s, (d+1)*s)) -> laid out in
    door-view order (block k = BAR_DOOR_NAMES[k]). Identity before 8 Oct 2026."""
    order = bar_door_order(recording)
    if order == list(range(len(order))):
        return times
    times = np.asarray(times, float)
    block = np.clip((times // direction_s).astype(int), 0, len(order) - 1)
    position = np.empty(len(order), int)
    position[order] = np.arange(len(order))
    return times + (position[block] - block) * direction_s


def stimulus_image_to_door(img, recording):
    """Square, centred image(s) [..., row = stimulus y (down), col = stimulus x] -> door view
    [..., row = down, col = right] (top of the array = up when shown with imshow/'upper')."""
    m = np.linalg.inv(door_matrix(recording))
    n = img.shape[-1]
    if img.shape[-2] != n:
        raise ValueError('stimulus_image_to_door needs a square image')
    o = np.arange(n) - (n - 1) / 2
    X, Y = np.meshgrid(o, o)                       # output grid: Y = row (down), X = col (right)
    ix = np.rint(m[0, 0] * X + m[0, 1] * Y + (n - 1) / 2).astype(int)
    iy = np.rint(m[1, 0] * X + m[1, 1] * Y + (n - 1) / 2).astype(int)
    return img[..., iy, ix]


def stimulus_point_to_door(x, y, recording):
    """Stimulus (x, y) um -> door view (x right, y down) um."""
    m = door_matrix(recording)
    return m[0, 0] * x + m[0, 1] * y, m[1, 0] * x + m[1, 1] * y


def rf_allowed(recording, channel, rule=RF_PIXEL_RULE):
    """True if the RF noise channel may be used for this recording under the pixel-size rule."""
    if not rule:
        return True
    match = re.match(r'(\d+)px', str(channel))
    pixel = int(match.group(1)) if match else None
    before = recording_date(recording) < recording_date(rule['change_date'])
    return pixel == (rule['before_px'] if before else rule['after_px'])


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def encode(array):
    buf = io.BytesIO()
    np.save(buf, np.asarray(array), allow_pickle=False)
    return zlib.compress(buf.getvalue(), 6)


def decode(blob):
    return np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)


def _default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def dumps(value):
    """Stable JSON text (sorted keys); used for settings hashes and metadata."""
    return json.dumps(value, sort_keys=True, default=_default, separators=(',', ':'))


def fingerprint(path):
    """Identifies the content of a master file. Changes whenever a recording is (re)written,
    but not when the file is merely copied (e.g. local <-> NAS)."""
    path = Path(path)
    try:
        con = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)
        try:
            row = con.execute("select value from meta where key='revision'").fetchone()
        finally:
            con.close()
        if row:
            return row[0]
    except sqlite3.Error:
        pass
    stat = path.stat()
    return f'{stat.st_size}-{stat.st_mtime_ns}'


# ----------------------------------------------------------------------------
# reader (used by clustering and explorer)
# ----------------------------------------------------------------------------
class MasterDataset:
    """Read-only access to the master file."""

    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Master file not found: {path}')
        self.path = path
        self.con = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def table(self, name):
        if name not in TABLES:
            raise KeyError(f'Unknown table {name!r}; tables are {TABLES}')
        drop = {'observations': ['spikes'], 'features': ['values_blob'], 'qi_timing': ['relative_triggers']}.get(name, [])
        columns = [r[1] for r in self.con.execute(f'pragma table_info({name})') if r[1] not in drop]
        return pd.read_sql_query(f'select {", ".join(columns)} from {name}', self.con)

    def observations(self):
        """One row per (recording, stimulus, cell) with role, flash group, spike count and QI."""
        return pd.read_sql_query(
            'select o.recording, o.stimulus_index, o.cell_index, s.name, s.role, s.group_number, s.kind, '
            's.selected, s.status, o.n_spikes, o.qi '
            'from observations o join stimuli s on s.recording=o.recording and s.stimulus_index=o.stimulus_index',
            self.con)

    def rf_scalars(self):
        """Wide table: recording, cell_index, context and one column per RF scalar."""
        long = pd.read_sql_query(
            "select recording, cell_index, context, name, value from features "
            "where value is not null and context like 'rf:%'", self.con)
        if long.empty:
            return pd.DataFrame(columns=['recording', 'cell_index', 'context', *RF_SCALARS])
        wide = long.pivot_table(index=['recording', 'cell_index', 'context'], columns='name',
                                values='value', aggfunc='first').reset_index()
        wide.columns.name = None
        for name in RF_SCALARS:
            if name not in wide:
                wide[name] = np.nan
        return wide

    def spikes(self, recording, cell_index, stimulus_index):
        row = self.con.execute(
            'select spikes from observations where recording=? and stimulus_index=? and cell_index=?',
            (recording, int(stimulus_index), int(cell_index))).fetchone()
        if row is None:
            raise KeyError(f'No observation for {recording}, cell {cell_index}, stimulus {stimulus_index}')
        return decode(row[0]).astype(np.int64)

    def stimulus_metadata(self, recording, stimulus_index):
        row = self.con.execute('select metadata from stimuli where recording=? and stimulus_index=?',
                               (recording, int(stimulus_index))).fetchone()
        if row is None:
            raise KeyError(f'No stimulus {stimulus_index} in {recording}')
        meta = json.loads(row[0])
        timing = self.con.execute('select relative_triggers from qi_timing where recording=? and stimulus_index=?',
                                  (recording, int(stimulus_index))).fetchone()
        meta['trigger_fr_relative'] = decode(timing[0]).astype(np.int64) if timing else np.array([], np.int64)
        return meta

    def feature(self, recording, cell_index, context, name):
        row = self.con.execute(
            'select value, values_blob, metadata from features where recording=? and cell_index=? and context=? and name=?',
            (recording, int(cell_index), context, name)).fetchone()
        if row is None:
            raise KeyError(f'No feature {name} for {recording} cell {cell_index} ({context})')
        value, blob, meta = row
        values = decode(blob) if blob is not None else (None if value is None else np.asarray(value))
        return dict(values=values, metadata=json.loads(meta) if meta else {})


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------
def open_for_writing(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("insert or replace into meta values ('schema_version', ?)", (str(SCHEMA_VERSION),))
    con.commit()
    return con


def bump_revision(con):
    con.execute("insert or replace into meta values ('revision', ?)", (uuid.uuid4().hex[:20],))
    con.execute("insert or replace into meta values ('modified', ?)", (datetime.datetime.now().isoformat(timespec='seconds'),))


def recordings_in(path):
    path = Path(path)
    if not path.is_file():
        return set()
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute('select recording from recordings')}
    except sqlite3.Error:
        return set()
    finally:
        con.close()


def delete_recording(con, recording):
    for table in TABLES[1:]:
        con.execute(f'delete from {table} where recording=?', (recording,))


def update_recording_info(path, selection):
    """Copy region / notes from the selection file into the master without re-extracting."""
    table = selection.get('regions', {})
    con = open_for_writing(path)
    try:
        for rec, cfg in selection['recordings'].items():
            region = cfg.get('region') or table.get(f"{rec}/{cfg.get('phase')}")
            con.execute('update recordings set region=?, notes=?, selection=? where recording=?',
                        (region, cfg.get('notes', ''), dumps(cfg), rec))
        bump_revision(con)
        con.commit()
    finally:
        con.close()


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------
def repair_triggers(rel, logic, expected_repeats, max_ratio=1.4):
    """Insert dropped triggers so a stimulus has expected_repeats full repeats.

    A dropped trigger shows up as an interval longer than the usual frame interval. The
    first and last triggers are always present (every bar spans exactly 400 s), so the
    missing ones are put, evenly spaced, into the longest intervals - including the last
    one (a long final interval means the second-to-last trigger is missing). An interval
    about k frames long receives k-1 triggers. Ties go to the earlier interval. Placing a
    trigger one gap off shifts the frames in between by one frame (16.7 ms at 60 Hz).
    Returns (new relative triggers, positions inserted)."""
    rel = np.asarray(rel, np.int64)
    inserted = []
    needed = expected_repeats * logic + 1
    while len(rel) < needed:
        d = np.diff(rel)
        typical = np.median(d)
        j = int(np.argmax(d))
        if d[j] < max_ratio * typical:
            break  # no clear gap: leave it, the stimulus will keep fewer repeats
        k = int(min(max(1, round(d[j] / typical) - 1), needed - len(rel)))
        new = rel[j] + (np.arange(1, k + 1) * d[j]) // (k + 1)
        rel = np.insert(rel, j + 1, new)
        inserted += list(range(j + 1, j + 1 + k))
    return rel, inserted


def stimulus_roles(cfg, stimulus_df):
    """Map stimulus_index -> (role, group_number, selected) from the selection entry."""
    roles = {}
    for group, indices in cfg.get('groups', {}).items():
        for role, idx in zip(FFF, indices):
            if idx is not None:
                roles[int(idx)] = (role, int(group), 1)
    for idx in cfg.get('moving_bar', []):
        roles[int(idx)] = ('moving_bar', None, 1)
    if cfg.get('rf'):
        roles[int(cfg['rf']['stimulus_index'])] = ('noise_rf', None, 1)
    out = {}
    for row in stimulus_df.itertuples():
        idx = int(row.stimulus_index)
        if idx in roles:
            out[idx] = roles[idx]
            continue
        name = str(row.stimulus_name).strip()
        if name in FFF:
            out[idx] = (name, None, 0)        # presentation not in any group: stored, not used
        elif 'shuffle' in name or re.match(r'\d+px', name):
            out[idx] = ('noise', None, 0)
        elif 'moving' in name:
            out[idx] = ('other_moving', None, 0)
        else:
            out[idx] = ('other', None, 0)
    return out


def repeat_bounds(begin_fr, end_fr, rel, logic, nr_repeats, inferred):
    """Absolute start/end sample of each full repeat (same rule as clustering_master_functions.response)."""
    absolute = int(begin_fr) + rel
    count = min(int(nr_repeats), (len(rel) - 1) // logic) if logic > 0 else 0
    if not inferred:
        while count and absolute[count * logic] > int(end_fr):
            count -= 1
    return [(int(absolute[r * logic]), int(absolute[(r + 1) * logic])) for r in range(count)]


QI_SETTINGS = dict(method='Baden et al. 2016: var_t(mean_r C) / mean_r(var_t C)',
                   bandwidth_s=0.05, samples=1000, durations=dict(scf=32., csteps=40., chirp=35.))


def _smooth(times, grid, bandwidth):
    """Gaussian kernel density, identical to clustering_master_functions.smooth."""
    values = np.zeros(len(grid))
    for start in range(0, len(times), 1024):
        z = (grid[:, None] - times[None, start:start + 1024]) / bandwidth
        values += np.exp(-.5 * z * z).sum(axis=1)
    return values / (bandwidth * np.sqrt(2 * np.pi))


def flash_qis(spikes_by_cell, bounds, fs, role):
    """Response quality index (Baden et al. 2016) of each cell for one flash stimulus.

    Each repeat is smoothed with a 50 ms Gaussian onto 1000 points over the stimulus
    duration; QI = variance over time of the mean response / mean over repeats of the
    variance over time. 1 = identical every repeat, ~0 = no reliable response.
    Checked against the QIs of the October 2026 explorer (14_08_2026: correlation 0.994-0.998,
    typical difference 0.003-0.007; polarspike.quality_tests gives a different, unbounded value)."""
    if len(bounds) < 2:
        return {}
    duration = QI_SETTINGS['durations'][role]
    grid = np.linspace(0, duration, QI_SETTINGS['samples'])
    out = {}
    for cell, spikes in spikes_by_cell.items():
        curves = []
        for b, e in bounds:
            x = (spikes[(spikes >= b) & (spikes < e)] - b) / fs
            curves.append(_smooth(x[x < duration], grid, QI_SETTINGS['bandwidth_s']))
        curves = np.asarray(curves)
        noise = np.mean(np.var(curves, axis=1))
        out[int(cell)] = float(np.var(curves.mean(axis=0)) / noise) if noise > 0 else np.nan
    return out


RF_EDGE_MAP_FRACTION = 0.3   # RF blob = connected part of the RMS map above this fraction of its peak


def _stimulus_half_widths(sel, ds, folder, quality):
    """Half-widths (um) of the noise stimulus in v2 map coordinates = extent of the STA kernel.

    kernel.npy is H x W stimulus pixels, centred on 0, v2 grid spacing dx um per pixel
    (checked: kernel peak = RMS peak; 14_08 maps drop to nothing exactly at +-800 um).
    Stimuli are not always square (15_10_2026: 1024 x 800 px). None if no kernel found.
    """
    if 'sta_path' not in sel:
        return None
    dx = float(np.median(np.abs(np.diff(ds['x'].values.astype(float)))))
    for c in np.where(np.isfinite(quality))[0]:
        kfile = Path(folder) / str(sel['sta_path'].isel(cell_index=int(c)).item()).replace('\\', '/')
        if kfile.is_file():
            shape = np.load(kfile, mmap_mode='r').shape
            return shape[2] * dx / 2, shape[1] * dx / 2, dx
    return None


def _rf_position_and_edge(sel, ds, folder, quality, fallback_x, fallback_y):
    """RF centre (um) and how far the RF stays inside the noise stimulus (um; <= 0 = cut off).

    Centre = peak of the v2 RMS map (same point the STA videos are centred on). The stored
    'positions' are offset for non-square stimuli (15_10: ~200 um in y), so only a fallback.

    Two edge checks, the smaller distance is kept as rf_edge_margin_um:
      outline: centre outline (center_outline_um, x = cx + r cos a, y = cy + r sin a,
               checked against the cm maps) distance to the edge. Misses many cut RFs,
               because the outline is fitted to the visible part and stops just short.
      map:     the RF blob (connected part of the RMS map above RF_EDGE_MAP_FRACTION of
               its peak, containing the peak) distance to the last stimulus pixel
               (0 = blob touches the edge).
    surround_outline_um is NOT a full radius (median ~5 um), so no surround check.
    """
    from scipy import ndimage
    xs, ys = ds['x'].values.astype(float), ds['y'].values.astype(float)
    cx, cy = np.array(fallback_x, float), np.array(fallback_y, float)
    n = len(quality)
    halves = _stimulus_half_widths(sel, ds, folder, quality)
    out = {}
    map_edge = np.full(n, np.nan)
    if 'rms' in sel:
        for i in np.where(np.isfinite(quality))[0]:
            m = np.abs(np.nan_to_num(sel['rms'].isel(cell_index=int(i)).values.astype(float)))
            if m.max() <= 0:
                continue
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            cx[i], cy[i] = xs[ix], ys[iy]
            if halves:
                hx, hy, dx = halves
                labels, _ = ndimage.label(m > RF_EDGE_MAP_FRACTION * m.max())
                by, bx = np.where(labels == labels[iy, ix])
                map_edge[i] = min(hx - dx - np.abs(xs[bx]).max(), hy - dx - np.abs(ys[by]).max())
    if halves:
        hx, hy, dx = halves
        out.update(stimulus_half_width_x_um=np.full(n, hx), stimulus_half_width_y_um=np.full(n, hy),
                   rf_map_edge_um=map_edge)
        edge = map_edge.copy()
        if 'center_outline_um' in sel:
            angle = np.deg2rad(ds['degree'].values.astype(float))
            r = sel['center_outline_um'].transpose('cell_index', 'degree').values.astype(float)
            margin = np.minimum(hx - np.abs(cx[:, None] + r * np.cos(angle)), hy - np.abs(cy[:, None] + r * np.sin(angle)))
            with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter('ignore', RuntimeWarning)
                outline = np.where(np.isfinite(r).any(axis=1), np.nanmin(margin, axis=1), np.nan)
            out['rf_outline_edge_um'] = outline
            edge = np.fmin(edge, outline)
        out['rf_edge_margin_um'] = edge
    return cx, cy, out


def refresh_rf(master_path, rec, cfg, nas_root, log=print):
    """Re-read only the receptive-field values of one recording (no spike re-extraction)."""
    folder = Path(nas_root) / f'zebrafish_{rec}' / cfg['phase']
    rows, rf_row = _rf_features(rec, cfg, folder, log)
    con = open_for_writing(master_path)
    try:
        con.execute("delete from features where recording=? and context like 'rf:%'", (rec,))
        con.execute('delete from rf_sources where recording=?', (rec,))
        con.executemany('insert into features values (?,?,?,?,?,?,?)', rows)
        con.execute('insert into rf_sources values (?,?,?,?,?,?)', rf_row)
        bump_revision(con)
        con.commit()
    finally:
        con.close()
    return len(rows)


def _rf_features(rec, cfg, folder, log):
    """Read receptive-field values for one recording. Returns (feature rows, rf_source row)."""
    import xarray as xr
    rf = cfg['rf']
    path = folder / rf['file']
    channel, index = rf['channel'], int(rf['stimulus_index'])
    context = f'rf:{index}:{channel}'
    match = re.match(r'(\d+)px', channel)
    pixel = int(match.group(1)) if match else None
    rows = []
    with xr.open_dataset(path) as ds:
        channels = [str(c) for c in ds['channel'].values]
        if channel not in channels:
            raise ValueError(f'{rec}: channel {channel!r} not in {path} (has {channels})')
        sel = ds.sel(channel=channel)
        cells = ds['cell_index'].values.astype(int)
        get = lambda name: sel[name].values.astype(float) if name in sel else np.full(len(cells), np.nan)
        quality = get('quality')
        tilt = get('tilt')
        angle = get('angle')
        finite_angle = angle[np.isfinite(angle)]
        angle_unit = 'radians' if finite_angle.size and np.nanmax(np.abs(finite_angle)) <= 2 * np.pi + 1e-6 else 'degrees'
        with np.errstate(divide='ignore', invalid='ignore'):
            elongation = np.where(tilt > 0, 1.0 / tilt, np.nan)
        scalars = dict(rf_quality=quality, center_size_mm2=get('center_size_mm2'),
                       surround_size_mm2=get('surround_size_mm2'), elongation=elongation,
                       orientation_deg=angle, tir=get('tir'), center_shift=get('center_shift'),
                       center_std=get('center_std'), surround_std=get('surround_std'))
        if 'positions' in sel:
            # positions are indices into the v2 map grid (x, y) -> convert to micrometres
            pos = sel['positions'].transpose('cell_index', 'pos_dim').values.astype(int)
            dims = [str(p) for p in ds['pos_dim'].values]
            xs, ys = ds['x'].values.astype(float), ds['y'].values.astype(float)
            ix = np.clip(pos[:, dims.index('x')], 0, len(xs) - 1)
            iy = np.clip(pos[:, dims.index('y')], 0, len(ys) - 1)
            scalars['rf_x_um'], scalars['rf_y_um'], edges = _rf_position_and_edge(sel, ds, folder, quality, xs[ix], ys[iy])
            scalars.update(edges)
        units = dict(orientation_deg=angle_unit, center_size_mm2='mm2', surround_size_mm2='mm2',
                     elongation='1/tilt', rf_x_um='um', rf_y_um='um', rf_edge_margin_um='um', rf_outline_edge_um='um', rf_map_edge_um='um',
                     stimulus_half_width_x_um='um', stimulus_half_width_y_um='um')
        sta = sel['sta_single_pixel'].transpose('cell_index', 'time_max').values.astype(float) \
            if 'sta_single_pixel' in sel else None
        sta_time = ds['time_max'].values.astype(float) if sta is not None else None
        sta_meta = dict(dims=['time_max'], coordinates={'time_max': dict(
            values=sta_time.tolist() if sta_time is not None else [],
            attrs=dict(units=str(ds['time_max'].attrs.get('units', 'ms'))))},
            source='sta_single_pixel')
        for i, cell in enumerate(cells):
            if not np.isfinite(quality[i]):
                continue
            for name, values in scalars.items():
                v = float(values[i])
                if np.isfinite(v):
                    rows.append((rec, int(cell), context, name, v, encode(np.array([v])),
                                 dumps(dict(attrs=dict(units=units.get(name, ''))))))
            if sta is not None and np.isfinite(sta[i]).any():
                rows.append((rec, int(cell), context, 'temporal_sta', None, encode(sta[i]), dumps(sta_meta)))
        stat = path.stat()
        provenance = dict(file=dict(path=str(path), size=stat.st_size, mtime=stat.st_mtime),
                          channel=channel, stimulus_index=index, attrs={k: str(v) for k, v in ds.attrs.items()},
                          angle_unit_inferred=angle_unit)
    edge = scalars.get('rf_edge_margin_um')
    edge_note = '' if edge is None else \
        f', {int(np.sum((quality > 10) & (edge <= 0)))} of those cut off by the stimulus edge'
    log(f'    RF: {context}: {sum(np.isfinite(quality))} cells with quality, '
        f'{int(np.sum(quality > 10))} > 10{edge_note}, angle unit {angle_unit}')
    return rows, (rec, index, channel, 'complete', pixel, dumps(provenance))


def extract_recording(master_path, rec, cfg, nas_root, expected_bar_repeats=10, log=print):
    """Read one recording from the NAS and (re)write it into the master file."""
    import polars as pl
    from polarspike import Overview

    folder = Path(nas_root) / f'zebrafish_{rec}' / cfg['phase']
    log(f'== {rec} ({folder})')
    recording = Overview.Recording.load(folder / 'overview')
    stim = recording.stimulus_df.sort_values('stimulus_index').reset_index(drop=True)
    cells = (recording.spikes_df.drop_duplicates('cell_index')[['cell_index', 'centres_x', 'centres_y']]
             .sort_values('cell_index').reset_index(drop=True))
    cell_ids = cells.cell_index.astype(int).to_numpy()
    fs_all = stim.sampling_freq.astype(float).unique()
    log(f'    {len(cell_ids)} cells, {len(stim)} stimuli, sampling rate {fs_all.tolist()} Hz')

    # all spikes of the recording, split by cell (absolute sample numbers, sorted)
    table = pl.read_parquet(folder / 'alldata.parquet', columns=['cell_index', 'times']).sort(['cell_index', 'times'])
    c = table['cell_index'].to_numpy()
    t = table['times'].to_numpy().astype(np.int64)
    del table
    starts = np.searchsorted(c, cell_ids, side='left')
    ends = np.searchsorted(c, cell_ids, side='right')
    per_cell = {int(cell): t[s:e] for cell, s, e in zip(cell_ids, starts, ends)}

    roles = stimulus_roles(cfg, stim)
    stimuli_rows, timing_rows, obs_rows = [], [], []
    for row in stim.itertuples():
        idx = int(row.stimulus_index)
        role, group, selected = roles[idx]
        fs = float(row.sampling_freq)
        logic = int(row.stimulus_repeat_logic)
        rel = np.asarray(row.trigger_fr_relative, dtype=np.int64)
        nr_repeats = int(row.nr_repeats)
        inserted = []
        if role == 'moving_bar' and expected_bar_repeats:
            rel, inserted = repair_triggers(rel, logic, expected_bar_repeats)
            if inserted:
                nr_repeats = max(nr_repeats, (len(rel) - 1) // logic)
                log(f'    stimulus {idx} ({row.stimulus_name}): inserted {len(inserted)} dropped trigger(s) '
                    f'at position(s) {inserted} -> {nr_repeats} repeats')
        bounds = repeat_bounds(row.begin_fr, row.end_fr, rel, logic, nr_repeats, bool(inserted))
        if role == 'moving_bar' and expected_bar_repeats and len(bounds) != expected_bar_repeats:
            log(f'    WARNING stimulus {idx}: moving bar has {len(bounds)} repeats, expected {expected_bar_repeats}')
        status = 'complete' if bounds else 'no complete repeats'
        meta = dict(begin_fr=int(row.begin_fr), end_fr=int(row.end_fr), nr_repeats=nr_repeats,
                    original_nr_repeats=int(row.nr_repeats), sampling_freq=fs, stimulus_repeat_logic=logic,
                    stimulus_repeat_sublogic=int(row.stimulus_repeat_sublogic),
                    repeats_found=len(bounds), inferred_triggers=inserted,
                    inferred_final_trigger_count=0)
        kind = 'raw+QI' if role in FFF else 'raw'
        stimuli_rows.append((rec, idx, str(row.stimulus_name).strip(), role, group, selected, status, kind,
                             nr_repeats, dumps(meta)))
        timing_rows.append((rec, idx, encode(rel), dumps(dict(padding=dict(inferred_count=len(inserted),
                                                                         positions=inserted)))))
        b, e = int(row.begin_fr), int(row.end_fr)
        spikes = {cell: s[(s >= b) & (s <= e)] for cell, s in per_cell.items()}
        qis = flash_qis(spikes, bounds, fs, role) if role in FFF else {}
        if role in FFF:
            meta['qi'] = QI_SETTINGS
            stimuli_rows[-1] = stimuli_rows[-1][:-1] + (dumps(meta),)
        for cell in cell_ids:
            s = spikes[int(cell)]
            obs_rows.append((rec, idx, int(cell), int(len(s)), qis.get(int(cell)), encode(s)))
        if role in FFF or role == 'moving_bar':
            n_qi = sum(np.isfinite(list(qis.values()))) if qis else 0
            log(f'    stimulus {idx:>2} {row.stimulus_name:<32} role={role:<10} group={group} '
                f'repeats={len(bounds)}' + (f' QI for {n_qi} cells' if role in FFF else ''))

    feature_rows, rf_row = [], None
    if cfg.get('rf'):
        feature_rows, rf_row = _rf_features(rec, cfg, folder, log)

    pixel = rf_row[4] if rf_row else None
    rec_row = (rec, cfg['phase'], str(folder), len(cell_ids), float(fs_all[0]), cfg.get('region'), pixel,
               cfg.get('notes', ''), dumps(cfg), datetime.datetime.now().isoformat(timespec='seconds'))
    cell_rows = [(rec, int(r.cell_index), f'{rec}:cell_{int(r.cell_index)}', float(r.centres_x), float(r.centres_y))
                 for r in cells.itertuples()]

    con = open_for_writing(master_path)
    try:
        delete_recording(con, rec)
        con.execute('insert into recordings values (?,?,?,?,?,?,?,?,?,?)', rec_row)
        con.executemany('insert into cells values (?,?,?,?,?)', cell_rows)
        con.executemany('insert into stimuli values (?,?,?,?,?,?,?,?,?,?)', stimuli_rows)
        con.executemany('insert into qi_timing values (?,?,?,?)', timing_rows)
        con.executemany('insert into observations values (?,?,?,?,?,?)', obs_rows)
        con.executemany('insert into features values (?,?,?,?,?,?,?)', feature_rows)
        if rf_row:
            con.execute('insert into rf_sources values (?,?,?,?,?,?)', rf_row)
        bump_revision(con)
        con.commit()
    finally:
        con.close()
    log(f'    written: {len(obs_rows)} observations, {len(feature_rows)} RF feature values')
    del per_cell, c, t, recording
