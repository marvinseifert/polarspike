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
              'tir', 'center_shift', 'center_std', 'surround_std', 'rf_x_um', 'rf_y_um')

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
            pos = sel['positions'].transpose('cell_index', 'pos_dim').values.astype(float)
            dims = [str(p) for p in ds['pos_dim'].values]
            scalars['rf_x_um'], scalars['rf_y_um'] = pos[:, dims.index('x')], pos[:, dims.index('y')]
        units = dict(orientation_deg=angle_unit, center_size_mm2='mm2', surround_size_mm2='mm2',
                     elongation='1/tilt', rf_x_um='um', rf_y_um='um')
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
    log(f'    RF: {context}: {sum(np.isfinite(quality))} cells with quality, '
        f'{int(np.sum(quality > 10))} > 10, angle unit {angle_unit}')
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
