"""Make receptive field (STA) videos from the v2 noise analysis, one mp4 per cell.

Rewritten 2026-11-19 (the October 2026 script was lost). For every cell of the master
file whose RF quality passes RF_QUALITY_MIN, it cuts a WINDOW_UM square around the cell's
receptive field out of its spike-triggered average (kernel.npy) and writes the frames from
T_FROM_MS to T_TO_MS (0 = spike) as a grey video: bright = light before the spike, dark =
dark before the spike, mid-grey = no preference.

Orientation and scale follow noise_data_v2.nc: kernel.npy is in stimulus coordinates
(x = column, y = row, +y down), exactly like the v2 rms / cm_most_important maps, so the
videos line up with the RF maps in the explorer. The window is centred on the peak of the
v2 RMS map (fallback: covariance map), converted to kernel pixels with the v2 grid
spacing (micrometres per stimulus pixel).

File names match what build_cluster_explorer.py expects:
    <recording>_<channel>_cell<N>_<um>um_<fps>fps_from<a>to<b>ms.mp4

Run from the polarspike repository folder with the rf-analysis environment:
    <rf-analysis python> polarspike/Laura_scripts/make_sta_videos.py
Needs ffmpeg on the PATH. Existing videos are skipped unless FORCE = True.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clustering_data_extraction_all_functions as storage  # noqa: E402

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
MASTER_FILE = HERE / 'retina_master.sqlite'
OUT_DIR = Path('/mnt/mea_nas/Laura/RGC_clustering/sta_videos_v2')
RECORDINGS = None          # None = every recording with RF data in the master; or a list
RF_QUALITY_MIN = 10.0      # only cells whose RF quality is above this (None = all cells with RF data)
WINDOW_UM = 500            # width of the square cut around the RF
T_FROM_MS, T_TO_MS = -400, 30
FPS = 10                   # playback speed (frames per second); one frame = one STA bin
OUT_PX = 252               # video width/height in pixels (even number)
CONTRAST_PERCENTILE = 99.5 # colour scale: this percentile of |STA - baseline| maps to white/black
FORCE = False


def kernel_path(phase_dir, sta_path):
    """sta_path in noise_data.nc is relative to the phase folder, with Windows separators."""
    return Path(phase_dir) / str(sta_path).replace('\\', '/')


def to_frames(win, out_px, percentile):
    """STA window (t, y, x) -> uint8 grey frames (t, out_px, out_px), one colour scale for all frames."""
    base = np.nanmedian(win)
    dev = win - base
    scale = np.nanpercentile(np.abs(dev), percentile)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    grey = np.clip(128 + 127 * dev / scale, 0, 255)
    grey = np.nan_to_num(grey, nan=128)
    n = grey.shape[1]
    idx = np.minimum((np.arange(out_px) * n / out_px).astype(int), n - 1)  # nearest neighbour: keeps checkers sharp
    return grey[:, idx][:, :, idx].astype(np.uint8)


def write_mp4(frames, path, fps):
    t, h, w = frames.shape
    tmp = path.with_suffix('.partial.mp4')
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{w}x{h}', '-r', str(fps),
           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', '-preset', 'medium',
           '-movflags', '+faststart', str(tmp)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    tmp.replace(path)


def videos_for_recording(db, rec, out_dir, log=print):
    import xarray as xr
    src = db.table('rf_sources')
    src = src[(src.recording == rec) & (src.status == 'complete')]
    if src.empty:
        return 0
    row = src.iloc[0]
    prov = json.loads(row.provenance)
    nc = Path(prov['file']['path'])
    channel = row.channel
    phase_dir = nc.parent.parent
    rf = db.rf_scalars()
    rf = rf[(rf.recording == rec) & (rf.context == f'rf:{int(row.stimulus_index)}:{channel}')]
    if RF_QUALITY_MIN is not None:
        rf = rf[rf.rf_quality > RF_QUALITY_MIN]
    cells = sorted(int(c) for c in rf.cell_index)
    i0 = None
    made = skipped = failed = 0
    with xr.open_dataset(nc) as ds:
        dsc = ds.sel(channel=channel)
        xs, ys = ds['x'].values.astype(float), ds['y'].values.astype(float)
        dx = float(np.median(np.abs(np.diff(xs))))          # micrometres per stimulus pixel (v2)
        tm = ds['time_max'].values.astype(float)
        i0 = int(np.argmin(np.abs(tm - T_FROM_MS)))
        i1 = int(np.argmin(np.abs(tm - T_TO_MS))) + 1
        half = int(round(WINDOW_UM / 2 / dx))
        for c in cells:
            name = f'{rec}_{channel}_cell{c}_{WINDOW_UM}um_{FPS}fps_from{T_FROM_MS}to{T_TO_MS}ms.mp4'
            out = out_dir / name
            if out.exists() and not FORCE:
                skipped += 1
                continue
            try:
                base = dsc['rms'].sel(cell_index=c).values.astype(float)
                if not np.isfinite(base).any() or np.nanmax(base) <= 0:
                    base = np.abs(dsc['cm_most_important'].sel(cell_index=c).values.astype(float))
                iy, ix = np.unravel_index(np.nanargmax(base), base.shape)
                kfile = kernel_path(phase_dir, dsc['sta_path'].sel(cell_index=c).item())
                k = np.load(kfile, mmap_mode='r')
                H, W = k.shape[1], k.shape[2]
                cx = int(round(xs[ix] / dx + W / 2))
                cy = int(round(ys[iy] / dx + H / 2))
                win = np.full((i1 - i0, 2 * half, 2 * half), np.nan)
                y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
                sy0, sy1, sx0, sx1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
                if sy1 > sy0 and sx1 > sx0:
                    win[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = k[i0:i1, sy0:sy1, sx0:sx1]
                write_mp4(to_frames(win, OUT_PX, CONTRAST_PERCENTILE), out, FPS)
                made += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log(f'    {rec} cell {c}: failed ({exc})')
    log(f'  {rec}: {made} videos made, {skipped} already there, {failed} failed '
        f'({len(cells)} cells, channel {channel}, {dx:g} um/px, frames {T_FROM_MS}..{T_TO_MS} ms)')
    return made


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with storage.MasterDataset(MASTER_FILE) as db:
        src = db.table('rf_sources')
        recs = sorted({r.recording for r in src.itertuples() if storage.rf_allowed(r.recording, r.channel)})
        if RECORDINGS:
            recs = [r for r in recs if r in RECORDINGS]
        print(f'STA videos -> {OUT_DIR}  ({len(recs)} recordings, RF quality > {RF_QUALITY_MIN})', flush=True)
        for rec in recs:
            start = time.time()
            videos_for_recording(db, rec, OUT_DIR, log=lambda m: print(m, flush=True))
            print(f'    took {time.time() - start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
