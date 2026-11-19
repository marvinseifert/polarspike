"""Build / extend the master data file (retina_master.sqlite) used for RGC clustering.

Run ONCE per recording. Afterwards clustering reads only the master file.

    cd ~/PycharmProjects/polarspike
    <rf-analysis env python> polarspike/Laura_scripts/build_master_file.py

Which stimuli are used is set in master_stimulus_selection.json (next to this script).
Recordings already in the master are skipped unless listed in FORCE.
The file is built locally (SQLite must not be written over the network), then copied,
together with the selection file, to NAS_DIR.
Region / notes in the selection file are re-synced every run without re-extracting.
"""
from pathlib import Path
import json
import shutil
import sqlite3
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # polarspike repository root, for `from polarspike import Overview`

import clustering_data_extraction_all_functions as storage  # noqa: E402

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
SELECTION_FILE = HERE / 'master_stimulus_selection.json'
LOCAL_MASTER = HERE / 'retina_master.sqlite'          # clustering reads this copy
NAS_DIR = Path('/mnt/mea_nas/Laura/RGC_clustering/master_data_extraction')
COPY_TO_NAS = True
RECORDINGS = None      # None = every recording in the selection file; or e.g. ['14_08_2026']
FORCE = []             # recordings to re-extract even if already in the master


def copy_to_nas(local, nas_dir):
    """Never open SQLite on the NAS (CIFS file locking hangs): snapshot locally, then plain file copy."""
    nas_dir.mkdir(parents=True, exist_ok=True)
    snapshot = local.with_name(local.name + '.snapshot')
    src = sqlite3.connect(local)
    dst = sqlite3.connect(snapshot)
    try:
        src.backup(dst)          # consistent copy even if the file is open elsewhere
    finally:
        dst.close()
        src.close()
    temp = nas_dir / (local.name + '.partial')
    shutil.copyfile(snapshot, temp)
    temp.replace(nas_dir / local.name)
    snapshot.unlink()
    shutil.copy2(SELECTION_FILE, nas_dir / SELECTION_FILE.name)
    print(f'Copied master + selection file to {nas_dir}')


def main():
    selection = json.loads(SELECTION_FILE.read_text())
    wanted = RECORDINGS or list(selection['recordings'])
    done = storage.recordings_in(LOCAL_MASTER)
    for rec in wanted:
        if rec not in selection['recordings']:
            raise KeyError(f'{rec} is not in {SELECTION_FILE.name}')
        if rec in done and rec not in FORCE:
            print(f'{rec}: already in master, skipped (add to FORCE to re-extract)')
            continue
        start = time.time()
        storage.extract_recording(LOCAL_MASTER, rec, selection['recordings'][rec], selection['nas_root'],
                                  selection.get('expected_bar_repeats', 10))
        print(f'    {rec} took {time.time() - start:.0f} s', flush=True)
    storage.update_recording_info(LOCAL_MASTER, selection)
    with storage.MasterDataset(LOCAL_MASTER) as db:
        print('\nMaster file contents:')
        print(db.table('recordings')[['recording', 'phase', 'n_cells', 'sampling_freq', 'noise_pixel_size', 'region']]
              .to_string(index=False))
    print(f'Size: {LOCAL_MASTER.stat().st_size / 1e6:.0f} MB ({LOCAL_MASTER})')
    if COPY_TO_NAS:
        copy_to_nas(LOCAL_MASTER, NAS_DIR)


if __name__ == '__main__':
    main()
