"""Four clusterings of the same cells, to compare what responses and RFs each contribute.

    responses  only flash / steps / chirp / moving bar / DSI / OSI
    rf         only RF centre size, elongation, temporal STA (12 px sizes at 5 um/px, RF_SCALE_BY_PIXEL)
    rf_dsi_osi the RF features plus direction (DSI) and orientation (OSI) selectivity
    mixed      everything, weighted 75 % RF and 25 % responses (ANALYSIS['rf_share'])

Everything else comes from clustering_master_main.py (cells, quality, processing). All three
use the same cells: RFs cut off by the stimulus edge are left out completely (rf_edge =
'drop_cell'), because two of the runs depend on RF shape. Explorers are built for each.

Run from the polarspike repository folder with the rf-analysis environment:
    <rf-analysis python> polarspike/Laura_scripts/run_cluster_comparison.py [--no-open]
"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clustering_master_main as run  # noqa: E402

RESPONSES = ('scf', 'csteps', 'chirp', 'moving_bar', 'DSI', 'OSI')
RF = ('center_size_mm2', 'elongation', 'temporal_sta')   # surround size left out (unreliable in 32 px noise)
RUNS = {
    'responses': dict(features=RESPONSES, rf_share=None),
    'rf':        dict(features=RF, rf_share=None),
    'rf_dsi_osi': dict(features=RF + ('DSI', 'OSI'), rf_share=None),
    'mixed':     dict(features=RESPONSES + RF, rf_share=0.75),
}
RF_EDGE = 'drop_cell'


def main(open_browser=True):
    base_features = dict(run.FEATURES)
    base_analysis = dict(run.ANALYSIS)
    run.QUALITY = dict(run.QUALITY, rf_edge=RF_EDGE)
    dataset = run.load_data()          # same cells for all three runs
    keys = {}
    for name, setting in RUNS.items():
        run.FEATURES = {k: (k in setting['features']) for k in base_features}
        run.ANALYSIS = dict(base_analysis, rf_share=setting['rf_share'])
        print(f'\n=== {name} ===', flush=True)
        result = run.cluster(dataset)
        keys[name] = result['key']
        print(f'{name}: fit {result["key"]}, K = {result["model"]["k"]}', flush=True)
    for name, key in keys.items():
        cmd = [sys.executable, str(HERE / 'build_cluster_explorer.py'), key, '--title', f'RGC clusters: {name}']
        if not open_browser:
            cmd.append('--no-open')
        subprocess.run(cmd, check=True)
    print('\nFits:', keys, flush=True)
    return keys


if __name__ == '__main__':
    main(open_browser='--no-open' not in sys.argv)
