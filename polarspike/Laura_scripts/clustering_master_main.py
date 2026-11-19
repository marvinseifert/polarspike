"""Select cells from retina_master.sqlite, then cluster and plot separately.

Save beside clustering_master_functions.py and the extraction functions script.
    import clustering_master_main as run
    dataset = run.load_data()
    result = run.cluster(dataset)
    run.plot(result, dataset)
"""

from pathlib import Path
import copy
import importlib
import clustering_master_functions as functions

MASTER_FILE = Path(__file__).resolve().parent / 'retina_master.sqlite'
CACHE_DIR = Path(__file__).resolve().parent / 'master_clustering_cache'

# None = all recordings in the master; otherwise a list of recording IDs.
RECORDINGS = None
# [2] = only group 2; [1,2] = both; [2,3] = second and third; 'all' = all.
FFF_GROUPS = [1]
FFF_GROUPS_BY_RECORDING = {'24_04_2026': 'all', '22_04_2026': 'all'}
# Optional, ONLY if you want these exported exception groups despite their
# original group number: {'24_04_2026': 'all', '22_04_2026': 'all'}
# No automatic substitute for a missing requested group. RF can still rescue
# its cells, in which case unavailable full-field features stay missing.
GROUP_MODE = 'average'  # 'average' (one row/cell) or 'separate' (one row/cell/group)

QUALITY = dict(
    # Settings of the 15 Oct 2026 example explorer (bd434141): both checks, RF > 10.
    mode='fff_and_rf',  # 'fff_or_rf', 'fff_and_rf', 'fff_only', 'rf_only', 'none'
    fff_threshold=0.5, fff_rule='any',  # 'any' or 'all' of the SAME group's 3 QIs
    rf_threshold=10.0,
    # RFs cut off by the edge of the noise stimulus (their size/elongation are wrong): the
    # RF blob (RMS map > 30 % of peak) or the centre outline comes within rf_edge_margin_um
    # of the edge (0 = touches it). Only happens where the stimulus is smaller than the
    # array (32 px, 1000 px ND and x12 noise).
    #   'drop_rf'   keep the cell (flash + bar responses, temporal STA), leave out its RF
    #               shape features (centre/surround size, elongation, orientation)
    #   'drop_cell' leave the cell out of the clustering completely
    #   None        use cut-off RFs as they are (October behaviour)
    rf_edge='drop_rf', rf_edge_margin_um=0.,
)
# True: only use recordings that contain eligible receptive fields, meaning at
# least RF_MIN_CELLS cells with RF quality above rf_threshold. Cells in those
# recordings are then selected by the usual QUALITY mode (use 'fff_only' to
# compare cells with and without RF within the same recordings).
RF_RECORDINGS_ONLY = False
RF_MIN_CELLS = 1

# Selection always uses ONLY requested FFF groups and the chosen RF source.
# Once a CELL passes, ALL selected measured responses/features are used, including
# low-QI groups and low-quality RF. Moving-bar QI is NEVER used for selection.

FEATURES = dict(scf=True, csteps=True, chirp=True, moving_bar=True,
                DSI=True, OSI=True, center_size_mm2=True, surround_size_mm2=False,  # surround: 32 px finds ~1/3 the size of 12 px
                elongation=True, orientation=False, temporal_sta=True)
EXCLUDE_MISSING_FEATURES = False
# False: use a Gaussian-mixture likelihood over available features, with no
# guessed RF values. Cells with no usable variable features remain unassigned.
# True: remove rows missing ANY enabled feature before fitting PCA/clustering.

# All exported bar presentations are averaged unless overridden by exact index.
MOVING_BAR_INDICES = 'all'
MOVING_BAR_INDICES_BY_RECORDING = {}  # e.g. {'12_08_2026': [11]}
# Choose one exact RF context per recording if you exported multiple RF sources.
# Print run.inspect_master() to see names. None means do not use RF for that rec.
RF_SOURCES = {}  # e.g. {'02_12_2025': 'rf:4:12px_20Hz_shuffle'}
# Projector pixel size changed on 20/04/2026: use 12 px noise only before that date and
# 32 px noise only after it. RF data breaking the rule is ignored (cells keep their flash
# and bar responses). None = use every RF source.
RF_PIXEL_RULE = dict(change_date='20_04_2026', before_px=12, after_px=32)
# RF size corrections (reversible; nothing in the master file changes):
# RF_SCALE_BY_PIXEL: real size of a noise pixel relative to what the RF analysis assumed.
#   {12: 1.25} = the 12 px noise was really 5 um per projector pixel, not 4 (makes 12 px and
#   32 px centre sizes match: medians 0.019 vs 0.020 mm2). {} = use sizes as stored.
RF_SCALE_BY_PIXEL = {12: 1.25}
# RF_SURROUND: 32 px noise finds much smaller surrounds than 12 px (median ~1/3), which no
#   scale explains. 'normalise_by_pixel' = compare each surround only with cells of the same
#   noise pixel size; 'drop' = leave surround size out; 'as_is' = use as stored.
RF_SURROUND = 'as_is'   # surround size is not used as a feature (FEATURES) since 19 Nov 2026

PROCESSING = dict(
    bandwidth_s=0.05, samples=1000,
    durations=dict(scf=32., csteps=40., chirp=35., moving_bar=40.),
    frames_per_direction=300, directions=8, direction_duration_s=5.,
    directions_deg=[0,180,90,270,45,135,315,225],
    direction_names=['up','down','right','left','up-right','down-right','up-left','down-left'],
    selectivity_bin_s=0.10,  # Change to .05 for 50 ms; previous latest setting was 100 ms.
    bar_peak_align=True, peak_target_s=2.5, baseline_percentile=20.,
    # How each bar direction is shifted to peak_target_s: 'first' = first clear peak
    # (bar leading edge; ON-OFF pairs stay in order), 'peak' = highest point (original;
    # gives three peaks in cluster averages). bar_align_fraction: a peak is 'clear' if it
    # rises at least this fraction of the highest point above baseline (0.3-0.5 all work).
    bar_align_method='first', bar_align_fraction=0.4,
    # STA is resampled only within measured support. Missing cells are never filled.
    sta_time_range_s=[-0.5,0.0], sta_samples=51,
    sta_time_unit_if_absent='ms',  # Explicit assumption; source coordinate units take priority.
    orientation_unit_if_absent='degrees',
)
ANALYSIS = dict(
    pca_variance=0.50,
    cluster_counts=[1,2,3,4,6,8,10,12,16,20,30,50],
    covariance_types=['diag','tied_diag'],
    n_initialisations=20, max_iterations=500, tolerance=1e-4,
    reg_covar=1e-5, random_seed=0,
    # How much each feature counts when cells are assigned to clusters (weighted
    # likelihood). Each column (response PC or RF number) counts 1 by default, so with
    # everything on, the 54 response columns outweigh the 5 RF columns (RF ~2 % of the decision).
    # rf_share: None = no balancing; 0.75 = RF features (centre, surround, elongation,
    # orientation, temporal STA) get 75 % of the total weight, responses (flash, steps,
    # chirp, bar, DSI, OSI) 25 %. Needs both kinds enabled in FEATURES.
    rf_share=None,
    # Extra weight per feature on top (1 = normal), e.g. {'elongation': 2.}
    feature_weights={},
)
PLOTTING = dict(min_cluster_cells=20, clusters_per_page=10,
                linewidth=0.8, contrast_alpha=0.18, show_interim=True,
                show=True, save=True, sta_baseline=0.5)


def config():
    return copy.deepcopy(dict(master_file=MASTER_FILE, cache_dir=CACHE_DIR,
        recordings=RECORDINGS, groups=FFF_GROUPS, groups_by_recording=FFF_GROUPS_BY_RECORDING,
        group_mode=GROUP_MODE, quality=QUALITY, features=FEATURES,
        rf_recordings_only=RF_RECORDINGS_ONLY, rf_min_cells=RF_MIN_CELLS,
        exclude_missing=EXCLUDE_MISSING_FEATURES, bar_indices=MOVING_BAR_INDICES,
        bar_indices_by_recording=MOVING_BAR_INDICES_BY_RECORDING, rf_sources=RF_SOURCES, rf_pixel_rule=RF_PIXEL_RULE,
        rf_scale_by_pixel=RF_SCALE_BY_PIXEL, rf_surround=RF_SURROUND,
        processing=PROCESSING, analysis=ANALYSIS, plotting=PLOTTING))


def inspect_master():
    return functions.inspect_master(config())


def load_data(force=False):
    return functions.load_data(config(), force=force)


def cluster(dataset=None, force=False):
    return functions.cluster(config(), dataset, force=force)


def plot(result=None, dataset=None):
    importlib.reload(functions)
    return functions.plot(config(), result, dataset)


if __name__=='__main__':
    print('In the console: import clustering_master_main as run')
    print('dataset = run.load_data(); result = run.cluster(dataset); run.plot(result, dataset)')
