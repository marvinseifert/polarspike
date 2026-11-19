# RF axis vs moving-bar orientation (19 Nov 2026)

Does a cell's RF long axis predict which bar orientation it prefers?
Figures and a copy of every script: NAS
`/mnt/mea_nas/Laura/RGC_clustering/clustering_runs/2026-11-19_RF_DSI_OSI_90faade5/analysis/`.

## Data
`build_rf_vs_bar_axis_table.py` makes `rf_vs_bar_axis.csv` (one row per cell, 607 cells) from the
final clustering cache (fit `90faade5`, 9 recordings, 12 px at 5 µm/px, surround dropped,
edge-cut RFs left out). All angles are in the door view (polar-plot frame, 0° = up, clockwise).
`rf_axis` = RF long axis; `motion_axis` = preferred bar *motion* axis; the preferred bar's long
edge is `motion_axis + 90`.

Run from `Laura_scripts`, Python = the rf-analysis poetry env:

    python rf_bar_alignment/build_rf_vs_bar_axis_table.py rf_vs_bar_axis.csv
    python rf_bar_alignment/<plot script>.py rf_vs_bar_axis.csv out.png   # scatter_grid takes 2 outputs

## Plots (in the order they were made)
| Script | Figure |
|---|---|
| `plot_rf_vs_bar_axis.py` | RF axis vs bar motion axis, 12 vs 32 px, by recording, by elongation threshold |
| `plot_rf_axis_roses.py` | Do RF axes cluster at particular angles (rose plots per recording) |
| `plot_os_elongated_alignment.py` | Elongated (> 1.2) + OSI > 0.15 cells: scatter + histogram |
| `plot_alignment_vs_elongation(_os).py` | Score vs elongation bins, all cells / OS cells, split 12/32 px |
| `plot_os_alignment_hist.py` | OS cells, three elongation groups, bars + histograms |
| `plot_os_scatter_grid.py` | Scatter grid at 4 OSI thresholds x 3 elongation groups, + robustness over cut-offs (CSV) |
| `plot_os_summary.py` | Summary: scatter with circular r / r², angle histogram, score by group |
| `plot_os_elong_bins.py` | Score in 0.05 elongation steps |
| `plot_os_elong_groups.py` | Score per cell (dots) with mean ± 95% CI, three groups |
| `plot_os_angle_hist_groups.py` | **Latest / preferred**: angle histograms (4 x 22.5° bins, chance 25%) per group |

Score = mean cos 2(angle between RF long axis and preferred bar): +1 aligned, 0 unrelated, −1 across.
p = shuffle test (bar orientations re-assigned among cells); "within recording" only among cells of
the same recording.

## Main results (OSI > 0.10)
* Slightly elongated (1.05–1.2, n = 165) and elongated (≥ 1.2, n = 104) cells prefer bars lying
  along the RF more than chance: 35–36 % within 22.5° vs 25 % (p = 0.004, 0.02); elongated cells
  are less often across (14 %). Round (< 1.05, n = 20) too few to say.
* Effect is weak: circular r = 0.14 (r² = 0.02); alignment score +0.20 (p < 0.001, within
  recording p = 0.007). No steady rise with elongation (rho = +0.05). Same direction at every OSI
  threshold (0–0.25) and elongation cut-off tested.
* Per-cell cos 2(angle) dot plots look like "no effect" because the score piles up at ±1 even by
  chance; the angle histogram shows it better.
* OSI does not increase with elongation.

## aligned_recordings/
Recordings with cone-image alignment in rf_analysis (14/05, 15/05, 22/07, 14/08, 15/10; 07/05
and 07/08 not extracted). Adds a moving-bar QI (Baden repeat reliability) and joins the
rf_analysis cone-row table (Box `Figures/RF_cone_row_angle_prediction/rf_vs_cone_row_angle_all_recordings_cells.csv`).

    python rf_bar_alignment/aligned_recordings/compute_osi_elongation_aligned.py aligned_osi_elong.csv
    python rf_bar_alignment/aligned_recordings/plot_osi_vs_elongation_aligned.py aligned_osi_elong.csv out.png
    python rf_bar_alignment/aligned_recordings/plot_axes_rf_bar_row.py aligned_osi_elong.csv out.png  # run from aligned_recordings/

Results: OSI vs elongation, no relation in reliable responders; RF axis, bar axis and radiating
cone row not linked cell by cell when 12 and 32 px are pooled (the 32 px recordings alone do
show an RF/bar link); 22/07 has one orientation shared by the whole recording.
