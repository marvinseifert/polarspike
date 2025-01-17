import polars as pl
import numpy as np


def load_cells(cell_index, file, waveforms=False):
    df = pl.scan_parquet(str(file))
    if not waveforms:
        df = df.select(pl.col("cell_index"), pl.col("times"))

    # Set up the filtering conditions
    # condition = pl.col("cell_index") == cell_index[0]
    # for value in cell_index[1:]:
    #     condition |= pl.col("cell_index") == value
    subset_df = df.filter(pl.col("cell_index").is_in(cell_index))
    # subset_df = df.filter(condition)
    subset_df = subset_df.collect(streaming=True)
    return subset_df.sort("cell_index")


def df_filter_stimulus(cell_index, stim_begin, stim_end, file, waveforms=False):
    subset_df = load_cells(cell_index, file, waveforms=waveforms)
    return subset_df.filter(
        (pl.col("times") > stim_begin) & (pl.col("times") <= stim_end)
    )


def load_stimulus(cell_index, stim_begin, stim_end, file):
    df_times = df_filter_stimulus(cell_index, stim_begin, stim_end, file)
    return df_times.partition_by(by="cell_index", as_dict=False, include_key=False)


def load_triggered(
    cell_index,
    stim_begin,
    stim_end,
    trigger_start,
    trigger_end,
    logic,
    file,
    waveforms=False,
):
    subset_df = df_filter_stimulus(cell_index, stim_begin, stim_end, file, waveforms)
    subset_df = subset_df.with_columns(
        pl.col("times").sub(stim_begin).alias("times_relative")
    )
    rel_trigger = map_triggers(np.arange(trigger_start.shape[0]), logic)
    trigger_sub = sub_trigger(trigger_start, rel_trigger, logic)

    dfs = []
    repeat = 0
    for stimulus, rel_t, sub_t in zip(
        range(trigger_end.shape[0]), rel_trigger, trigger_sub
    ):
        times = subset_df.filter(
            (pl.col("times_relative") > trigger_start[stimulus])
            & (pl.col("times_relative") <= trigger_end[stimulus])
        )
        times = times.with_columns(trigger=pl.lit(rel_t))
        times = times.with_columns(repeat=pl.lit(repeat))

        times = times.with_columns(
            pl.col("times_relative").sub(sub_t).alias("times_triggered")
        )
        if rel_t == logic - 1:
            repeat += 1

        dfs.append(times)
    df = pl.concat(dfs)
    return df


def load_triggered_lazy(files_dict, filter_dict, time="seconds"):
    # Next, we create a dictionary with all the filters applied to the lazy dataframes
    combined_lazy_dfs = {}
    for key, path in files_dict.items():
        filters = filter_dict.get(key, {})
        filtered_dfs = []
        for stim_id, filt in filters.items():
            # Apply filter and add a 'filter_id' column to track the filter origin
            df_filtered = (
                pl.scan_parquet(path)
                .filter(
                    (pl.col("times") > filt["start"])
                    & (pl.col("times") <= filt["end"])
                    & pl.col("cell_index").is_in(filt["cell_indices"])
                )
                .with_columns(pl.lit(stim_id).alias("stimulus_index"))
                .with_columns(pl.lit(key).alias("recording"))
            )
            # Create new columns for trigger, repeat, times_relative and times_triggered
            df_filtered = df_filtered.with_columns(times_relative=pl.lit(0))
            df_filtered = df_filtered.with_columns(trigger=pl.lit(0))
            df_filtered = df_filtered.with_columns(repeat=pl.lit(0))
            df_filtered = df_filtered.with_columns(times_triggered=pl.lit(0))
            # Add times relative to the stimulus onset
            df_filtered = df_filtered.with_columns(
                times_relative=pl.col("times") - filt["start"]
            )
            rel_trigger = map_triggers(
                np.arange(filt["trigger"][0].shape[0]),
                filt["stim_repeat_logic"][0],
            )
            trigger_sub = sub_trigger(
                filt["trigger"][0],
                rel_trigger,
                filt["stim_repeat_logic"][0],
            )
            trigger_start = filt["trigger"][0]
            trigger_end = filt["trigger"][0][1:]
            # Align the spikes with the repeats and triggers (still lazy)
            repeat = 0
            for trigg_id, rel_t, sub_t in zip(
                range(trigger_end.shape[0]),
                rel_trigger,
                trigger_sub,
            ):
                df_filtered = df_filtered.with_columns(
                    times_triggered=pl.when(
                        pl.col("times_relative") > trigger_start[trigg_id],
                        pl.col("times_relative") <= trigger_end[trigg_id],
                    )
                    .then(pl.col("times_relative") - sub_t)
                    .otherwise(pl.col("times_triggered"))
                )
                df_filtered = df_filtered.with_columns(
                    trigger=pl.when(
                        pl.col("times_relative") > trigger_start[trigg_id],
                        pl.col("times_relative") <= trigger_end[trigg_id],
                    )
                    .then(rel_t)
                    .otherwise(pl.col("trigger"))
                )
                df_filtered = df_filtered.with_columns(
                    repeat=pl.when(
                        pl.col("times_relative") > trigger_start[trigg_id],
                        pl.col("times_relative") <= trigger_end[trigg_id],
                    )
                    .then(repeat)
                    .otherwise(pl.col("repeat"))
                )
                if rel_t == filt["stim_repeat_logic"] - 1:
                    repeat += 1
            if time == "seconds":
                df_filtered = df_filtered.with_columns(
                    times=pl.col("times").truediv(filt["sampling_freq"])
                )
                df_filtered = df_filtered.with_columns(
                    times_relative=pl.col("times_relative").truediv(
                        filt["sampling_freq"]
                    )
                )
                df_filtered = df_filtered.with_columns(
                    times_triggered=pl.col("times_triggered").truediv(
                        filt["sampling_freq"]
                    )
                )
            filtered_dfs.append(df_filtered)

        if filtered_dfs:
            # Concatenate all filtered DataFrames for the current file
            combined_df = pl.concat(filtered_dfs, how="vertical").unique()
            combined_lazy_dfs[key] = combined_df
        else:
            # If no filters are defined for the file, assign an empty DataFrame
            combined_lazy_dfs[key] = pl.LazyFrame(schema=pl.Schema([]))

    return combined_lazy_dfs


def stimulus_relative(cell_index, stim_begin, stim_end, file, waveforms=False):
    subset_df = df_filter_stimulus(cell_index, stim_begin, stim_end, file, waveforms)
    subset_df = subset_df.with_columns(
        pl.col("times").sub(stim_begin).alias("times_relative")
    )
    return subset_df


def map_triggers(input_sequence, logic):
    # Map each element in the input sequence to its corresponding value in the output sequence
    output_sequence = [x % logic for x in input_sequence]
    return np.asarray(output_sequence[:-1])


def sub_trigger(trigger, rel_trigger, logic):
    trigger_sub = trigger[:-1][rel_trigger == 0]
    trigger_sub = np.repeat(trigger_sub, logic)
    return trigger_sub


def mean_trigger_times(df, stimulus, time="seconds"):
    if type(stimulus[0]) is str and stimulus[0] == "all":
        stimulus = df["stimulus_index"].unique().tolist()

    stim_df = df.query(f"stimulus_index=={stimulus}")
    new_trigger_int = np.repeat(
        np.mean(
            np.reshape(
                stim_df["trigger_int"].values[0][
                    : stim_df["nr_repeats"].values[0]  # Only get finished repeats
                    * stim_df["stimulus_repeat_logic"].values[0]
                ],
                (
                    stim_df["nr_repeats"].values[0],
                    stim_df["stimulus_repeat_logic"].values[0],
                ),
            ),
            axis=0,
        )
        / stim_df["stimulus_repeat_sublogic"].values[0],
        stim_df["stimulus_repeat_sublogic"].values[0],
    )
    if time == "seconds":
        new_trigger_int = new_trigger_int / stim_df["sampling_freq"].values[0]

    return new_trigger_int


def max_trigger_times(df, stimulus, time="seconds"):
    if type(stimulus[0]) is str and stimulus[0] == "all":
        stimulus = df["stimulus_index"].unique().tolist()

    stim_df = df.query(f"stimulus_index=={stimulus}")
    new_trigger_int = np.repeat(
        np.max(
            np.reshape(
                stim_df["trigger_int"].values[0][
                    : stim_df["nr_repeats"].values[0]  # Only get finished repeats
                    * stim_df["stimulus_repeat_logic"].values[0]
                ],
                (
                    stim_df["nr_repeats"].values[0],
                    stim_df["stimulus_repeat_logic"].values[0],
                ),
            ),
            axis=0,
        )
        / stim_df["stimulus_repeat_sublogic"].values[0],
        stim_df["stimulus_repeat_sublogic"].values[0],
    )
    if time == "seconds":
        new_trigger_int = new_trigger_int / stim_df["sampling_freq"].values[0]

    return new_trigger_int


def min_trigger_times(df, stimulus, time="seconds"):
    if type(stimulus[0]) is str and stimulus[0] == "all":
        stimulus = df["stimulus_index"].unique().tolist()

    stim_df = df.query(f"stimulus_index=={stimulus}")
    new_trigger_int = np.repeat(
        np.max(
            np.reshape(
                stim_df["trigger_int"].values[0][
                    : stim_df["nr_repeats"].values[0]  # Only get finished repeats
                    * stim_df["stimulus_repeat_logic"].values[0]
                ],
                (
                    stim_df["nr_repeats"].values[0],
                    stim_df["stimulus_repeat_logic"].values[0],
                ),
            ),
            axis=0,
        )
        / stim_df["stimulus_repeat_sublogic"].values[0],
        stim_df["stimulus_repeat_sublogic"].values[0],
    )
    if time == "seconds":
        new_trigger_int = new_trigger_int / stim_df["sampling_freq"].values[0]

    return new_trigger_int


def stim_duration(df, stimulus, time="seconds"):
    stim_duration = np.sum(mean_trigger_times(df, stimulus, time))
    return stim_duration
