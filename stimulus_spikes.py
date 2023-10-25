import polars as pl
import numpy as np


def load_cells(cell_index, file, waveforms=False):
    df = pl.scan_parquet(file)
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
        range(trigger_end.shape[0] - 1), rel_trigger[:-1], trigger_sub
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


def trigger_sublogic(df, stimulus, time="seconds"):
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
