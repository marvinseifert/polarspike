import pandas as pd
import polars as pl
import numpy as np
import re


def parse_condition(s):
    match = re.match(r"(>=|<=|>|<|==)?\s*(\d*\.?\d+)", s.strip())
    if match:
        operator, number = match.groups()
        if operator and number:
            return operator, float(number)
        else:
            return "==", float(number)
    else:
        return "==", s


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


def filter_dataframe_complex(df, query_conditions):
    """
    Filters the DataFrame based on a list of condition dictionaries.

    Each dictionary in query_conditions represents a branch of the query
    (i.e. conditions joined with AND) and branches are combined with OR.

    For list values, the condition is expanded into an OR of equality checks.
    """
    branch_conditions = []
    for condition in query_conditions:
        condition_parts = []
        for col, val in condition.items():
            if isinstance(val, list):
                # Expand list into OR conditions: (col==val1 or col==val2 or ...)
                cond = (
                    "("
                    + " or ".join(
                        f"{col} {parse_condition(str(item))[0]} {repr(parse_condition(str(item))[1])}"
                        for item in val
                    )
                    + ")"
                )
            else:
                cond = f"{col} {parse_condition(str(val))[0]} {repr(parse_condition(str(val))[1])}"
            condition_parts.append(cond)
        # Combine conditions in a branch with AND
        branch_conditions.append("(" + " and ".join(condition_parts) + ")")

    # Combine branches with OR
    query_str = " or ".join(branch_conditions)
    return df.query(query_str)


def get_spikes(
    files_dict: dict,
    filter_dict: dict,
    original_df: pd.DataFrame,
    time: str,
    waveforms: bool,
    pandas: bool,
    carry: list[str],
):
    # Next, we create a dictionary with all the filters applied to the lazy dataframes
    df = load_triggered_lazy(files_dict, filter_dict, time)

    # Sort data in a meaningful way
    df = df.sort(
        ["cell_index", "repeat", "stimulus_index", "trigger", "times_triggered"]
    )
    # Carry columns if needed
    if carry:
        df = df.sort(["recording", "cell_index", "stimulus_index"])

        df = df.to_pandas().set_index(["recording", "cell_index", "stimulus_index"])

        for column in carry:
            df[column] = 0
            df[column] = df[column].astype(original_df[column].dtype)

        temp_df = original_df.set_index(["recording", "cell_index", "stimulus_index"])
        df.update(temp_df[carry])

        df = df.reset_index()
        df = pl.from_pandas(df)

    if pandas:
        return df.to_pandas()
    else:
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
            # add trigger to the filter_dict
            filter_dict[key][stim_id]["trigger_start"] = trigger_start
            filter_dict[key][stim_id]["trigger_end"] = trigger_end
            filter_dict[key][stim_id]["rel_trigger"] = rel_trigger
            filter_dict[key][stim_id]["trigger_sub"] = trigger_sub
            filtered_dfs.append(df_filtered)

            # load the filtered dataframes into memory
        if filtered_dfs:
            # Concatenate all filtered DataFrames for the current file
            combined_df = pl.concat(filtered_dfs, how="vertical").unique()
            combined_lazy_dfs[key] = combined_df
        else:
            # If no filters are defined for the file, assign an empty DataFrame
            combined_lazy_dfs[key] = pl.LazyFrame(schema=pl.Schema([]))

    lazy_frames = list(combined_lazy_dfs.values())

    # Collect all LazyFrames in parallel
    df = pl.concat(pl.collect_all(lazy_frames)).lazy()
    combined_dfs = []
    filtered_dfs = []
    for key, path in files_dict.items():
        filters = filter_dict.get(key, {})
        for stim_id, filt in filters.items():
            df_filtered = df.filter(
                (pl.col("stimulus_index") == stim_id) & (pl.col("recording") == key)
            )
            nr_repeats = len(filt["trigger"][0]) // filt["stim_repeat_logic"][0]
            repeats = np.repeat(
                np.arange(nr_repeats), filt["stim_repeat_logic"][0]
            )

            trig_df = pl.DataFrame({
                "start": filt["trigger"][0][:-1],
                "end": filt["trigger"][0][1:],
                "rel_t": filt["rel_trigger"],
                "sub_t": filt["trigger_sub"],
                "repeat": repeats
            }).sort("start").lazy()

            df_filtered = (
                df_filtered.lazy()
                .sort("times_relative")
                .join_asof(
                    trig_df,
                    left_on="times_relative",
                    right_on="start",
                    suffix="_trig"
                )
                .filter(pl.col("times_relative") <= pl.col("end"))
                .with_columns([
                    (pl.col("times_relative") - pl.col("sub_t")).alias("times_triggered"),
                    pl.col("rel_t").alias("trigger"),
                    pl.col("repeat_trig").alias("repeat"),
                ])
                .collect()
            )
            # drop the columns that are not needed
            df_filtered = df_filtered.drop(
                ["start", "end", "rel_t", "sub_t", "repeat_trig"]
            )

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

        # Concatenate all filtered DataFrames for the current file
        combined_dfs.append(pl.concat(filtered_dfs, how="vertical").unique())

    return pl.concat(combined_dfs)


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
