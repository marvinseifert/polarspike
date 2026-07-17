import pandas as pd
import polars as pl
import numpy as np
import re
from typing import Callable


def parse_condition(s: str) -> tuple[str, float | str]:
    """Split a filter string into a comparison operator and a value.

    Input:
        s (str): A condition such as "<=5", ">3.5", "7" or a bare label.

    Output:
        tuple[str, float | str]: (operator, value). The operator defaults to
        "==" when none is given; value is a float when the string is numeric,
        otherwise the original string.
    """
    match = re.fullmatch(r"(>=|<=|>|<|==)?\s*(\d*\.?\d+)", s.strip())
    if match:
        operator, number = match.groups()
        if operator and number:
            return operator, float(number)
        else:
            return "==", float(number)
    else:
        return "==", s


def spike_columns(file: str, waveforms: bool) -> list[str]:
    """Choose which columns to read from a spikes parquet.

    Input:
        file (str): Path to the spikes parquet.
        waveforms (bool): If True include the "waveforms" list column.

    Output:
        list[str]: ["cell_index", "times"], plus "waveforms" when requested.

    Raises:
        ValueError: if waveforms is True but the file has no "waveforms" column.
    """
    cols = ["cell_index", "times"]
    if waveforms:
        if "waveforms" not in pl.scan_parquet(str(file)).collect_schema().names():
            raise ValueError(f"waveforms requested but '{file}' has no 'waveforms' column")
        cols.append("waveforms")
    return cols


def load_cells(cell_index: list[int], file: str, waveforms: bool = False) -> pl.DataFrame:
    """Load the spikes of the given cells from a parquet file.

    Input:
        cell_index (list[int]): Cell indices to load.
        file (str): Path to the spikes parquet (columns cell_index, times).
        waveforms (bool): If True also load the "waveforms" list column.

    Output:
        pl.DataFrame: The matching spikes, sorted by cell_index.
    """
    df = pl.scan_parquet(str(file)).select(spike_columns(file, waveforms))
    subset_df = df.filter(pl.col("cell_index").is_in(cell_index))
    subset_df = subset_df.collect(engine="streaming")
    return subset_df.sort("cell_index")


def df_filter_stimulus(
    cell_index: list[int],
    stim_begin: int,
    stim_end: int,
    file: str,
    waveforms: bool = False,
) -> pl.DataFrame:
    """Load the spikes of the given cells that fall inside a time window.

    Input:
        cell_index (list[int]): Cell indices to load.
        stim_begin (int): Window start in frames, exclusive.
        stim_end (int): Window end in frames, inclusive.
        file (str): Path to the spikes parquet.
        waveforms (bool): If True also load the "waveforms" list column.

    Output:
        pl.DataFrame: Spikes with stim_begin < times <= stim_end.
    """
    subset_df = load_cells(cell_index, file, waveforms=waveforms)
    return subset_df.filter(
        (pl.col("times") > stim_begin) & (pl.col("times") <= stim_end)
    )


def load_stimulus(
    cell_index: list[int],
    stim_begin: int,
    stim_end: int,
    file: str,
    waveforms: bool = False,
) -> list[pl.DataFrame]:
    """Load window spikes split into one frame per cell.

    Input:
        cell_index (list[int]): Cell indices to load.
        stim_begin (int): Window start in frames, exclusive.
        stim_end (int): Window end in frames, inclusive.
        file (str): Path to the spikes parquet.
        waveforms (bool): If True also load the "waveforms" list column.

    Output:
        list[pl.DataFrame]: One frame per cell (cell_index column dropped).
    """
    df_times = df_filter_stimulus(cell_index, stim_begin, stim_end, file, waveforms)
    return df_times.partition_by(by="cell_index", as_dict=False, include_key=False)


def load_triggered(
    cell_index: list[int],
    stim_begin: int,
    stim_end: int,
    trigger_start: np.ndarray,
    trigger_end: np.ndarray,
    logic: int,
    file: str,
    waveforms: bool = False,
) -> pl.DataFrame:
    """Load window spikes and tag each with its trigger and repeat.

    Input:
        cell_index (list[int]): Cell indices to load.
        stim_begin (int): Window start in frames, exclusive.
        stim_end (int): Window end in frames, inclusive.
        trigger_start (np.ndarray): Trigger onsets relative to stim_begin.
        trigger_end (np.ndarray): Trigger ends, i.e. trigger_start shifted by one.
        logic (int): Number of triggers that make up one repeat.
        file (str): Path to the spikes parquet.
        waveforms (bool): If True also load the "waveforms" list column.

    Output:
        pl.DataFrame: Spikes with times_relative, trigger, repeat and
        times_triggered columns.
    """
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


def filter_dataframe_complex(
    df: pd.DataFrame, query_conditions: list[dict]
) -> pd.DataFrame:
    """Filter a dataframe with an OR of AND-branches.

    Input:
        df (pd.DataFrame): The dataframe to filter.
        query_conditions (list[dict]): Each dict is one branch; its items are
            joined with AND and branches are joined with OR. A list value
            expands to an OR of equality checks on that column.

    Output:
        pd.DataFrame: The rows matching the combined query.
    """
    branch_conditions = []
    for condition in query_conditions:
        condition_parts = []
        for col, val in condition.items():
            if isinstance(val, list):
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
        branch_conditions.append("(" + " and ".join(condition_parts) + ")")

    query_str = " or ".join(branch_conditions)
    return df.query(query_str)


def get_spikes(
    files_dict: dict[str, str],
    filter_dict: dict,
    original_df: pd.DataFrame,
    time: str,
    waveforms: bool,
    pandas: bool,
    carry: list[str],
) -> pd.DataFrame | pl.DataFrame:
    """Load triggered spikes for many recordings and optionally carry columns.

    Input:
        files_dict (dict[str, str]): recording name -> spikes parquet path.
        filter_dict (dict): recording -> stimulus -> filter parameters (start,
            end, cell_indices, trigger, stim_repeat_logic, sampling_freq).
        original_df (pd.DataFrame): Source of the carry columns, indexed by
            recording/cell_index/stimulus_index.
        time (str): "seconds" to convert frame counts, else frames are kept.
        waveforms (bool): If True also load the "waveforms" list column.
        pandas (bool): Return a pandas frame if True, else a polars frame.
        carry (list[str]): Extra columns to copy from original_df, or empty.

    Output:
        pd.DataFrame | pl.DataFrame: Triggered spikes, sorted, with the carried
        columns joined on when carry is given.
    """
    df = load_triggered_lazy(files_dict, filter_dict, time, waveforms)
    if df.height == 0:
        return df.to_pandas() if pandas else df

    df = df.sort(
        ["cell_index", "repeat", "stimulus_index", "trigger", "times_triggered"]
    )
    if carry:
        df = carry_columns(df, original_df, carry)

    if pandas:
        return df.to_pandas()
    else:
        return df


def carry_columns(
    df: pl.DataFrame, original_df: pd.DataFrame, carry: list[str]
) -> pl.DataFrame:
    """Copy extra columns from original_df onto the spikes.

    Input:
        df (pl.DataFrame): Triggered spikes with recording, cell_index and
            stimulus_index columns.
        original_df (pd.DataFrame): Source frame containing the carry columns.
        carry (list[str]): Column names to copy, matched on recording,
            cell_index and stimulus_index.

    Output:
        pl.DataFrame: df with the carry columns joined on.

    Raises:
        ValueError: if a requested carry column is not in original_df.
    """
    missing = [c for c in carry if c not in original_df.columns]
    if missing:
        raise ValueError(f"carry columns not found in original_df: {missing}")
    df = df.sort(["recording", "cell_index", "stimulus_index"])
    pdf = df.to_pandas().set_index(["recording", "cell_index", "stimulus_index"])
    for column in carry:
        pdf[column] = 0
        pdf[column] = pdf[column].astype(original_df[column].dtype)
    temp_df = original_df.set_index(["recording", "cell_index", "stimulus_index"])
    pdf.update(temp_df[carry])
    return pl.from_pandas(pdf.reset_index())


def load_triggered_lazy(
    files_dict: dict[str, str],
    filter_dict: dict,
    time: str = "seconds",
    waveforms: bool = False,
) -> pl.DataFrame:
    """Load spikes for all recordings and align them to triggers and repeats.

    Input:
        files_dict (dict[str, str]): recording name -> spikes parquet path.
        filter_dict (dict): recording -> stimulus -> filter parameters (start,
            end, cell_indices, trigger, stim_repeat_logic, sampling_freq).
        time (str): "seconds" to convert frame counts, else frames are kept.
        waveforms (bool): If True also load the "waveforms" list column.

    Output:
        pl.DataFrame: Spikes with recording, stimulus_index, times_relative,
        trigger, repeat and times_triggered columns.
    """
    derive_trigger_arrays(filter_dict)
    spikes_by_rec = filter_spikes(files_dict, filter_dict, waveforms)
    return align_to_triggers(spikes_by_rec, filter_dict, time)


def derive_trigger_arrays(filter_dict: dict) -> None:
    """Add the trigger arrays the alignment step needs to filter_dict.

    Input:
        filter_dict (dict): recording -> stimulus -> filter parameters; each
            entry must hold trigger and stim_repeat_logic. Modified in place.

    Output:
        None. Adds rel_trigger and trigger_sub to every stimulus entry.

    Raises:
        ValueError: if a trigger array length is incompatible with its
            stimulus_repeat_logic (len(trigger) - 1 must be a multiple of it).
    """
    for rec, filters in filter_dict.items():
        for stim_id, filt in filters.items():
            trigger = filt["trigger"][0]
            logic = filt["stim_repeat_logic"][0]
            if (len(trigger) - 1) % logic != 0:
                raise ValueError(
                    f"recording {rec} stimulus {stim_id}: trigger array of "
                    f"length {len(trigger)} is incompatible with "
                    f"stimulus_repeat_logic {logic}; (len - 1) must be a "
                    f"multiple of the logic."
                )
            rel_trigger = map_triggers(np.arange(trigger.shape[0]), logic)
            filt["rel_trigger"] = rel_trigger
            filt["trigger_sub"] = sub_trigger(trigger, rel_trigger, logic)


def filter_spikes(
    files_dict: dict[str, str], filter_dict: dict, waveforms: bool = False
) -> dict[str, pl.LazyFrame]:
    """Filter each recording's spikes to the requested stimuli and cells.

    Input:
        files_dict (dict[str, str]): recording name -> spikes parquet path.
        filter_dict (dict): recording -> stimulus -> filter parameters (start,
            end, cell_indices).
        waveforms (bool): If True also load the "waveforms" list column.

    Output:
        dict[str, pl.LazyFrame]: recording -> its filtered spikes with
        placeholder trigger/repeat/times columns. Recordings with no filters
        are omitted.
    """
    combined_lazy_dfs = {}
    for key, path in files_dict.items():
        filters = filter_dict.get(key, {})
        cols = spike_columns(path, waveforms)
        filtered_dfs = []
        for stim_id, filt in filters.items():
            df_filtered = (
                pl.scan_parquet(path)
                .select(cols)
                .filter(
                    (pl.col("times") > filt["start"])
                    & (pl.col("times") <= filt["end"])
                    & pl.col("cell_index").is_in(filt["cell_indices"])
                )
                .with_columns(pl.lit(stim_id).alias("stimulus_index"))
                .with_columns(pl.lit(key).alias("recording"))
                .with_columns(times_relative=pl.col("times") - filt["start"])
            )
            filtered_dfs.append(df_filtered)

        if filtered_dfs:
            combined_lazy_dfs[key] = pl.concat(filtered_dfs, how="vertical").unique()
    return combined_lazy_dfs


def align_to_triggers(
    spikes_by_rec: dict[str, pl.LazyFrame],
    filter_dict: dict,
    time: str,
) -> pl.DataFrame:
    """Assign each spike its trigger, repeat and stimulus-relative times.

    Input:
        spikes_by_rec (dict[str, pl.LazyFrame]): filter_spikes output; only
            recordings that matched at least one filter.
        filter_dict (dict): recording -> stimulus -> filter parameters enriched
            by derive_trigger_arrays.
        time (str): "seconds" to convert frame counts, else frames are kept.

    Output:
        pl.DataFrame: Spikes with trigger, repeat and times_triggered columns;
        empty if nothing matched.
    """
    if not spikes_by_rec:
        return pl.DataFrame()
    df = pl.concat(pl.collect_all(list(spikes_by_rec.values()))).lazy()
    combined_dfs = []
    for key in spikes_by_rec:
        filtered_dfs = []
        filters = filter_dict.get(key, {})
        for stim_id, filt in filters.items():
            df_filtered = df.filter(
                (pl.col("stimulus_index") == stim_id) & (pl.col("recording") == key)
            )
            nr_repeats = (len(filt["trigger"][0]) - 1) // filt["stim_repeat_logic"][0]
            repeats = np.repeat(
                np.arange(nr_repeats), filt["stim_repeat_logic"][0]
            )

            trig_df = pl.DataFrame({
                "start": filt["trigger"][0][:-1],
                "end": filt["trigger"][0][1:],
                "rel_t": filt["rel_trigger"],
                "sub_t": filt["trigger_sub"],
                "repeat_trig": repeats
            }).sort("start").lazy()

            df_filtered = (
                df_filtered.lazy()
                .sort("times_relative")
                .join_asof(trig_df, left_on="times_relative", right_on="start")
                .filter(pl.col("times_relative") <= pl.col("end"))
                .with_columns([
                    (pl.col("times_relative") - pl.col("sub_t")).alias("times_triggered"),
                    pl.col("rel_t").alias("trigger"),
                    pl.col("repeat_trig").alias("repeat"),
                ])
                .collect()
            )
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

        combined_dfs.append(pl.concat(filtered_dfs, how="vertical").unique())

    return pl.concat(combined_dfs)


def stimulus_relative(
    cell_index: list[int],
    stim_begin: int,
    stim_end: int,
    file: str,
    waveforms: bool = False,
) -> pl.DataFrame:
    """Load window spikes with a times_relative column.

    Input:
        cell_index (list[int]): Cell indices to load.
        stim_begin (int): Window start in frames, exclusive.
        stim_end (int): Window end in frames, inclusive.
        file (str): Path to the spikes parquet.
        waveforms (bool): If True also load the "waveforms" list column.

    Output:
        pl.DataFrame: Window spikes plus times_relative = times - stim_begin.
    """
    subset_df = df_filter_stimulus(cell_index, stim_begin, stim_end, file, waveforms)
    subset_df = subset_df.with_columns(
        pl.col("times").sub(stim_begin).alias("times_relative")
    )
    return subset_df


def map_triggers(input_sequence: np.ndarray, logic: int) -> np.ndarray:
    """Map trigger indices to their position within a repeat.

    Input:
        input_sequence (np.ndarray): Trigger indices, e.g. np.arange(n).
        logic (int): Number of triggers that make up one repeat.

    Output:
        np.ndarray: index % logic for each element, dropping the last one
        (length len(input_sequence) - 1).
    """
    output_sequence = [x % logic for x in input_sequence]
    return np.asarray(output_sequence[:-1])


def sub_trigger(
    trigger: np.ndarray, rel_trigger: np.ndarray, logic: int
) -> np.ndarray:
    """Compute the per-repeat time offset to subtract from each trigger.

    Input:
        trigger (np.ndarray): Trigger onsets.
        rel_trigger (np.ndarray): Trigger positions within a repeat (map_triggers).
        logic (int): Number of triggers that make up one repeat.

    Output:
        np.ndarray: Each repeat's onset, repeated logic times, so it can be
        subtracted from that repeat's triggers.
    """
    trigger_sub = trigger[:-1][rel_trigger == 0]
    trigger_sub = np.repeat(trigger_sub, logic)
    return trigger_sub


def reduce_trigger_times(
    df: pd.DataFrame, stimulus: list, reducer: Callable, time: str = "seconds"
) -> np.ndarray:
    """Reduce each trigger interval across the finished repeats of a stimulus.

    Input:
        df (pd.DataFrame): Stimulus dataframe with trigger_int, nr_repeats,
            stimulus_repeat_logic, stimulus_repeat_sublogic, sampling_freq.
        stimulus (list): Stimulus indices, or ["all"] for every stimulus.
        reducer (Callable): Reduction over repeats, called as reducer(a, axis=0),
            e.g. np.mean or np.max.
        time (str): "seconds" to divide by sampling_freq, else frames.

    Output:
        np.ndarray: The reduced interval per trigger within one repeat.
    """
    if type(stimulus[0]) is str and stimulus[0] == "all":
        stimulus = df["stimulus_index"].unique().tolist()

    stim_df = df[df["stimulus_index"].isin(stimulus)]
    new_trigger_int = np.repeat(
        reducer(
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


def mean_trigger_times(
    df: pd.DataFrame, stimulus: list, time: str = "seconds"
) -> np.ndarray:
    """Average each trigger interval across the finished repeats of a stimulus.

    Input:
        df (pd.DataFrame): Stimulus dataframe (see reduce_trigger_times).
        stimulus (list): Stimulus indices, or ["all"] for every stimulus.
        time (str): "seconds" to divide by sampling_freq, else frames.

    Output:
        np.ndarray: The mean interval per trigger within one repeat.
    """
    return reduce_trigger_times(df, stimulus, np.mean, time)


def max_trigger_times(
    df: pd.DataFrame, stimulus: list, time: str = "seconds"
) -> np.ndarray:
    """Take the max of each trigger interval across finished repeats of a stimulus.

    Input:
        df (pd.DataFrame): Stimulus dataframe (see reduce_trigger_times).
        stimulus (list): Stimulus indices, or ["all"] for every stimulus.
        time (str): "seconds" to divide by sampling_freq, else frames.

    Output:
        np.ndarray: The max interval per trigger within one repeat.
    """
    return reduce_trigger_times(df, stimulus, np.max, time)


def stim_duration(df: pd.DataFrame, stimulus: list, time: str = "seconds") -> float:
    """Sum the mean trigger intervals to get one repeat's duration.

    Input:
        df (pd.DataFrame): Stimulus dataframe (see mean_trigger_times).
        stimulus (list): Stimulus indices, or ["all"] for every stimulus.
        time (str): "seconds" to divide by sampling_freq, else frames.

    Output:
        float: Total duration of one repeat of the stimulus.
    """
    return np.sum(mean_trigger_times(df, stimulus, time))