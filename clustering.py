import numpy as np
import polars as pl
from dataclasses import dataclass, field
import zstandard
import multiprocessing as mp
import traceback
from itertools import combinations
from functools import partial
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
def sax_transform(all_values, bins):
    """Transforms values into strings according to bins.

    Parameters
    ----------
    all_values : np.array
        The array containing the observations
    bins : np.array
        Bin edges that will be used for the bining

    Returns
    -------
    bytestring
        A string of bytes representing the bined input values.

    """

    indices = np.digitize(all_values, bins) - 1
    max_unicode = 1114111  # unicode max value
    text = "".join(chr(i % max_unicode) for i in indices)
    return str.encode(text)






@dataclass
class CompressionBasedDissimilarity:
    """Clustering class which combined methods and parameters for CBD clustering.


    Parameters
    ----------
    bins : np.array
        Contains the bins upper edges that will be used for the sax transformation.

    data : np.array
        Contains the data that shall be clustered. Each entry in the array is a numpy array
        containing a trace.
        Traces can have different lengths.

    Attributes
    ----------
    bins : np.array
        Contains the bins left edges that will be used in case "bz2" method for
        compression is used.

    data : np.array
        Contains the data that shall be clustered. Each entry in the array is a numpy array
        containing a trace.
        Traces can have different lengths.

    """
    data: np.ndarray
    bins: np.ndarray
    compression_factor: int = field(init=True, default=3)
    min_value: float = field(init=True, default=0.0)
    max_value: float = field(init=True, default=1.0)
    nr_bins: int = field(init=True, default=10)
    time_invariant: bool = field(init=False, default=False)
    data_df: pl.DataFrame = field(init=False, default=None)
    nr_traces: int = field(init=False, default=0)
    pairwise_df: pl.DataFrame = field(init=False, default=None)
    binsize : float = field(init=False, default=0.0)
    linkage_data: np.ndarray = field(init=False, default=None)
    clusters : np.ndarray = field(init=False, default=None)
    max_trace_length: int = field(init=False, default=0)


    def __post_init__(self):
        # Calculate the min and max value of the data and create the bins
        self.bins = np.linspace(self.min_value, self.max_value, self.nr_bins)
        self.binsize = self.bins[1] - self.bins[0]
        # Create a dataframe from the data

        self.data_df = pl.DataFrame({
        "trace": [i for i, trace in enumerate(self.data) for _ in trace],
        "value": [value for trace in self.data for value in trace],

        })
        self.find_max_trace_length()


        self.nr_traces = len(self.data)

    def find_max_trace_length(self):
        counts_df = self.data_df.groupby("trace").agg(pl.count("trace").alias("count"))
        counts_df = counts_df.sort(pl.col("count").reverse())

        most_frequent_trace = counts_df["trace"].head(1).to_numpy()[0]
        self.max_trace_length = counts_df["count"].head(1).to_numpy()[0]

    def add_noise(self):
        split_temp = self.data_df.partition_by("trace")
        floats = [0, self.binsize]
        noise_stock = np.random.choice(floats, self.max_trace_length)


        for i in range(self.nr_traces):
            noise = noise_stock[:len(split_temp[i])]
            split_temp[i] = split_temp[i].with_columns((pl.col("value") + noise).alias("value"))

        self.data_df = pl.concat(split_temp)
        self.bins = np.append(self.bins, self.bins[-1] + self.binsize)

    def normalize_01(self, how="all"):
        """Normalizes the data to the range [0,1].

        Parameters
        ----------
        how : str, optional
            How the normalization should be performed. Can be "all" or "per_trace". The default is "per_trace".


        Returns
        -------
        The object itself with the normalized data as new trace values.

        """
        if how == "all":
            value = self.data_df["value"] # Extract the values to avoid multiple lookups
            # Normalize the values
            self.data_df = self.data_df.with_columns(
                ((value - value.min()) / (value.max() - value.min())).alias("value")
            )
        elif how == "per_trace": # Normalize per trace

            self.data_df = self.data_df.with_columns(
                (
                    (pl.col("value") - pl.col("value").min().over("trace")) /
                    (pl.col("value").max().over("trace") - pl.col("value").min().over("trace"))
                ).alias("value")
            )
        # Create a new object with the normalized data
        norm_obj = self
        norm_obj.max_value = 1.0 # Change default max value
        norm_obj.min_value = 0.0 # Change default min value

        return norm_obj




    def calculate_single_on_df(self):
        # Create a list of arrays
        store = [np.asarray(trace["value"]) for trace in self.data_df.partition_by("trace")]

        pool = mp.Pool(mp.cpu_count())
        calculate_single_partial = partial(calculate_single, self.bins, self.compression_factor)
        try:
            results = np.asarray(pool.map(calculate_single_partial, store))
            pool.close()
            pool.join()
            return results
        except Exception:
            traceback.print_exc()
            pool.close()
            pool.join()
            return



    def pair_wise_distances(self, n_jobs=-1, method="relative"):

        if n_jobs == -1:
            n_jobs = np.min([mp.cpu_count(), self.nr_traces])

        # Calculate the compression lengths for each trace
        lengths = self.calculate_single_on_df()

        # Get all possible combinations of traces in a list of tuples
        combs_and_traces, lengths_comb = self.combinations_df(lengths,n_jobs=n_jobs)

        # Calculate the pair-wise distances
        pool = mp.Pool(n_jobs)
        partial_distance_combinations = partial(distance_combinations, self.bins, self.compression_factor)
        try:
            results = pool.map(partial_distance_combinations, combs_and_traces)
            results = np.concatenate(results)

            self.pairwise_df =pl.DataFrame({
                "length_m": lengths_comb[:,0],
                "length_n": lengths_comb[:,1],
                "distance_combined": results.flatten()
            })
            if method == "relative":
                self.pairwise_df = self.pairwise_df.with_columns((pl.col("distance_combined") / (pl.col("length_m") +
                                                                   pl.col("length_n"))).alias("distance"))


            pool.close()
            pool.join()
        except Exception:
            traceback.print_exc()
            pool.close()
            pool.join()





    def combinations_df(self, lengths, n_jobs):
        # Get all possible combinations of traces
        combs = combinations(range(len(self.data)), 2)
        combs = np.asarray(list(combs))
        # split the lengths according to combinations
        lengths = lengths[combs]
        # split combs into multiple arrays
        combs_split = np.array_split(combs, n_jobs)
        # split the dataframe accordingly:
        split_dfs = []
        for i, comb in enumerate(combs_split):
            traces_to_select = {i for i in np.unique(comb)}
            split_dfs.append(self.data_df.filter(pl.col("trace").is_in(traces_to_select)))

        combs_and_traces = zip(combs_split, split_dfs)
        return combs_and_traces, lengths


    def dendrogram(self, cluster_method="ward", **kwargs):
        """Create dendrogram from linkage data.Wrapper for scipy linkage and dendrogram
        functions.

        Parameters
        ----------
        cluster_method : string
            See scipy linkage for details. Standard is "ward"

        Returns
        -------
        fig : matplotlib.figure
            Contains the dendrogram

        dend_out : dict
        contains the information that created the dendrogram


        """
        self.linkage_data = linkage(
            self.pairwise_df["distance"].to_numpy(), cluster_method
        )
        fig, ax = plt.subplots()
        plt.xlabel("Traces")
        plt.ylabel("Distance")
        dend_out = dendrogram(self.linkage_data, ax=ax, **kwargs)
        return fig, dend_out

    def get_clusters(self, cut=1, criterion="distance", **kwargs):
        """Cuts the dendrogram at a specific point and returns the cluster indices.

        Parameters
        ----------
        cut : float
            Position at which the dendrogram shall be cut.
        criterion : string
            Definition of how the cut shall be performed. See scipy fcluster.

        Returns
        -------
        clusters : np.array
            Contains the cluster indices.

        """
        self.clusters = fcluster(self.linkage_data, t=cut, criterion="distance", **kwargs)
        return self.clusters
def distance_combinations(bins, compression_factor, combs_and_traces):
    """Performs pair-wise compression based dissimilarity calculation on
    a subset of self.data which is given by comb_df.

    Parameters
    ----------
    combs_and_traces : list of tuples
        Contains the combinations of traces and the corresponding traces.

    Returns
    -------
    comb_df : pd.DataFrame
        The input dataframe with an extra column "distance_combined" which
        contains the combined distance of strings m+n.

    """

    combs = combs_and_traces[0]
    df = combs_and_traces[1]
    nr_comb = combs.shape[0]
    results = np.zeros(nr_comb)
    # Loop over all possible combinations in comb_df
    for idx, comb in enumerate(combs):
        results[idx] = calculate_combined(
            df.filter(pl.col("trace") == comb[0])["value"], df.filter(pl.col("trace") == comb[1])["value"],
            bins, compression_factor
        )
    return results



def calculate_combined(trace_x, trace_y, bins, compression_factor):
    """This function only calculates the compression length for the combination
    of both traces. This can be used in a parallel loop (below). Otherwise,
    similar to self.calculate_single.

    Parameters
    ----------
    trace_x : np.array
        Contains a trace that shall be compressed.
    trace_y : type
        Contains a trace that shall be compressed and compared to m.

    Returns
    -------
    float
        The similarity value of m and n

    """

    trace_x_str = sax_transform(trace_x, bins)
    trace_y_str = sax_transform(trace_y, bins)
    # print(n)

    cctx = zstandard.ZstdCompressor(level=compression_factor, write_content_size=False, write_dict_id=False)
    len_combined = len(cctx.compress(trace_x_str + trace_y_str))
    return len_combined


def calculate_single(bins, compression_factor, trace):
    """This function only calculates the compression length one individual
    trace and returns the compression length.

    Parameters
    ----------
    trace : np.array
        Contains a trace that shall be compressed.

    Returns
    -------
    float
        The compression length of the trace.

    """


    trace_str = sax_transform(trace, bins)
    cctx = zstandard.ZstdCompressor(level=compression_factor, write_content_size=False, write_dict_id=False)
    len_trace = len(cctx.compress(trace_str))
    return len_trace

