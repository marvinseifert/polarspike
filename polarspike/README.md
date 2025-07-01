# Polarspike

Welcome to Polarspike - a Python library for analyzing and visualizing spike trains. Written by
[Marvin Seifert](m.seifert@sussex.ac.uk)

## Selling points / Quick summary

The idea of Polarspike is simple and effective: to provide a simple and intuitive interface for analyzing and
visualizing spike trains. At the backend, Polarspike uses [Polars](https://www.pola.rs)
and [Pyarrow](https://arrow.apache.org/docs/python/) which allows for fast processing of large datasets. On the frontend
spike trains are organized in [Pandas](https://pandas.pydata.org/) dataframes (experts in polars can stick to polars
also
on the frontend), which allows for easy data manipulation
and plotting.
Polarspike was written with the aim to be able to process the data from multiple multi-electrode array (MEA) recordings
without the need to set up a "real" database running on a server.
This is archived by allowing the user to load chunks of multiple recordings into memory and analyze them. This way,
the user can analyze a potentially unlimited number of recordings, as long as a single stimulus/ recording/ chunk
fits into memory.
<b>Plotting</b> is done making heavy use of [Datashader](https://datashader.org/), this allows for fast plotting
of millions of spikes. Polarspike also provides simple interfaces for plotting raster plots and for further data
analysis.

## Installation

Currently, Polarspike is not available on PyPI, so you have to install it from source. To do so, clone this repository
and run `pip install .` in the root directory of the repository. Alternatively, you can use Python Poetry to install
Polarspike. To do so, run `poetry install` in the root directory of the repository.

### Data format

The most challenging step might be to organize your data in a way that Polarspike can read it. Polarspike was tested
on datasets
using [Herding Spikes 2](https://github.com/mhhennig/HS2), [Kilosort2](https://github.com/MouseLand/Kilosort2) and
[Spyking Circus](https://spyking-circus.readthedocs.io/en/latest/). The respective Extractors are included in
the Extractors module. I am happy to help people to write their own Extractors for other spike sorting software.

## Structure

On the frontend, Polarspike is organized around two dataframes: One dataframe contains information about the
cells (e.g. their location, their type, etc.) and the other dataframe contains information about the stimulus
(e.g. the stimulus type, the stimulus parameters, etc.). This makes Polarspike very flexible, as it allows the user
to easily filter cells and stimuli based on their properties. The user can also add additional "stimuli" to the
stimulus dataframe, e.g. to add a stimulus that is not part of the original recording. Suppose, in the original
recording
you played a stimulus that lasts 10s and repeats several times but your analysis revealed that the relevant time window
is only the first 1 second.
Instead of having to re-strucutre your data, you can simply add a new stimulus to the stimulus dataframe that only
points to the first second of the original stimulus and load only the respective spikes into memory. (Look at the
example
notebooks, to get an idea of how this is very useful and flexible).
If you need the original 10 seconds stimulus again, you can simply load it when needed.

## Interactivity

Often the most important part of data analysis is to explore the data. Polarspike provides interactive
features that can be used in Jupyter lab or Jupyter notebook (not sure all widgets will work). For example, you can
interact with all
dataframes interactively using [Panel's tabulator](https://panel.holoviz.org), e.g. to filter cells or stimuli based on
their properties. You can
also plot
spike trains interactively, e.g. to zoom in on a specific time window or to select a subset of cells using either
[Matplotlib](https://matplotlib.org/) widget or [Plotly](https://plotly.com/).

## Example

You can download an example dataset from [Box](). Check out the example notebooks in the [examples](examples) folder.

## Documentation

Full documentaion is available [here](https://polarspike.readthedocs.io/en/latest/)

## Contributing

I am happy to receive contributions to Polarspike. I am not a professional software developer, so I am sure there is
a lot of room for improvement. If you want to contribute, please open an issue first to discuss
the changes you want to make. Thank you. Happy spike-training!
