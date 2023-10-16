import Overview
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import stimulus_trace
import plotly.graph_objects as go
import param
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import time
from backbone import SelectFilesButton
import backbone
import Extractors
import recording_overview
import ipywidgets as widgets
import single_cell_plots
from pathlib import Path


plt.style.use("dark_background")


class Explorer:
    def __init__(self):
        self.stimulus_input = SelectFilesButton("Stimulus File")
        self.recording_input = SelectFilesButton("Recording File")
        self.stimulus_input.observe(self._on_stimulus_selected, names="files")
        self.recording_input.observe(self._on_recording_selected, names="files")
        self.stimulus = None
        self.text = """ ## Single Recording Overview.
        This UI allows you to explore
        a single MEA recording fast
        and interactively.

        """

        warnings = pn.widgets.StaticText(name="Warnings", value="No warnings")
        self.overview_df = None
        self.load_button = pn.widgets.Button(
            name="Load Recording", button_type="primary", width=200
        )
        self.save_button = pn.widgets.Button(
            name="Save to Overview", button_type="primary", width=200
        )
        self.save_button.on_click(self.save_to_overview)
        self.load_button.on_click(self.load_data)

        self.recording_name = pn.widgets.TextInput(
            name="Recording Name", placeholder="Recording Name", width=200
        )

        self.define_stimuli_button = pn.widgets.Button(
            name="Define Stimuli", button_type="primary", width=150
        )
        self.stimulus_spikes_button = pn.widgets.Button(
            name="Match stimulus spikes", button_type="primary", width=150
        )
        self.define_stimuli_button.on_click(self.define_stimuli)
        self.stimulus_spikes_button.on_click(self.stimulus_spikes)

        # Creating an instance of PlotApp
        self.plot_app = PlotApp()

        self.recording = None

        self.stim_figure = self.stim_figure = pn.panel(
            self.plot_app.plot,
            width=1000,
            height=500,
        )
        self.frequency_input = pn.widgets.FloatInput(
            name="Recording Frequency",
            value=10000.0,
            step=1.0,
            start=0.0,
            end=50000.0,
            width=200,
        )

        self.spikes_fig = widgets.Output()

        self.isi_fig = widgets.Output()

        self.spike_trains = widgets.Output()
        self.isi_clus_fig = widgets.Output()

        self.stimulus_df = pn.widgets.Tabulator(
            pd.DataFrame(),
            name="Stimuli",
            theme="simple",
            widths=150,
            width=1000,
            height=200,
        )
        self.stimulus_select = pn.widgets.Select(name="Select Stimulus", options=[])
        self.stimulus_select.param.watch(self.update_stimulus_tabulator, "value")

        self.status = pn.indicators.Progress(
            name="Indeterminate Progress", active=False, width=200
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
        fig.update_layout(width=1200, height=500)
        self.single_cell_raster = pn.pane.Plotly(fig)

        self.single_stimulus_df = pn.widgets.Tabulator(
            pd.DataFrame(),
            name="single stimulus",
            theme="simple",
            widths=150,
            width=900,
            height=200,
        )
        self.single_stimulus_df.on_click(self.update_raster)

        self.sidebar = pn.layout.WidgetBox(
            pn.pane.Markdown(self.text, margin=(0, 10)),
            "Stimulus File",
            self.stimulus_input,
            "Recording File",
            self.recording_input,
            self.recording_name,
            self.frequency_input,
            self.load_button,
            self.save_button,
            self.status,
            warnings,
            max_width=300,
            max_height=1000,
            sizing_mode="stretch_width",
        ).servable(area="sidebar")

        self.main = pn.Tabs(
            (
                "Stimulus",
                pn.Row(
                    pn.Column(
                        pn.Column(
                            pn.Row(
                                self.stim_figure,
                                height=600,
                                width=1000,
                            ),
                            pn.Row(self.stimulus_df, height=200, width=1000),
                        ),
                        height=800,
                        width=1000,
                    ),
                    pn.Column(
                        self.define_stimuli_button,
                        self.stimulus_spikes_button,
                    ),
                ),
            ),
            (
                "Spikes",
                pn.Tabs(
                    (
                        "Spike Counts",
                        pn.Column(self.spikes_fig, sizing_mode="stretch_width"),
                    ),
                    ("ISI_time", pn.Column(self.isi_fig, sizing_mode="stretch_width")),
                    (
                        "ISI_cluster",
                        pn.Column(self.isi_clus_fig, sizing_mode="stretch_width"),
                    ),
                    (
                        "Raster",
                        pn.Column(self.spike_trains, sizing_mode="stretch_width"),
                    ),
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "Stimuli",
                pn.Column(
                    pn.Row(
                        pn.Column(self.stimulus_select),
                        pn.Column(self.single_stimulus_df),
                    ),
                    pn.Row(self.single_cell_raster, width=1000),
                ),
            ),
            height=600,
        )

    def run(self):
        app = pn.Row(
            pn.Column(self.sidebar, sizing_mode="fixed", width=300),
            pn.Spacer(width=20),  # Adjust width for desired spacing
            pn.Column(self.main, sizing_mode="stretch_width"),
            width=1000,
            height=1000,
        )
        return app

    def load_data(self, event):
        if self.stimulus_input.loaded:
            self.status.active = True
            self.load_stimulus()
            self.stimulus_input.loaded = False
            self.status.active = False
        if self.recording_input.loaded:
            self.status.active = True
            self.load_recording()
            self.recording_input.loaded = False
            self.status.active = False

    def _on_stimulus_selected(self, change):
        self.stimulus_file = change["new"][0] if change["new"] else ""
        # print(f"File selected: {self.stimulus_file}")

    def _on_recording_selected(self, change):
        self.recording_file = change["new"][0] if change["new"] else ""
        # print(f"File selected: {self.recording_file}")

    def define_stimuli(self, event):
        begins, ends = self.plot_app.get_frames()
        stim_df = self.stimulus.get_stim_range_new(begins, ends)
        self.stimulus_df.value = stim_df
        self.stimulus_df.frozen_columns = [
            "stimulus_name",
            "stimulus_index",
            "stimulus_repeat_logic",
        ]

    def stimulus_spikes(self, event):
        self.stimulus_df.value = stimulus_trace.find_ends(self.stimulus_df.value)
        self.recording.add_stimulus_df(self.stimulus_df.value)
        dataframes = {
            "spikes_df": self.recording.load(),
            "stimulus_df": self.stimulus_df.value,
        }
        dataframes["spikes_df"]["filter"] = True
        self.overview_df = Overview.Recording(
            str(self.recording.file.with_suffix(".parquet")),
            self.recording_name.value,
            dataframes,
            self.frequency_input.value,
        )
        stimulus_names = self.overview_df.stimulus_df["stimulus_name"].tolist()
        options_dict = {name: idx for idx, name in enumerate(stimulus_names)}
        self.stimulus_select.options = options_dict

    def load_stimulus(self):
        self.stimulus = stimulus_trace.Stimulus_Extractor(
            self.stimulus_file, self.frequency_input.value, 0
        )
        channel = self.stimulus.downsample("500ms")

        # Debugging: Print or log the loaded data to ensure it's what you expect
        # print(channel)

        self.plot_app.update_plot(
            channel["Frame"].to_numpy(), channel["Voltage"].to_numpy()
        )

        # print("Before update: ", self.stim_figure.object)  # Debug
        self.stim_figure.object = self.plot_app.plot
        # print("After update: ", self.stim_figure.object)  # Debug

    def load_recording(self):
        file_format = backbone.get_file_ending(self.recording_file)
        if file_format == ".dat":
            self.recording = Extractors.Extractor_SPC(self.recording_file)
        elif file_format == ".hdf5":
            self.recording = Extractors.Extractor_HS2(self.recording_file)
            self.frequency_input.value = float(self.recording.spikes["sampling"])

        self.recording.get_spikes()
        self.plot_spike_counts()
        self.plot_isi("times", self.isi_fig)
        self.plot_isi("cell_index", self.isi_clus_fig)
        self.plot_spike_trains()

    def plot_spike_counts(self):
        fig, ax = recording_overview.spike_counts_from_file(
            str(self.recording.file.with_suffix(".parquet")), "viridis"
        )
        fig.set_size_inches(10, 8)
        with self.spikes_fig:
            fig.show()

    def plot_isi(self, x, widget):
        fig, ax = recording_overview.isi_from_file(
            str(self.recording.file.with_suffix(".parquet")),
            self.frequency_input.value,
            x=x,
            cmap="viridis",
            cutoff=np.log10(0.001),
        )
        fig.set_size_inches(10, 8)
        with widget:
            fig.show()

    def plot_spike_trains(self):
        fig, ax = recording_overview.spiketrains_from_file(
            str(self.recording.file.with_suffix(".parquet")),
            self.frequency_input.value,
            cmap="gist_gray",
        )
        fig.set_size_inches(10, 8)
        with self.spike_trains:
            fig.show()

    def update_stimulus_tabulator(self, event):
        stimulus_id = self.stimulus_select.value

        self.single_stimulus_df.value = self.overview_df.spikes_df.query(
            f"stimulus_index == {stimulus_id}"
        )

    def update_raster(self, event):
        try:
            indices = event.row
            cell_indices = [self.single_stimulus_df.value.loc[indices]["cell_index"]]

            plot_df = self.overview_df.get_spikes_triggered(
                cell_indices, [self.stimulus_select.value], time="seconds"
            )
            raster_plot = single_cell_plots.whole_stimulus_plotly(plot_df)
            self.single_cell_raster.object = raster_plot

        except KeyError:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
            fig.update_layout(width=1200, height=500)
            self.single_cell_raster.object = fig

    def save_to_overview(self, event):
        if self.overview_df is not None:
            self.overview_df.save(Path(self.stimulus_file).parent / "overview")
        else:
            print("No overview to save")


class PlotApp(param.Parameterized):
    plot = param.Parameter(precedence=-1)
    switch = param.Boolean(default=False, precedence=-1)

    def __init__(self, x=None, y=None):
        super().__init__()
        if x is None:
            self.x = np.array([0])
        if y is None:
            y = np.array([0])

        # Initial state setup
        self.begins = np.zeros(self.x.shape[0], dtype=bool)
        self.ends = np.zeros(self.x.shape[0], dtype=bool)
        self.switch = True
        self.last_callback_time = time.time()

        # Create initial data source
        self.source = ColumnDataSource(
            data=dict(
                x=self.x,
                y=y,
                color=["blue"] * self.x.shape[0],
                size=[7] * self.x.shape[0],
            )
        )
        self.plot = figure(
            tools="tap,pan,wheel_zoom,box_zoom,reset",
            width=1200,
            height=500,
            title="Stimulus selection",
            output_backend="webgl",
        )

        # Add glyphs to the plot
        self.plot.line("x", "y", source=self.source, line_width=2, line_color="blue")
        self.scatter_renderer = self.plot.circle(
            "x",
            "y",
            color="color",
            size="size",
            source=self.source,
            nonselection_alpha=1.0,
        )
        self.plot.xaxis.axis_label = "Frame"
        self.plot.yaxis.axis_label = "Voltage"

    def update_plot(self, x, y):
        """
        Update the plot with new data.

        Parameters
        ----------
        x : array-like
            New x-data.
        y : array-like
            New y-data.
        """
        self.x = x
        self.begins = np.zeros(self.x.shape[0], dtype=bool)
        self.ends = np.zeros(self.x.shape[0], dtype=bool)
        # Update data source
        self.source.data = dict(
            x=self.x, y=y, color=["blue"] * self.x.shape[0], size=[7] * self.x.shape[0]
        )
        # print(x)
        # Setup callback for selection changes
        self.source.selected.on_change("indices", self.callback)
        # Create a new figure object

    def callback(self, attr, old, new):
        current_time = time.time()
        try:
            elapsed_time = current_time - self.last_callback_time
        except AttributeError:
            elapsed_time = 1  # First callback, so allow it
        self.last_callback_time = current_time
        if (
            elapsed_time < 0.5
        ):  # If less than 0.5 seconds elapsed since last callback, ignore
            print("Callback debounced")
            return
        if new:
            colors = list(self.source.data["color"])
            sizes = list(
                self.source.data["size"]
            )  # Added this line to get current sizes

            # Get the index of the last selected point
            idx = new[-1]
            if colors[idx] == "blue" and self.switch:
                colors[idx] = "green"
                sizes[idx] = 10  # Added this line to change size
                self.begins[idx] = True
                self.switch = False
            elif colors[idx] == "blue" and not self.switch:
                colors[idx] = "red"
                sizes[idx] = 10  # Added this line to change size
                self.ends[idx] = True
                self.switch = True
            elif colors[idx] == "red" and self.switch:
                colors[idx] = "blue"
                sizes[idx] = 7  # Added this line to revert size
                self.ends[idx] = False
                self.switch = False
            elif colors[idx] == "green" and not self.switch:
                colors[idx] = "blue"
                sizes[idx] = 7  # Added this line to revert size
                self.begins[idx] = False
                self.switch = True

            # Update source data
            self.source.data["color"] = colors
            self.source.data[
                "size"
            ] = sizes  # Added this line to update sizes in source
            self.source.selected.indices = []

    def get_frames(self):
        return self.x[self.begins], self.x[self.ends]
