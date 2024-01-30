from polarspike import Overview
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
from polarspike import stimulus_trace
import plotly.graph_objects as go
import param
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import time
from polarspike.backbone import SelectFilesButton
from polarspike import backbone
from polarspike import Extractors
from polarspike import recording_overview
import ipywidgets as widgets
from polarspike import spiketrain_plots
from pathlib import Path
from polarspike import colour_template
from polarspike import stimulus_spikes
from polarspike import binarizer
from math import pi
from bokeh.palettes import Category20c, Category20
from bokeh.plotting import figure
from bokeh.transform import cumsum
import panel as pn
import matplotlib
from polarspike.grid import Table
from polarspike import plotly_templates, grid, waveforms
from functools import partial


def stimulus_df():
    uneditable_columns = [
        "stimulus_index",
        "begin_fr",
        "end_fr",
        "trigger_fr_relative",
        "trigger_ends",
        "trigger_int",
        "sampling_freq",
        "nr_repeats",
    ]
    editors = {col: {"type": "editable", "value": False} for col in uneditable_columns}

    widget = pn.widgets.Tabulator(
        pd.DataFrame(),
        name="single stimulus",
        editors=editors,
        theme="simple",
        widths=150,
        width=900,
        height=200,
    )
    widget.frozen_columns = [
        "stimulus_name",
        "stimulus_index",
        "stimulus_repeat_logic",
        "stimulus_repeat_sublogic",
    ]
    return widget


plt.style.use("dark_background")


class Explorer:
    def __init__(self):
        self.ct = colour_template.Colour_template()
        self.stimulus_input = SelectFilesButton("Stimulus File")
        self.recording_input = SelectFilesButton("Recording File")
        self.sorting_input = SelectFilesButton("Sorting File")
        self.stimulus_input.observe(self._on_stimulus_selected, names="files")
        self.recording_input.observe(self._on_recording_selected, names="files")
        self.sorting_input.observe(self._on_sorting_selected, names="files")
        self.stimulus = None
        self.mea_type = None
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
            name="recording name", placeholder="recording name (LC!)", width=200
        )

        self.define_stimuli_button = pn.widgets.Button(
            name="Define Stimuli", button_type="primary", width=150
        )
        self.stimulus_spikes_button = pn.widgets.Button(
            name="Match stimulus spikes", button_type="primary", width=150
        )
        self.define_stimuli_button.on_click(self.define_stimuli)
        self.stimulus_spikes_button.on_click(self.stimulus_spikes)
        self.calculate_qi_button = pn.widgets.Button(name="Calculate QI", width=150)
        self.calculate_qi_button.on_click(self.calculate_qi)
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

        self.stimulus_df = stimulus_df()
        self.stimulus_select = pn.widgets.Select(name="Select Stimulus", options=[])
        self.stimulus_select.param.watch(self.update_stimulus_tabulator, "value")
        self.initial_stim_df = False

        self.status = pn.indicators.Progress(
            name="Indeterminate Progress", active=False, width=200
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
        fig.update_layout(width=1200, height=700)
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

        # Color
        self.colour_selector = pn.widgets.Select(
            name="Select colour set", options=self.ct.list_stimuli().tolist()
        )

        self.colour_selector.param.watch(self.on_colour_change, "value")
        self.selected_color = pn.pane.IPyWidget(self.ct.pickstimcolour())

        self.sidebar = pn.layout.WidgetBox(
            pn.pane.Markdown(self.text, margin=(0, 10)),
            "Stimulus File",
            self.stimulus_input,
            "Recording File",
            self.recording_input,
            "Sorting File </br> (only for herdingspikes2)",
            self.sorting_input,
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
                pn.Row(
                    pn.Column(
                        self.stimulus_select, self.colour_selector, self.selected_color
                    ),
                    pn.Column(
                        pn.Row(
                            self.single_stimulus_df,
                            self.calculate_qi_button,
                            width=1000,
                        ),
                        pn.Row(self.single_cell_raster, width=1000),
                    ),
                ),
            ),
            height=800,
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
            self.recording_name.value = self.recording_name.value.lower()
            self.status.active = True
            self.load_recording()
            self.recording_input.loaded = False
            self.status.active = False

    def _on_stimulus_selected(self, change):
        self.stimulus_file = change["new"][0] if change["new"] else ""
        self.recording_name.value = (
            "".join(Path(self.stimulus_file).parts[1:-1])
        ).lower()
        # print(f"File selected: {self.stimulus_file}")

    def _on_recording_selected(self, change):
        self.recording_file = change["new"][0] if change["new"] else ""
        # print(f"File selected: {self.recording_file}")

    def _on_sorting_selected(self, change):
        self.sorting_file = change["new"][0] if change["new"] else ""
        # print(f"File selected: {self.sorting_file}")

    def define_stimuli(self, event):
        begins, ends = self.plot_app.get_frames()
        stim_df = self.stimulus.get_stim_range_new(begins, ends)
        stim_df["recording"] = self.recording_name.value
        stim_df["sampling_freq"] = self.frequency_input.value
        self.stimulus_df.value = stim_df

    def stimulus_spikes(self, event):
        self.stimulus_df.value = stimulus_trace.find_ends(self.stimulus_df.value)
        self.stimulus_df.value = stimulus_trace.get_nr_repeats(self.stimulus_df.value)
        self.recording.add_stimulus_df(self.stimulus_df.value)
        dataframes = {
            "spikes_df": self.recording.load(recording_name=self.recording_name.value),
            "stimulus_df": self.stimulus_df.value,
        }
        dataframes["spikes_df"]["filter"] = True

        self.overview_df = Overview.Recording(
            str(self.recording.file.with_suffix(".parquet")),
            self.recording_file,
            dataframes,
            self.frequency_input.value,
        )
        stimulus_names = self.overview_df.stimulus_df["stimulus_name"].tolist()
        options_dict = {f"{name}_{idx}": idx for idx, name in enumerate(stimulus_names)}
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

    def load_recording(self):
        file_format = backbone.get_file_ending(self.recording_file)
        if file_format == ".dat":
            self.recording = Extractors.Extractor_SPC(self.recording_file)
            self.mea_type = "MCS"
            try:
                self.frequency_input.value = float(
                    np.loadtxt(
                        Path(self.recording_file).parent / "bininfo.txt", dtype=object
                    )[1]
                )

            except FileNotFoundError:
                print("No bininfo.txt file found. Using user input frequency.")

        elif file_format == ".brw":
            self.mea_type = "3Brain"
            self.recording = Extractors.Extractor_HS2(self.sorting_file)
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
        ).reset_index(drop=True)

    def merge_stimulus_df(self):
        df_temp = self.single_stimulus_df.value.set_index(
            ["cell_index", "stimulus_index"]
        )
        self.overview_df.dataframes["spikes_df"] = self.overview_df.dataframes[
            "spikes_df"
        ].set_index(["cell_index", "stimulus_index"])
        if "qi" in self.overview_df.dataframes["spikes_df"].columns:
            self.overview_df.dataframes["spikes_df"].update(df_temp)
        else:
            self.overview_df.dataframes["spikes_df"] = pd.concat(
                [self.overview_df.dataframes["spikes_df"], df_temp["qi"]], axis=1
            )

        self.overview_df.dataframes["spikes_df"] = self.overview_df.dataframes[
            "spikes_df"
        ].reset_index(drop=False)

    def update_raster(self, event):
        indices = event.row
        cell_indices = [self.single_stimulus_df.value.loc[indices]["cell_index"]]

        plot_df = self.overview_df.get_spikes_triggered(
            [[self.stimulus_select.value]], [cell_indices], time="seconds"
        )
        if len(plot_df) != 0:
            raster_plot = spiketrain_plots.whole_stimulus_plotly(plot_df)
            # Add stimulus:
            raster_plot = self.ct.add_stimulus_to_plot(
                raster_plot,
                stimulus_spikes.mean_trigger_times(
                    self.overview_df.stimulus_df, [self.stimulus_select.value]
                ),
            )
            self.single_cell_raster.object = raster_plot

        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
            self.single_cell_raster.object = fig

    def save_to_overview(self, event):
        if self.overview_df is not None:
            if self.mea_type == "MCS":
                self.overview_df.save(Path(self.stimulus_file).parents[1] / "overview")
            elif self.mea_type == "3Brain":
                self.overview_df.save(Path(self.stimulus_file).parents[0] / "overview")

        else:
            print("No overview to save")

    def on_colour_change(self, event):
        self.ct.pick_stimulus(event.new)
        self.selected_color.object = self.ct.pickstimcolour(event.new)

    def calculate_qi(self, event):
        self.status.active = True
        cell_ids = self.single_stimulus_df.value["cell_index"].unique().tolist()
        spikes_df = self.overview_df.get_spikes_triggered(
            [[self.stimulus_select.value]], [cell_ids], time="seconds", pandas=False
        )
        # Update the cell indices, to account for cells without responses
        binary_df = binarizer.timestamps_to_binary_multi(
            spikes_df,
            0.001,
            np.sum(
                stimulus_spikes.mean_trigger_times(
                    self.overview_df.stimulus_df, [self.stimulus_select.value]
                )
            ),
            self.overview_df.stimulus_df.loc[self.stimulus_select.value]["nr_repeats"],
        )
        qis = binarizer.calc_qis(binary_df)
        cell_ids = binary_df["cell_index"].unique().to_numpy()

        df_temp = self.single_stimulus_df.value.set_index("cell_index")
        df_temp["qi"] = 0
        df_temp.loc[cell_ids, "qi"] = qis
        self.single_stimulus_df.value = df_temp.reset_index(drop=False)
        # Add the QI to the overview object
        self.merge_stimulus_df()
        self.status.active = False


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


class Recording_explorer:
    def __init__(self, analysis_path):
        self.analysis_path = analysis_path
        self.nr_added_recordings = 0
        self.recordings_dataframe = pd.DataFrame(
            columns=["name", "save_path", "raw_path", "parquet_path", "sampling_freq"]
        )
        self.recordings_dataframe_widget = pn.widgets.DataFrame(
            self.recordings_dataframe, max_height=5
        )

        # Placeholders
        self.spikes_fig = widgets.Output()

        self.isi_fig = widgets.Output()

        self.spike_trains = widgets.Output()
        self.isi_clus_fig = widgets.Output()

        # Nr spikes figure
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=["empty"], values=[0]))
        fig.update_layout(template="scatter_template_jupyter")

        self.nr_cells_fig = pn.pane.Plotly(fig, width=500, height=300)

        # Recording menu
        # Single Reocring Menu
        self.single_recording_items = []

        self.single_recording_menu = pn.widgets.MenuButton(
            name="Recording",
            items=self.single_recording_items,
            button_type="primary",
            width=200,
        )
        self.single_recording_menu.on_click(self.load_single_recording)

        # Dataframe Tab
        self.dataframe_names = ["spikes_df", "stimulus_df"]

        self.dataframe_menu = pn.widgets.MenuButton(
            name="Dataframe",
            items=self.dataframe_names,
            button_type="primary",
            width=200,
        )
        self.dataframe_menu.on_click(self.fill_dataframe)

        self.grid_row = pn.Row()

        # Action menu
        self.action_dict = {
            "single_cell_raster": {
                "func": partial(
                    spiketrain_plots.spikes_and_trace,
                    stacked=True,
                    indices=["cell_index", "repeat"],
                ),
                "type": "stimulus_trace",
            },
            "multiple_cell_raster": {
                "func": partial(
                    spiketrain_plots.spikes_and_trace,
                    stacked=True,
                    indices=["recording", "cell_index", "repeat"],
                ),
                "type": "stimulus_trace",
            },
            "many_cell_raster": {
                "func": spiketrain_plots.whole_stimulus_plotly,
                "type": "stimulus_trace",
            },
            "waveforms": waveforms.plot_waveforms,
        }
        self.action_items = [item for item in self.action_dict.keys()]
        self.action_menu = pn.widgets.MenuButton(
            name="Actions", items=self.action_items, button_type="primary", width=200
        )
        self.action_menu.on_click(self.trigger_action)

        # Output
        self.output = pn.Column()

        # Colour Panel
        self.ct = colour_template.Colour_template()
        self.colour_selector = pn.widgets.Select(
            name="Select colour set", options=self.ct.list_stimuli().tolist()
        )
        self.colour_selector.param.watch(self.on_colour_change, "value")
        self.selected_color = pn.pane.IPyWidget(self.ct.pickstimcolour())

        # Main

        self.recordings_object = Overview.Recording_s(analysis_path, "gui_analysis")
        self.load_button = SelectFilesButton("Recordings")
        self.load_button.observe(self.load_recording, names="files")
        self.spike_trains = widgets.Output()
        self.main = pn.Tabs(
            (
                "Overview",
                pn.Column(
                    "## Recordings loaded",
                    self.recordings_dataframe_widget,
                    pn.Accordion(
                        ("Nr cells", self.nr_cells_fig),
                    ),
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "Single Recording",
                pn.Column(
                    pn.Column(
                        (self.single_recording_menu),
                        pn.Column(
                            pn.Tabs(
                                (
                                    "Spike Counts",
                                    pn.Column(
                                        self.spikes_fig, sizing_mode="stretch_width"
                                    ),
                                ),
                                (
                                    "ISI_time",
                                    pn.Column(
                                        self.isi_fig, sizing_mode="stretch_width"
                                    ),
                                ),
                                (
                                    "ISI_cluster",
                                    pn.Column(
                                        self.isi_clus_fig, sizing_mode="stretch_width"
                                    ),
                                ),
                                (
                                    "Raster",
                                    pn.Column(
                                        self.spike_trains, sizing_mode="stretch_width"
                                    ),
                                ),
                                sizing_mode="stretch_width",
                            ),
                        ),
                    )
                ),
            ),
            ("Dataframes", pn.Column(pn.Row(self.dataframe_menu), self.grid_row)),
            ("Output", self.output),
            width=1500,
            height=1000,
        )

        # Sidebar
        self.sidebar = pn.layout.WidgetBox(
            "Load Recording",
            self.load_button,
            pn.Column(self.action_menu, sizing_mode="stretch_width"),
            "Active Colour template",
            self.colour_selector,
            self.selected_color,
            sizing_mode="fixed",
            width=310,
            height=1000,
        ).servable(area="sidebar")

    def plot_recording_spike_trains(self):
        fig, ax = recording_overview.spiketrains_from_file(
            self.recording.parquet_path, self.recording.sampling_freq, cmap="gist_gray"
        )
        fig = recording_overview.add_stimulus_df(fig, self.recording.stimulus_df)
        fig.set_size_inches(10, 8)

        # Create a Matplotlib pane and display it
        matplotlib_pane = pn.pane.Matplotlib(fig, width=400, height=400)
        self.spike_trains.object = matplotlib_pane

    def load_recording(self, change):
        if change["new"]:
            recording_path = change["new"][0] if change["new"] else ""
            recording = Overview.Recording.load(recording_path)
            self.recordings_object.add_recording(recording)
            self.nr_added_recordings += 1
            self.single_recording_items.append(
                (recording.name, recording.name)
            )  # Append as a tuple (name, callback)
            self.single_recording_menu.items = self.single_recording_items
            self.add_recording_info_to_df(recording)
            self.single_recording_menu.param.trigger("items")
            self.recordings_dataframe_widget.param.trigger("value")
            self.nr_cells_recording()
            self.nr_cells_fig.param.trigger("object")
            self.fill_dataframe_menu()
            self.fill_dataframe()

    def load_single_recording(self, event):
        recording_name = self.single_recording_menu.clicked
        recording = self.recordings_object.recordings[recording_name]
        self.plot_spike_counts(recording)
        self.plot_isi(recording, "times", self.isi_fig)
        self.plot_isi(recording, "cell_index", self.isi_clus_fig)
        self.plot_spike_trains(recording)

    def plot_spike_counts(self, recording):
        fig, ax = recording_overview.spike_counts_from_file(
            recording.parquet_path, "viridis"
        )
        fig.set_size_inches(10, 8)
        self.spikes_fig.clear_output()
        with self.spikes_fig:
            fig.show()

    def plot_isi(self, recording, x, widget):
        fig, ax = recording_overview.isi_from_file(
            recording.parquet_path,
            recording.sampling_freq,
            x=x,
            cmap="viridis",
            cutoff=np.log10(0.001),
        )
        fig.set_size_inches(10, 8)
        widget.clear_output()
        with widget:
            fig.show()

    def plot_spike_trains(self, recording):
        fig, ax = recording_overview.spiketrains_from_file(
            recording.parquet_path,
            recording.sampling_freq,
            cmap="gist_gray",
        )
        fig = recording_overview.add_stimulus_df(fig, recording.stimulus_df)
        fig.set_size_inches(10, 8)
        self.spike_trains.clear_output()
        with self.spike_trains:
            fig.show()

    def add_recording_info_to_df(self, recording):
        new_df = pd.DataFrame(
            {
                "name": recording.name,
                "save_path": recording.load_path,
                "raw_path": recording.raw_path,
                "parquet_path": recording.parquet_path,
                "sampling_freq": recording.sampling_freq,
            },
            index=pd.Index([self.nr_added_recordings]),
        )
        self.recordings_dataframe = pd.concat([self.recordings_dataframe, new_df])
        self.recordings_dataframe_widget.value = self.recordings_dataframe

    def nr_cells_recording(self):
        nr_cells = []
        recordings = []
        for recording in self.recordings_object.recordings.keys():
            nr_cells.append(self.recordings_object.recordings[recording].nr_cells)
            recordings.append(str(recording))

        fig = go.Figure(data=[go.Pie(labels=recordings, values=nr_cells)])

        fig.update_traces(hoverinfo="label+percent", textinfo="value")
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), template="scatter_template_jupyter"
        )

        self.nr_cells_fig.object = fig

    def fill_dataframe_menu(self):
        self.dataframe_menu.items = [
            item for item in self.recordings_object.dataframes.keys()
        ]

    def fill_dataframe(self, *args, **kwargs):
        df_name = self.dataframe_menu.clicked
        table_object = grid.Table(
            self.recordings_object.dataframes[df_name], width=1500
        ).panel
        table_object[-1].selectable = "checkbox"
        self.grid_row.clear()
        self.grid_row.objects = [table_object]

    def on_colour_change(self, event):
        self.ct.pick_stimulus(event.new)
        self.selected_color.object = self.ct.pickstimcolour(event.new)

    def trigger_action(self, event):
        action = self.action_menu.clicked
        func = self.action_dict[action]["func"]
        cells = self.grid_row.objects[0][1].selection
        self.recordings_object.dataframes[
            "cell_selection"
        ] = self.recordings_object.dataframes[self.dataframe_menu.clicked].iloc[cells]
        stimulus_indices = (
            self.recordings_object.dataframes["cell_selection"]
            .stimulus_index.unique()
            .tolist()
        )
        spikes = self.recordings_object.get_spikes_df("cell_selection")
        fig = func(df=spikes)

        flash_duration = stimulus_spikes.mean_trigger_times(
            self.recordings_object.dataframes["stimulus_df"], stimulus_indices
        )

        fig = self.ct.add_stimulus_to_plot(fig, flash_duration)

        self.output.clear()
        self.output.objects = [fig]

    def serve(self):
        app = pn.Row(
            pn.Column(self.sidebar, sizing_mode="fixed", height=300, width=300),
            pn.Spacer(width=20),  # Adjust width for desired spacing
            pn.Column(self.main, sizing_mode="fixed", height=1000, width=1000),
            sizing_mode="stretch_width",
        )
        return app
