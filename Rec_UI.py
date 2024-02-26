from polarspike import Overview
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
from polarspike import stimulus_trace
import plotly.graph_objects as go
import param
from bokeh.plotting import figure, curdoc
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
from polarspike import binarizer, quality_tests
from math import pi
from bokeh.palettes import Category20c, Category20
from bokeh.plotting import figure, output_file, save
from bokeh.transform import cumsum
import panel as pn
import matplotlib
from polarspike.grid import Table
from polarspike import plotly_templates, grid, waveforms
from functools import partial
from panel.theme import Bootstrap, Material, Native
from bokeh.io import export_png
import plotly.express as px
import polars as pl


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
        batch_size = 500
        self.status.active = True
        cell_ids = self.single_stimulus_df.value["cell_index"].unique().tolist()
        spikes_df = self.overview_df.get_spikes_triggered(
            [[self.stimulus_select.value]], [cell_ids], time="seconds", pandas=False
        )

        nr_batches = len(cell_ids) // batch_size
        df_temp = self.single_stimulus_df.value.set_index("cell_index")
        df_temp["qi"] = 0
        for i in range(nr_batches):
            spikes_df_subset = spikes_df.filter(
                pl.col("cell_index").is_in(
                    cell_ids[(i * batch_size) : (i * batch_size) + batch_size]
                )
            )
            if len(spikes_df_subset) > 0:
                # Update the cell indices, to account for cells without responses
                binary_df = binarizer.timestamps_to_binary_multi(
                    spikes_df_subset,
                    0.001,
                    np.sum(
                        stimulus_spikes.mean_trigger_times(
                            self.overview_df.stimulus_df, [self.stimulus_select.value]
                        )
                    ),
                    self.overview_df.stimulus_df.loc[self.stimulus_select.value][
                        "nr_repeats"
                    ],
                )
                qis = binarizer.calc_qis(binary_df)
                cell_ids = binary_df["cell_index"].unique().to_numpy()

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
        pn.config.theme = "dark"
        pn.config.design = Bootstrap
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
            "raster_plot": {
                "func": partial(
                    spiketrain_plots.spikes_and_trace,
                    stacked=True,
                    height=800,
                    width=1500,
                ),
                "type": "stimulus_trace",
            },
            "datashader_plot": {
                "func": partial(
                    spiketrain_plots.whole_stimulus,
                    stacked=True,
                    cmap="gist_gray",
                    line_colour="white",
                    height=8,
                    width=15,
                ),
                "type": "stimulus_trace",
            },
            "waveforms": {"type": "waveforms"},
            "dataframe": {"type": "dataframe"},
            "quality_tests": {"type": "quality_tests"},
            "stats_plots": {"type": "plotly_stats"},
        }
        self.action_items = [item for item in self.action_dict.keys()]
        self.action_menu = pn.widgets.MenuButton(
            name="Actions", items=self.action_items, button_type="primary", width=200
        )
        self.action_menu.on_click(self.action_selected)

        self.action_column = pn.Column([], height=800, width=310)
        self.action_run_button = pn.widgets.Button(name="Go")
        self.action_run_button.on_click(self.trigger_action)

        # Output
        self.output = pn.Column([], sizing_mode="stretch_width")

        # Colour Panel
        self.ct = colour_template.Colour_template()
        self.colour_selector = pn.widgets.Select(
            name="Select colour set", options=self.ct.list_stimuli().tolist(), width=250
        )
        self.colour_checkbox = pn.widgets.Checkbox(name="Plot stimtrace")
        self.colour_checkbox.value = True

        self.colour_selector.param.watch(self.on_colour_change, "value")
        self.selected_color = pn.pane.IPyWidget(
            self.ct.pickstimcolour(small=True), width=250
        )

        self.figure_name_input = pn.widgets.TextInput(
            placeholder="Figure_1", max_length=20, width=100
        )
        self.save_figure_button = pn.widgets.Button(name="Save Figure")
        self.save_figure_button.on_click(self.save_figure)

        # Bin size
        self.bin_size_input = pn.widgets.FloatInput(
            name="Select bin size", value=0.05, step=0.01, start=0, end=10, width=100
        )

        # Stimulus trace context menu
        self.stimulus_trace_ct = pn.Column(
            pn.Column(
                pn.widgets.MultiChoice(
                    name="Select stack order",
                    value=["cell_index", "repeat"],
                    options=["cell_index", "repeat", "recording", "stimulus_index"],
                    width=250,
                ),
                width=150,
            ),
            self.bin_size_input,
            self.colour_checkbox,
            self.colour_selector,
            self.selected_color,
            "Plot",
            pn.Row(
                self.action_run_button, self.figure_name_input, self.save_figure_button
            ),
        )

        # Dataframe context menu
        self.new_df_name = pn.widgets.TextInput(
            name="New dataframe name", placeholder="new_df", width=250
        )
        self.new_df_button = pn.widgets.Button(name="Create new dataframe")
        self.new_df_button.on_click(self.create_new_df)
        self.dataframe_ct = pn.Column(self.new_df_name, "Create", self.new_df_button)

        # Quality tests context menu
        self.run_quality_tests_button = pn.widgets.Button(name="Run Quality Test")
        self.run_quality_tests_button.on_click(self.run_quality_tests)
        self.quality_radio = pn.widgets.RadioButtonGroup(
            name="How",
            options=["combined", "one_by_one"],
            value="combined",
        )
        self.quality_tests_ct = pn.Column(
            "How to handle multiple stimuli",
            self.quality_radio,
            self.run_quality_tests_button,
        )

        # Recording Overview Accordion
        self.rec_overview_container = pn.Accordion(
            ("Nr cells", self.nr_cells_fig), toggle=True, active=[0]
        )

        # Plotly stats
        self.stats_plot_input = pn.widgets.Select(
            name="Select plot type",
            options=["histogram", "line", "marker", "bar", "scatter", "box", "violin"],
            width=250,
        )
        self.stat_x_input = pn.widgets.Select(name="X", options=[], width=250)
        self.stat_y_input = pn.widgets.Select(name="Y", options=[], width=250)
        self.plot_stats_button = pn.widgets.Button(name="Plot")
        self.plot_stats_button.on_click(self.plot_stats)
        self.square_plot_radio = pn.widgets.RadioButtonGroup(
            name="Square Plot", options=[0, 1]
        )
        self.stats_colour_column = pn.widgets.Select(
            name="Colour column", options=[], width=250
        )
        self.plotly_stats_ct = pn.Column(
            "Select plot type",
            self.stats_plot_input,
            self.stat_x_input,
            self.stat_y_input,
            self.stats_colour_column,
            "Square plot",
            self.square_plot_radio,
            self.plot_stats_button,
        )

        # Main

        self.recordings_object = Overview.Recording_s(analysis_path, "gui_analysis")
        self.load_button = SelectFilesButton("Recordings")
        self.load_button.observe(self.load_recording, names="files")
        self.load_analysis_button = SelectFilesButton("Analysis")
        self.load_analysis_button.observe(self.load_analysis, names="files")
        self.save_name_input = pn.widgets.TextInput(placeholder="overview", width=120)
        self.save_analysis_button = pn.widgets.Button(
            name="💾 Save Analysis",
        )
        self.save_analysis_button.on_click(self.save_analysis)

        self.spike_trains = widgets.Output()
        self.main = pn.Tabs(
            (
                "Overview",
                pn.Column(
                    "#### Recordings loaded",
                    self.recordings_dataframe_widget,
                    self.rec_overview_container,
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
            "Load previous analysis",
            self.load_analysis_button,
            pn.Row(self.save_name_input, self.save_analysis_button),
            pn.Column(self.action_menu, sizing_mode="stretch_width"),
            self.action_column,
            sizing_mode="fixed",
            width=310,
            height=1000,
            scroll=True,
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
            try:
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
                self.load_analysis_button.disabled = True
                self.nr_cells_fig.param.trigger("object")
                self.fill_dataframe_menu()
                self.fill_dataframe()

            except FileNotFoundError:
                print("No recording found")
            except AttributeError:
                print("Recording object not constructed correctly")

    def load_single_recording(self, event):
        recording_name = self.single_recording_menu.clicked
        recording = self.recordings_object.recordings[recording_name]
        self.plot_spike_counts(recording)
        self.plot_isi(recording, "times", self.isi_fig)
        self.plot_isi(recording, "cell_index", self.isi_clus_fig)
        self.plot_spike_trains(recording)

    def load_analysis(self, change):
        if change["new"]:
            self.recordings_object = Overview.Recording_s.load(
                change["new"][0] if change["new"] else ""
            )
            self.nr_cells_recording()
            self.fill_dataframe_menu()
            for recording in self.recordings_object.recordings.keys():
                self.add_recording_info_to_df(
                    self.recordings_object.recordings[recording]
                )
                self.single_recording_menu.items.append(recording)
            self.single_recording_menu.param.trigger("items")

    def save_analysis(self, change):
        self.recordings_object.save(
            Path(self.analysis_path) / self.save_name_input.value
        )

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
        self.grid_row.objects[0][1].on_edit(self.sync_dataframes)

    def on_colour_change(self, event):
        self.ct.pick_stimulus(event.new)
        self.selected_color.object = self.ct.pickstimcolour(event.new, small=True)

    def create_new_df(self, event):
        name = self.new_df_name.value
        self.recordings_object.dataframes[name] = self.grid_row.objects[0][1].value
        self.fill_dataframe_menu()

    def sync_dataframes(self, event):
        if event.new:
            df_name = self.dataframe_menu.clicked
            self.recordings_object.dataframes[df_name].update(
                self.grid_row.objects[0][1].value
            )

    def return_values_to_df(self, df, new_columns, column_dtypes):
        numeric_fill = 0
        string_fill = ""
        bool_fill = False
        df_name = self.dataframe_menu.clicked
        for column, dtype in zip(new_columns, column_dtypes):
            if dtype == "int":
                self.recordings_object.dataframes[df_name][column] = numeric_fill
                self.recordings_object.dataframes[df_name][
                    column
                ] = self.recordings_object.dataframes[df_name][column].astype(int)
            elif dtype == "float":
                self.recordings_object.dataframes[df_name][column] = numeric_fill
                self.recordings_object.dataframes[df_name][
                    column
                ] = self.recordings_object.dataframes[df_name][column].astype(float)
            elif dtype == "str":
                self.recordings_object.dataframes[df_name][column] = string_fill
                self.recordings_object.dataframes[df_name][
                    column
                ] = self.recordings_object.dataframes[df_name][column].astype(str)
            elif dtype == "bool":
                self.recordings_object.dataframes[df_name][column] = bool_fill
                self.recordings_object.dataframes[df_name][
                    column
                ] = self.recordings_object.dataframes[df_name][column].astype(bool)
        self.recordings_object.dataframes[df_name].update(df)
        self.fill_dataframe()

    def run_quality_tests(self, event):
        recordings = self.grid_row.objects[0][1].value.recording.unique()

        if self.quality_radio.value == "combined":
            for recording in recordings:
                self.recordings_object.dataframes[
                    "cell_selection"
                ] = self.grid_row.objects[0][1].value.query(
                    f"recording == '{recording}'"
                )
                spikes = self.recordings_object.get_spikes_df(
                    cell_df="cell_selection", pandas=False
                )
                # Get max time and nr_repeats
                mean_trigger_times = stimulus_spikes.mean_trigger_times(
                    self.recordings_object.recordings[recording].stimulus_df,
                    self.recordings_object.dataframes["cell_selection"]
                    .stimulus_index.unique()
                    .tolist(),
                )
                max_repeat = spikes["repeat"].max() + 1
                quality_df = quality_tests.spiketrain_qi(
                    spikes, np.sum(mean_trigger_times), max_repeat=max_repeat
                )
                insert_df = (
                    self.grid_row.objects[0][1]
                    .value.copy()
                    .set_index(["cell_index", "recording"])
                )
                insert_df["qi"] = 0
                insert_df.update(quality_df)
                insert_df = insert_df.reset_index(drop=False)
                self.return_values_to_df(insert_df, ["qi"], ["float"])

        # for recording in recordings:
        #     sub_df = self.recordings_object.dataframes["cell_selection"].query(
        #         f"recording == '{recording}'"
        #     )
        #     spikes = self.recordings_object.get_spikes_df("cell_selection")
        #
        # quality_df = quality_tests.spiketrain_qi(spikes)

    def plot_stats(self, event):
        plotting_functions = {
            "histogram": px.histogram,
            "line": px.line,
            "bar": px.bar,
            "scatter": px.scatter,
            "box": px.box,
            "violin": px.violin,
        }

        fig = plotting_functions[self.stats_plot_input.value](
            self.grid_row.objects[0][1].value,
            x=self.stat_x_input.value,
            y=self.stat_y_input.value,
            color=self.stats_colour_column.value,
        )
        fig.update_layout(template="scatter_template_jupyter")
        if self.square_plot_radio.value == 1:
            fig.update_layout(height=800, width=800)

        self.output.clear()
        self.output.append(fig)

    def action_selected(self, event):
        action = self.action_menu.clicked
        type = self.action_dict[action]["type"]
        if type == "stimulus_trace":
            self.action_column.clear()
            self.action_column.append(self.stimulus_trace_ct)
        if type == "dataframe":
            self.action_column.clear()
            self.action_column.append(self.dataframe_ct)
        if type == "quality_tests":
            self.action_column.clear()
            self.action_column.append(self.quality_tests_ct)
        if type == "plotly_stats":
            self.stat_x_input.options = [None] + self.recordings_object.dataframes[
                self.dataframe_menu.clicked
            ].columns.tolist()
            self.stat_y_input.options = [None] + self.recordings_object.dataframes[
                self.dataframe_menu.clicked
            ].columns.tolist()
            self.stats_colour_column.options = [
                None
            ] + self.recordings_object.dataframes[
                self.dataframe_menu.clicked
            ].columns.tolist()
            self.action_column.clear()
            self.action_column.append(self.plotly_stats_ct)

    def trigger_action(self, event):
        action = self.action_menu.clicked
        action_type = self.action_dict[action]["type"]
        func = self.action_dict[action]["func"]
        df_idx = self.grid_row.objects[0][1].selection
        if len(df_idx) == 0:
            df_idx = self.grid_row.objects[0][1].value.index.tolist()
        else:
            df_idx = self.grid_row.objects[0][1].value.iloc[df_idx].index
        self.recordings_object.dataframes["cell_selection"] = self.grid_row.objects[0][
            1
        ].value.loc[df_idx]
        stimulus_indices = (
            self.recordings_object.dataframes["cell_selection"]
            .stimulus_index.unique()
            .tolist()
        )
        spikes = self.recordings_object.get_spikes_df("cell_selection")
        if len(spikes) == 0:
            return

        if action_type == "stimulus_trace":
            fig = func(
                df=spikes,
                indices=self.stimulus_trace_ct.objects[0][0].value,
                bin_size=self.bin_size_input.value,
            )

            self.output.clear()  #

            if self.colour_checkbox.value:
                flash_duration = stimulus_spikes.mean_trigger_times(
                    self.recordings_object.dataframes["stimulus_df"], stimulus_indices
                )
                try:
                    fig = self.ct.add_stimulus_to_plot(fig, flash_duration)
                except TypeError:  # In case the figure gets returned as fig, ax tuple
                    fig = self.ct.add_stimulus_to_plot(fig[0], flash_duration)
            if type(fig) == plt.Figure:
                self.output.objects = [pn.pane.Matplotlib(fig, height=800, width=1500)]
            elif type(fig) == tuple:
                self.output.objects = [
                    pn.pane.Matplotlib(fig[0], height=800, width=1500)
                ]
            elif type(fig) == go.Figure:
                self.output.objects = [fig]

    def save_figure(self, event):
        if self.output.objects:
            if type(self.output.objects[0]) == pn.pane.plot.Matplotlib:
                self.output.objects[0].object.savefig(
                    Path(self.analysis_path) / f"{self.figure_name_input.value}.svg"
                )
            elif type(self.output.objects[0]) == go.Figure:
                self.output.objects[0].write_image(
                    Path(self.analysis_path) / f"{self.figure_name_input.value}.svg"
                )
            elif type(self.output.objects[0]) == pn.pane.plot.Bokeh:
                export_png(
                    self.output.objects[0].object,
                    filename=Path(self.analysis_path)
                    / f"{self.figure_name_input.value}.png",
                    width=1500,
                    height=800,
                )

    def serve(self):
        app = pn.Row(
            pn.Column(self.sidebar, sizing_mode="fixed", height=300, width=300),
            pn.Spacer(width=20),  # Adjust width for desired spacing
            pn.Column(self.main, sizing_mode="fixed", height=1000, width=1000),
            sizing_mode="stretch_width",
        )
        return app
