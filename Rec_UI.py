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



class Explorer:
    def __init__(self):
        self.stimulus_input = SelectFilesButton('Stimulus File')
        self.recording_input = SelectFilesButton('Recording File')
        self.stimulus_input.observe(self._on_stimulus_selected, names='files')
        self.recording_input.observe(self._on_recording_selected, names='files')
        self.stimulus = None
        self.text = """ ## Single Recording Overview.
        This UI allows you to explore
        a single MEA recording fast
        and interactively.

        """
        self.load_button = pn.widgets.Button(name="Load Recording", button_type="primary", width=150)
        self.load_button.on_click(self.load_recording)

        self.define_stimuli_button = pn.widgets.Button(name="Define Stimuli", button_type="primary", width=150)
        self.define_stimuli_button.on_click(self.define_stimuli)


        # Creating an instance of PlotApp
        self.plot_app = PlotApp()

        # Replacing the Plotly figure with the Bokeh plot from PlotApp
        self.stim_figure = self.stim_figure = pn.panel(self.plot_app.plot, width=1000, height=500)
        self.frequency_input = pn.widgets.FloatInput(name='Recording Frequency', value=1, step=1e-1, start=0., end=50000.)
        self.stimulus_df = pn.panel(None, width=1000, height=100)
        self.sidebar = pn.layout.WidgetBox(
            pn.pane.Markdown(self.text, margin=(0, 10)),
            "Stimulus File",
            self.stimulus_input,
            "Recording File",
            self.recording_input,
            self.frequency_input,
            self.load_button,
            max_width=300,
            sizing_mode='stretch_width'
        ).servable(area='sidebar')

        self.main = pn.Tabs(
            ('Stimulus', pn.Column(
                pn.Row(
                    self.stim_figure,
                    self.define_stimuli_button,
                    height=600
                ),
                pn.Row(self.stimulus_df, height=200)
            )),
            ("Spikes", pn.Column()),
            ("Stimuli", pn.Column()),
            height=600
        )

    def run(self):
        app = pn.Row(
            pn.Column(self.sidebar, sizing_mode='fixed', width=300),
            pn.Spacer(width=20),  # Adjust width for desired spacing
            pn.Column(self.main, sizing_mode='stretch_width'),
            width=1000, height=800
        )
        return app

    def load_recording(self, event):
        self.stimulus = stimulus_trace.Stimulus_Extractor(self.stimulus_file, self.frequency_input.value, 0)
        channel = self.stimulus.downsample("500ms")

        # Debugging: Print or log the loaded data to ensure it's what you expect
        #print(channel)

        self.plot_app.update_plot(channel["Frame"].to_numpy(), channel["Voltage"].to_numpy())

        #print("Before update: ", self.stim_figure.object)  # Debug
        self.stim_figure.object = self.plot_app.plot
        #print("After update: ", self.stim_figure.object)  # Debug

    def _on_stimulus_selected(self, change):
        self.stimulus_file = change['new'][0] if change['new'] else ''
        #print(f"File selected: {self.stimulus_file}")
    def _on_recording_selected(self, change):
        self.recording_file = change['new'][0] if change['new'] else ''
        #print(f"File selected: {self.recording_file}")

    def define_stimuli(self, event):
        begins, ends = self.plot_app.get_frames()
        stim_df = self.stimulus.get_stim_range_new(begins, ends)
        self.stimulus_df.object = pn.widgets.Tabulator(stim_df, name='Stimuli', theme="simple", widths=150)


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
        self.source = ColumnDataSource(data=dict(x=self.x, y=y, color=['blue'] * self.x.shape[0],
                                                 size=[5] * self.x.shape[0]))
        self.plot = figure(tools='tap,pan,wheel_zoom,box_zoom,reset', width=1200, height=500,
                           title='Stimulus selection', output_backend="webgl")

        # Add glyphs to the plot
        self.plot.line('x', 'y', source=self.source, line_width=2, line_color='blue')
        self.scatter_renderer = self.plot.circle('x', 'y', color='color', size='size', source=self.source,
                                                 nonselection_alpha=1.0)

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
        self.source.data = dict(x=self.x, y=y, color=['blue'] * self.x.shape[0], size=[5] * self.x.shape[0])
        #print(x)
        # Setup callback for selection changes
        self.source.selected.on_change('indices', self.callback)
        # Create a new figure object

    def callback(self, attr, old, new):
        current_time = time.time()
        try:
            elapsed_time = current_time - self.last_callback_time
        except AttributeError:
            elapsed_time = 1  # First callback, so allow it
        self.last_callback_time = current_time
        if elapsed_time < 0.5:  # If less than 0.5 seconds elapsed since last callback, ignore
            print("Callback debounced")
            return
        if new:
            colors = list(self.source.data['color'])
            sizes = list(self.source.data['size'])  # Added this line to get current sizes

            # Get the index of the last selected point
            idx = new[-1]
            if colors[idx] == 'blue' and self.switch:
                colors[idx] = 'green'
                sizes[idx] = 10  # Added this line to change size
                self.begins[idx] = True
                self.switch = False
            elif colors[idx] == "blue" and not self.switch:
                colors[idx] = 'red'
                sizes[idx] = 10  # Added this line to change size
                self.ends[idx] = True
                self.switch = True
            elif colors[idx] == "red" and self.switch:
                colors[idx] = 'blue'
                sizes[idx] = 5  # Added this line to revert size
                self.ends[idx] = False
                self.switch = False
            elif colors[idx] == "green" and not self.switch:
                colors[idx] = 'blue'
                sizes[idx] = 5  # Added this line to revert size
                self.begins[idx] = False
                self.switch = True

            # Update source data
            self.source.data['color'] = colors
            self.source.data['size'] = sizes  # Added this line to update sizes in source
            self.source.selected.indices = []

    def get_frames(self):
        return self.x[self.begins], self.x[self.ends]
