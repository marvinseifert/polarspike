import Overview
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import stimulus_trace
import plotly.graph_objects as go
import numpy as np
import panel as pn
import param
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import time


class Explorer:
    def __init__(self, x, y):
        self.stimulus_input = pn.widgets.FileInput(sizing_mode='stretch_width', accept=".brw, .h5")
        self.recording_input = pn.widgets.FileInput(sizing_mode='stretch_width', accept=".brw, .dat")

        self.text = """ ## Single Recording Overview.
        This UI allows you to explore a single MEA recording fast and interactively.

        """
        self.load_button = pn.widgets.Button(name="Load Recording", button_type="primary", width=150)
        self.load_button.on_click(self.load_recording)

        # Creating an instance of PlotApp
        self.plot_app = PlotApp(x, y)

        # Replacing the Plotly figure with the Bokeh plot from PlotApp
        self.stim_figure = self.stim_figure = pn.panel(self.plot_app.plot)

        self.sidebar = pn.layout.WidgetBox(
            pn.pane.Markdown(self.text, margin=(0, 10)),
            "Stimulus File",
            self.stimulus_input,
            "Recording File",
            self.recording_input,
            self.load_button,
            max_width=300,
            sizing_mode='stretch_width'
        ).servable(area='sidebar')

        self.main = pn.Tabs(
            ('Stimulus', pn.Column(
                pn.Row(
                    self.stim_figure,
                    sizing_mode='stretch_both'
                ))))

    def run(self):
        app = pn.Row(
            pn.Column(self.sidebar, sizing_mode='fixed', width=300),
            pn.Spacer(width=20),  # Adjust width for desired spacing
            pn.Column(self.main, sizing_mode='stretch_width'),
            sizing_mode='stretch_both'
        )
        return app.servable(title="Single Recording Overview")

    def load_recording(self, event):

        return


class PlotApp(param.Parameterized):
    switch = param.Boolean(default=False, precedence=-1)

    def __init__(self, x=None, y=None):
        super().__init__()
        if x is None:
            x = np.arange(100)
        if y is None:
            y = np.zeros(100)
        self.begins = np.zeros(x.shape[0], dtype=bool)
        self.ends = np.zeros(x.shape[0], dtype=bool)
        self.source = ColumnDataSource(data=dict(x=x, y=y, color=['blue'] * x.shape[0], size=[5] * x.shape[0]))
        self.plot = figure(tools='tap,pan,wheel_zoom,box_zoom,reset', width=1000, height=400, title='Click on a point...',
                           output_backend="webgl")
        self.plot.line('x', 'y', source=self.source, line_width=2, line_color='blue')
        self.scatter_renderer = self.plot.circle('x', 'y', color='color', size='size', source=self.source,
                                                 # Ensure non-selected points remain visible
                                                 nonselection_alpha=1.0)

        self.plot_pane = pn.pane.Bokeh(self.plot)
        self.source.selected.on_change('indices', self.callback)
        self.app_layout = pn.Column(self.plot_pane)
        self.switch = True
        self.last_callback_time = time.time()

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