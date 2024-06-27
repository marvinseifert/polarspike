"""
This module provides a class for creating and managing colour templates for stimuli in plots. It reads
a DataFrame containing the colours for different stimuli.
The user can interactively select a stimulus and change the colours of the stimulus. Additional colours can be added
to the DataFrame and saved for future use. The class provides methods to add the stimulus to a plotly, matplotlib or
bokeh figure.
@Author: Marvin Seifert 2024
"""


import numpy as np
import pandas as pd
from ipywidgets import widgets, interact, Layout
import re
import math
from IPython.display import Markdown, display
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider, VBoxDivider
import mpl_toolkits.axes_grid1.axes_size as Size
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import polarspike
import bokeh
from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, Label


class Colour_template:
    """
    This class provides a template for creating and managing colour templates for stimuli in plots.

    """

    layout = {"sep": " ", "width": 4}

    Save_button = widgets.Button(
        value=False,
        description="Save colours",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    )

    def __init__(self, df_file=None):
        """
        Initialize the object.
        Parameters
        ----------
        df_file : str, optional
            The path to the DataFrame containing the colours for the stimuli. The default is None,
            which refers to: polarspike.__path__[0] + "/stim_colour_df".
        """
        if df_file is None:
            df_file = polarspike.__path__[0] + "/stim_colour_df"
        self.df_file = df_file

        self.plot_colour_dataframe = pd.read_pickle(df_file)

        self.stimulus_select = widgets.RadioButtons(
            options=list(self.plot_colour_dataframe.index),
            layout={"width": "max-content"},
            description="Colourset",
            disabled=False,
        )
        self.colours = []
        self.names = []
        self.wavelengths = []
        self.stimulus = None
        self.picker = None
        self.template = Interactive_template(self.Save_button)
        self.Save_button.on_click(self.add_new_stimulus)

    def pick_stimulus(self, name, sub_selection=None):
        """
        Pick a stimulus from the DataFrame and display the colours in a Markdown cell.
        Parameters
        ----------
        name : str
            The name of the stimulus.
        sub_selection : tuple, optional
            A tuple containing the start index and the step size for subsetting the colours. The default is None as in
            no sub-selection.
        """
        self.stimulus = name
        self.colours = self.get_stimulus_colors(name, sub_selection)
        self.names = self.get_stimulus_names(name, sub_selection)
        self.wavelengths = self.get_stimulus_wavelengths(name, sub_selection)
        return self.colour_to_markdown()

    def colour_to_markdown(self):
        """
        Display a selected colour template in a Markdown cell.

        """
        display(
            Markdown(
                self.layout["sep"].join(
                    f'<span style="font-family: monospace">{color} <span style="color: {color}">{chr(9608) * self.layout["width"]}</span></span>'
                    for color in self.colours
                )
            )
        )

    def get_stimulus_colors(self, name, sub_selection=None):
        """
        Get the colours for a stimulus from the DataFrame without "selecting" it.
        Parameters
        ----------
        name : str
            The name of the stimulus.
        sub_selection : tuple, optional
            A tuple containing the start index and the step size for subsetting the colours. The default is None as in
            no sub-selection.

        """
        colours = np.asarray(self.plot_colour_dataframe.loc[name]["Colours"])
        if sub_selection is not None:
            colours = colours[sub_selection[0] :: sub_selection[1]]

        return colours

    def get_stimulus_names(self, name, sub_selection=None):
        """
        Get the names for a stimulus from the DataFrame without "selecting" it.
        Parameters
        ----------
        name : str
            The name of the stimulus.
        sub_selection : tuple, optional
            A tuple containing the start index and the step size for subsetting the names. The default is None as in
            no sub-selection.

        """
        names = np.asarray(
            self.plot_colour_dataframe.loc[name]["Description"], dtype="<U11"
        )
        if sub_selection is not None:
            names = names[sub_selection[0] :: sub_selection[1]]

        return names

    def get_stimulus_wavelengths(self, name, sub_selection=None):
        """
        Get the wavelengths for a stimulus from the DataFrame without "selecting" it.
        Parameters
        ----------
        name : str
            The name of the stimulus.
        sub_selection : tuple, optional
            A tuple containing the start index and the step size for subsetting the wavelengths. The default is None as in
            no sub-selection.
        """
        names = self.get_stimulus_names(name, sub_selection)
        wavelengths = []
        for name in names:
            try:
                wavelengths.append(int(re.findall(r"\d+", name)[0]))
            except IndexError:
                wavelengths.append(name)
        return wavelengths

    def list_stimuli(self):
        """
        List all stimuli in the DataFrame.

        Returns
        -------
        numpy.ndarray
            An array containing the names of the stimuli.
        """
        return self.plot_colour_dataframe.index.to_numpy()

    def add_new_stimulus(self, button):
        """
        Callback function to add a new stimulus to the DataFrame and save it.
        Parameters
        ----------
        button : ipywidgets.widgets.widget_button.Button
            The button that was clicked.

        """
        self.plot_colour_dataframe.loc[
            self.template.name, "Colours"
        ] = self.template.colours
        self.plot_colour_dataframe.loc[
            self.template.name, "Description"
        ] = self.template.description
        self.plot_colour_dataframe.to_pickle(self.df_file)

    def interactive_stimulus(self):
        """
        Create an interactive widget for selecting a stimulus and changing the colours temporarily.

        Returns
        -------
        ipywidgets.widgets.interaction.interact
            The widget for selecting a stimulus.
        """
        interact(self.pickstimcolour, selected_stimulus=self.stimulus_select)

    def pickstimcolour(self, selected_stimulus=None, small=False):
        """
        Creates RadioButtons for selecting a stimulus and ColorPickers for changing the colours of the stimulus.
        Parameters
        ----------
        selected_stimulus : str, optional
            The name of the stimulus that is selected. The default is None.
        small : bool, optional
            If True, the ColorPickers will be displayed without the hexcolor string next to it.
            The default is False.

        """
        if not selected_stimulus:
            self.stimulus = self.stimulus_select.value
        else:
            self.stimulus = selected_stimulus
        words = []
        nr_trigger = len(self.plot_colour_dataframe.loc[self.stimulus]["Colours"])

        for trigger in range(nr_trigger):
            selected_word = self.plot_colour_dataframe.loc[self.stimulus][
                "Description"
            ][trigger]
            words.append(selected_word)

        items = [
            widgets.ColorPicker(
                description=w,
                value=self.plot_colour_dataframe.loc[self.stimulus]["Colours"][trigger],
                concise=small,
            )
            for w, trigger in zip(words, range(nr_trigger))
        ]
        first_box_items = []
        # second_box_items = []
        # third_box_items = []
        # fourth_box_items = []

        a = 0
        while True:
            try:
                first_box_items.append(items[a])
                a = a + 1
                # second_box_items.append(items[a])
                # a = a + 1
                # third_box_items.append(items[a])
                # a = a + 1
                # fourth_box_items.append(items[a])
                # a = a + 1
            except IndexError:
                break

        first_box = widgets.VBox(
            first_box_items,
            # layout=Layout(overflow="scroll", width="100%", height="100%"),
        )
        # second_box = widgets.VBox(second_box_items)
        # third_box = widgets.VBox(third_box_items)
        # fourth_box = widgets.VBox(fourth_box_items)
        self.picker = widgets.HBox(
            [first_box],
            # Prevent scrolling
            layout=Layout(height="300px", width="200px"),
        )

        return self.picker

    # def changed_selection(self):
    #     a = 0
    #     self.colours = []
    #     nr_trigger = len(self.plot_colour_dataframe.loc[self.stimulus]["Colours"])
    #     for trigger in range(math.ceil(nr_trigger)):
    #         try:
    #             self.colours.append(self.picker.children[0].children[trigger].value)
    #             a = a + 1
    #             # self.colours.append(self.picker.children[1].children[trigger].value)
    #             # a = a + 1
    #             # self.colours.append(self.picker.children[2].children[trigger].value)
    #             # a = a + 1
    #             # self.colours.append(self.picker.children[3].children[trigger].value)
    #             # a = a + 1
    #
    #             self.names = self.plot_colour_dataframe.loc[self.stimulus][
    #                 "Description"
    #             ]
    #         except IndexError:
    #             break
    #     return self.colour_to_markdown()

    def create_stimcolour(self, name, nr_colours, description):
        """
        Interactive widget for creating a new stimulus with colours. The colours will displayed interactively in Jupyter
        so that the user can choose a custom colour for each stimulus.
        Parameters
        ----------
        name : str
            The name of the stimulus.
        nr_colours : int
            The number of colours for the stimulus.
        description : list
            A list of descriptions for the colours.

        Returns
        -------
        ipywidgets.widgets.widget_box.VBox
            Vbox containing other widgets for creating a new stimulus.

        """
        return self.template.create_stimcolour(name, nr_colours, description)

    def add_stimulus_to_plot(self, initial_fig, flash_durations, names=True):
        """
        This adds a colour stimulus plot under an existing plotly, matplotlib or bokeh figure.
        In case of Matplotlib, the colour stimulus will be plotted into the last row of subplots.
        In case of Plotly, the colour stimulus will be plotted into a new subplot below the existing plot.
        In case of Bokeh, the colour stimulus will be plotted below the existing plot.

        Parameters
        ----------
        initial_fig : plotly.graph_objs._figure.Figure or matplotlib.figure.Figure or bokeh.models.plots.GridPlot
            The initial figure to which the stimulus will be added.
        flash_durations : list
            A list of durations for the rectangles, indicating their width.
        names : bool, optional
            If True, the names of the stimuli will be displayed on top of each stimulus. The default is True.


        """
        if names:
            names = self.names
        else:
            names = None

        if type(initial_fig) == go.Figure:
            fig = add_stimulus_to_plotly(
                initial_fig, self.colours, flash_durations, names=names
            )
            return fig
        elif type(initial_fig) == plt.Figure:
            fig = add_stimulus_to_matplotlib(
                initial_fig, self.colours, flash_durations, names=names
            )
            return fig
        elif type(initial_fig) == bokeh.models.plots.GridPlot:
            fig = add_stimulus_to_bokeh(
                initial_fig, self.colours, flash_durations, names=names
            )
            return fig
        else:
            raise TypeError(
                f"The type of the initial figure {type(initial_fig)} is not supported. Please use a plotly, matplotlib or bokeh figure."
            )


class Interactive_template:
    """
    This class creates a fully interactive widget for creating a new stimulus with colours.
    Each colour can be changed interactively by the user. Eventually, the user can save the new stimulus to the
    DataFrame.
    """

    Pick_button = widgets.Button(
        value=False,
        description="Set colours",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    )

    def __init__(self, save_button):
        self.name = ""
        self.nr_colours = 0
        self.description = []
        self.new_colour_select_box = widgets.HBox()
        self.Pick_button.on_click(self.get_colours)
        self.Save_button = save_button
        self.colours = []

    def create_stimcolour(self, name, nr_colours, description):
        """

        Parameters
        ----------
        name : str
            The name of the stimulus.
        nr_colours : int
            The number of colours for the stimulus.
        description : list
            A list of descriptions for the colours.

        Returns
        -------
        ipywidgets.widgets.widget_box.VBox
            Vbox containing other widgets for creating a new stimulus.

        """
        self.name = name
        self.nr_colours = nr_colours
        self.description = description

        items = [
            widgets.ColorPicker(
                description=w,
                value="#000000",
            )
            for w, trigger in zip(self.description, range(self.nr_colours))
        ]
        first_box_items = []
        second_box_items = []
        third_box_items = []
        fourth_box_items = []

        a = 0
        while True:
            try:
                first_box_items.append(items[a])
                a = a + 1
                second_box_items.append(items[a])
                a = a + 1
                third_box_items.append(items[a])
                a = a + 1
                fourth_box_items.append(items[a])
                a = a + 1
            except IndexError:
                break

        first_box = widgets.VBox(first_box_items)
        second_box = widgets.VBox(second_box_items)
        third_box = widgets.VBox(third_box_items)
        fourth_box = widgets.VBox(fourth_box_items)
        container = widgets.HBox([first_box, second_box, third_box, fourth_box])
        self.new_colour_select_box = widgets.VBox(
            [self.Pick_button, self.Save_button, container]
        )

        return self.new_colour_select_box

    def get_colours(self, value):
        """
        Callback function to get the colours from the widget.

        Parameters
        ----------
        value : ipywidgets.widgets.widget_button.Button
            The button that was clicked.

        """
        self.colours = []

        for trigger in range(math.ceil(self.nr_colours / 4)):
            try:
                self.colours.append(
                    self.new_colour_select_box.children[2]
                    .children[0]
                    .children[trigger]
                    .value
                )
                self.colours.append(
                    self.new_colour_select_box.children[2]
                    .children[1]
                    .children[trigger]
                    .value
                )
                self.colours.append(
                    self.new_colour_select_box.children[2]
                    .children[2]
                    .children[trigger]
                    .value
                )
                self.colours.append(
                    self.new_colour_select_box.children[2]
                    .children[3]
                    .children[trigger]
                    .value
                )
            except IndexError:
                break
            except TypeError:
                print("Here")

        return self.colours


def add_stimulus_to_plotly(initial_fig, colours, flash_durations, names=None):
    """
    Adds a subplot with colored rectangles to an existing figure.

    Parameters
    ----------
    initial_fig : plotly.graph_objs._figure.Figure
        The initial figure to which the subplot will be added.
    colours : list
        A list of colors for the rectangles.
    flash_durations : list
        A list of durations for the rectangles, indicating their width.
    names : list, optional
        A list of names for the rectangles. The default is None.
    Returns
    -------
    plotly.graph_objs._figure.Figure
        The modified figure with the original plot and the added rectangle subplot.
    """

    # Extract data and layout from the initial figure
    initial_data = initial_fig.data
    initial_layout = initial_fig.layout
    initial_y_label = initial_layout["yaxis"]["title"]["text"]

    # Rectangle dimensions

    height = 1

    # Create a new 2x1 subplot layout
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1, 0.1])

    # Add the extracted data to the first subplot
    for trace in initial_data:
        fig.add_trace(trace, row=1, col=1)

    # Add the rectangles to the second subplot using shapes
    x_start = 0
    for i, (color, width) in enumerate(zip(colours, flash_durations)):
        fig.add_shape(
            go.layout.Shape(
                type="rect",
                x0=x_start,
                y0=0,
                x1=x_start + width,
                y1=height,
                fillcolor=color,
                line=dict(width=0),
            ),
            row=2,
            col=1,
        )

        if names is not None and len(names) == len(colours):
            fig.add_annotation(
                x=x_start + width / 2,
                y=height / 2,
                xref="x2",
                yref="y2",
                text=names[i],
                showarrow=False,
                font=dict(size=14),
            )

        x_start += width

    # Merge the layouts
    # fig.update_layout(initial_layout)

    # Further update layout as needed
    fig.update_layout(
        xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
        yaxis2=dict(
            range=[0, height], showgrid=False, zeroline=False, showticklabels=False
        ),
    )
    fig.update_xaxes(title_text="Time (s)", row=2)
    fig.update_yaxes(title_text="Repeats", row=2)
    fig.update_yaxes(title_text=initial_y_label, row=1)
    fig.update_xaxes(title_text="", row=1)
    fig.update_layout(template="simple_white")

    return fig


def add_stimulus_to_matplotlib(initial_fig, colours, flash_durations, names=None):
    """
    Adds a subplot with colored rectangles to an existing Matplotlib figure.

    Parameters:
        - initial_fig: The initial Matplotlib figure to which the subplot will be added.
        - colours: A list of colors for the rectangles.
        - flash_durations: A list of durations for the rectangles, indicating their width.
        - names: (Optional) A list of names for the rectangles.

    Returns:
        - The modified figure with the original plot and the added rectangle subplot.
    """
    # Use constrained layout to handle the spacing automatically
    # initial_fig.set_constrained_layout(True)

    # Create the new subplot for the stimulus with shared x-axis
    axs = np.asarray(initial_fig.get_axes()).reshape(
        initial_fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    )
    ax_stimulus = axs[-1, 0]
    original_axes = axs[:-1, 0]
    # for ax in original_axes:
    #     ax.set_xlabel("")
    org_pos = original_axes[-1].get_position()
    stim_pos = ax_stimulus.get_position()

    ax_stimulus.set_position([org_pos.x0, stim_pos.y0, org_pos.width, stim_pos.height])

    # Add rectangles to the new subplot
    x_position = 0
    for color, width in zip(colours, flash_durations):
        rect = patches.Rectangle(
            (x_position, 0), width, 1, linewidth=0, edgecolor="none", facecolor=color
        )
        ax_stimulus.add_patch(rect)
        x_position += width

    # Add text if names are provided
    if names is not None and len(names) == len(colours):
        aspect = ax_stimulus.get_data_ratio()
        x_position = 0
        for i, width in enumerate(flash_durations):
            if calculate_luminance(hex_to_rgb(colours[i])) < 128:
                text_color = "white"
            else:
                text_color = "black"
            ax_stimulus.text(
                x_position + width / 2,
                0.5,
                names[i],
                ha="center",
                va="center",
                fontsize=12**2 * aspect,
                color=text_color,
            )
            x_position += width

    # Adjust the layout
    # initial_fig.tight_layout()

    return initial_fig


def add_stimulus_to_bokeh(initial_fig, colours, flash_durations, names=None):
    """
    Adds a subplot with colored rectangles to an existing Bokeh figure.

    Parameters
    ----------
    initial_fig : bokeh.models.plots.GridPlot
        The initial Bokeh figure to which the subplot will be added.
    colours : list
        A list of colors for the rectangles.
    flash_durations : list
        A list of durations for the rectangles, indicating their width.
    names : list, optional
        A list of names for the rectangles.

    Returns
    -------
    bokeh.models.plots.GridPlot
        The modified figure with the original plot and the added rectangle subplot.

    """
    # Assuming 'height', 'colours', 'flash_durations', and optionally 'names' are defined
    # Create a new plot
    height = 2
    width = initial_fig.children[0][0].width
    fig = figure(
        width=width,
        height=50,
        x_range=initial_fig.children[0][0].x_range,
        sizing_mode="fixed",
    )

    x_start = 0
    xs = []  # List to hold x coordinates of all patches
    ys = []  # List to hold y coordinates of all patches
    for i, (color, width) in enumerate(zip(colours, flash_durations)):
        # Define the coordinates for each rectangle (patch)
        xs.append([x_start, x_start, x_start + width, x_start + width])
        ys.append([0, height, height, 0])

        # Add labels if names are provided
        if names is not None and len(names) == len(colours):
            # Check if colour is dark:
            if calculate_luminance(hex_to_rgb(color)) < 128:
                text_color = "white"
            else:
                text_color = "black"

            label = Label(
                x=x_start + width / 2,
                y=height / 2,
                text=names[i],
                text_font_size="11pt",
                text_align="center",
                text_color=text_color,
            )
            fig.add_layout(label)

        x_start += width

    # Add the patches to the figure
    fig.patches(xs, ys, color=colours, alpha=0.8, line_width=2)
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.xaxis.major_tick_line_color = None
    fig.yaxis.major_tick_line_color = None
    fig.xaxis.minor_tick_line_color = None
    fig.yaxis.minor_tick_line_color = None
    fig.xaxis.major_label_text_font_size = "0pt"
    fig.yaxis.major_label_text_font_size = "0pt"
    fig.yaxis.axis_label = "Stimulus"
    fig.yaxis.axis_label_orientation = 0

    rows = len(initial_fig.children)
    initial_fig.children.append((fig, rows, 0))

    return initial_fig


def hex_to_rgb(hex_color):
    """
    Convert a hex color code to an RGB tuple.

    Parameters
    ----------
    hex_color : str
        The hex color code to convert.

    """
    # Convert hex to RGB
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def calculate_luminance(rgb):
    """
    Calculate the luminance of an RGB color.

    Parameters
    ----------
    rgb : tuple
        The RGB color tuple.
    """
    # Calculate luminance using the formula
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
