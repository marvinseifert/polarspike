"""
This module contains custom templates for Plotly plots. You can use them by importing this module and setting the template
of your plotly figure to one of the templates defined in this module.
Templates:
- scatter_template: A white template with black axes and white ticks.
- scatter_template_black: A black template with white axes and black ticks.
- scatter_template_jupyter: A template for Jupyter notebooks with a transparent background and white axes.
- bar_template: A white template with black axes and white ticks for bar plots.
@Marvin Seifert 2024
"""

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates["scatter_template"] = go.layout.Template(
    layout=go.Layout(
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickfont=dict(family="Arial"),
            titlefont=dict(family="Arial"),
            showgrid=False,
            zeroline=False,
            title=dict(standoff=1),
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickfont=dict(family="Arial"),
            titlefont=dict(family="Arial"),
            showgrid=False,
            zeroline=False,
            title=dict(standoff=1),
        ),
        plot_bgcolor="rgba(255, 255, 255, 255)",
        paper_bgcolor="rgba(255, 255, 255, 255)",
        margin_pad=5,
    )
)

pio.templates["scatter_template_black"] = go.layout.Template(
    layout=go.Layout(
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="white",
            ticks="outside",
            tickfont=dict(family="Arial", color="white"),
            titlefont=dict(family="Arial", color="white"),
            showgrid=False,
            title=dict(standoff=1),
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="white",
            ticks="outside",
            tickfont=dict(family="Arial", color="white"),
            titlefont=dict(family="Arial", color="white"),
            showgrid=False,
            title=dict(standoff=1),
        ),
        plot_bgcolor="rgba(0, 0, 0, 255)",
        paper_bgcolor="rgba(0, 0, 0, 255)",
        margin_pad=10,
    )
)

pio.templates["scatter_template_jupyter"] = go.layout.Template(
    layout=go.Layout(
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="white",
            ticks="outside",
            tickfont=dict(family="Arial", color="white"),
            titlefont=dict(family="Arial", color="white"),
            showgrid=False,
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="white",
            ticks="outside",
            tickfont=dict(family="Arial", color="white"),
            titlefont=dict(family="Arial", color="white"),
            showgrid=False,
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin_pad=10,
        colorway=px.colors.qualitative.Dark2,
    )
)


pio.templates["bar_template"] = go.layout.Template(
    layout=go.Layout(
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickfont=dict(family="Arial"),
            titlefont=dict(family="Arial"),
            showgrid=False,
            zeroline=False,
            title=dict(standoff=1),
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickfont=dict(family="Arial"),
            titlefont=dict(family="Arial"),
            showgrid=False,
            zeroline=False,
            title=dict(standoff=1),
        ),
        plot_bgcolor="rgba(255, 255, 255, 255)",
        paper_bgcolor="rgba(255, 255, 255, 255)",
        margin_pad=0,
    )
)
