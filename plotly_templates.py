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
            title=dict(standoff=1)
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="white",
            ticks="outside",
            tickfont=dict(family="Arial", color="white"),
            titlefont=dict(family="Arial", color="white"),
            showgrid=False,
            title=dict(standoff=1)
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
