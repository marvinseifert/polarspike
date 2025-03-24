import matplotlib.pyplot as plt
import panel as pn
import pandas as pd
import numpy as np

pn.extension(design=pn.theme.Material)


plot_opts = dict(responsive=True, min_height=400)

template = pn.template.SlidesTemplate(
    title="The slide title",
    logo="https://raw.githubusercontent.com/holoviz/panel/main/doc/_static/logo_stacked.png",
)

template.main.extend(
    [
        pn.pane.Markdown(
            "Slides Template", styles={"font-size": "3em"}, align="center"
        ),
        pn.Card(
            pn.pane.Image(r"D:\slide_app_presentation\spike_animation.gif", embed=True),
            title="Sine",
            margin=20,
            tags=["fragment"],
        ),
    ]
)

template.servable()
