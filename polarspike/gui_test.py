"""
This script can be used to test the GUI of the polarspike project within Pycharm. For this to work, the Pycharm run
parameters have to be set to the following:
module: panel
below module window:
serve $FilePath$
working directory: polarspike directory
Environment variables: PYTHONUNBUFFERED=1; PYTHONBUFFERED=1

Works on Pycharm 2024.3.2


"""


from polarspike import Rec_UI

import panel as pn


pn.extension("tabulator")
pn.extension("plotly")
pn.extension(design="material")

from importlib import reload
from polarspike import grid

analysis_path = "D:/combined_analysis"
explorer = Rec_UI.Recording_explorer(analysis_path)
explorer.serve().servable()
