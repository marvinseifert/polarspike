from polarspike.Rec_UI import Recording_explorer, Explorer
import panel as pn

pn.extension('ipywidgets')
pn.extension(debug=True)
# %%
rec_ex = Explorer()
if __name__ == "__main__":
    pn.serve(rec_ex.run(), show=True, port=5006)
