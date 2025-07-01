from polarspike.Rec_UI import Recording_explorer
import panel as pn
pn.extension(debug=True)
# %%
rec_ex = Recording_explorer(r"~/data_disk/combined_analysis")
if __name__ == "__main__":
    pn.serve(rec_ex.serve(), show=True, port=5006)