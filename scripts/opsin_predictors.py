from polarspike import Opsins
import numpy as np
import pandas as pd
from polarspike import colour_template

# %%
ct = colour_template.Colour_template()
ct.pick_stimulus("FFF_8_MC")
leds = ct.wavelengths

# %%
wavelengths = np.unique(leds)

# %% plot this with the oil droplet sensitivity curves
double_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_double")
single_ab_df = pd.read_pickle(r"/mnt/workdrive/Chicken_24/opsins_oil")
combined_df = pd.concat([double_ab_df, single_ab_df])
combined_df = combined_df.sort_values(["cone", "wavelength"])
combined_df["absorption"] = combined_df["absorption"] * np.e
combined_df.loc[combined_df["absorption"] < 0] = 0
combined_df = combined_df.set_index("wavelength")
subset_df = combined_df.loc[wavelengths]
# %%
subset_df.to_pickle(r"/home/mawa/PycharmProjects/linear_nonlinear_model/test_predictor")
# %% Or get animal specific opsins
Ops = Opsins.Opsin_template()
templates = Ops.govardovskii_animal("Chicken").T
# this returns an array 400 x 4 for wavelength 300-700 nm and 4 opsins need to export it to a dataframe with
# one column for wavelength and one column for absorption and one column with the cone names
cone_names = ["LWS"] * 400 + ["MWS"] * 400 + ["SWS2"] * 400 + ["SWS1"] * 400
combined_df = pd.DataFrame(
    data={
        "wavelength": np.tile(np.arange(300, 700, 1), 4),
        "absorption": templates.flatten(),
        "cone": cone_names,
    }
)
# %% log transform the absorption
combined_df = combined_df.sort_values(["cone", "wavelength"])
combined_df["absorption"] = combined_df["absorption"] * np.e
combined_df.loc[combined_df["absorption"] < 0.05, "absorption"] = 0.05
combined_df.loc[combined_df["absorption"] == np.NaN, "absorption"] = 0.05
combined_df["absorption"] = np.log(combined_df["absorption"])
# combined_df.loc[combined_df["absorption"] <= np.log(10e-10), "absorption"] = -np.inf
# renormalize to range 0-1
combined_df["absorption"] = (
                                    combined_df["absorption"] - np.nanmin(combined_df["absorption"])
                            ) / (np.nanmax(combined_df["absorption"]) - np.nanmin(combined_df["absorption"]))
combined_df.loc[combined_df["absorption"] < 0] = 0
combined_df = combined_df.set_index("wavelength")
subset_df = combined_df.loc[wavelengths]
# %%
subset_df.to_pickle(
    r"/home/mawa/PycharmProjects/linear_nonlinear_model/train_predictor"
)
