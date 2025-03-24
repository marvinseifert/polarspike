from polarspike import Opsins, colour_template
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
import numpy as np

# %% create a gaussian window with size 20
led_positions = [610, 560, 535, 460, 413, 365]
led_x = []
for idx, pos in enumerate(led_positions):
    led_x.append(np.arange(pos - 50, pos + 50))
window = gaussian(100, 5)
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(window)
fig.show()
# %%
CT = colour_template.Colour_template()
CT.pick_stimulus("FFF_6_MC")
colours = CT.colours[::2]
cone_colours = [
    colours[0],
    colours[2],
    colours[4],
    colours[5],
    "darkgoldenrod",
    "orange",
    "saddlebrown",
]
cone_color_dict = {
    "LWS": colours[0],
    "MWS": colours[2],
    "SWS2": colours[4],
    "SWS1": colours[5],
    "principal": "orange",
    "accessory": "darkgoldenrod",
}


# %%
OP = Opsins.Opsin_template()
templates = OP.govardovskii_animal("Human")
double_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_double")
single_ab_df = pd.read_pickle(r"D:\Chicken_24\opsins_oil")
combined_df = pd.concat([double_ab_df, single_ab_df])
combined_df = combined_df.sort_values(["cone", "wavelength"])
# %%
fig, axs = plt.subplots(nrows=2, figsize=(10, 20))
for idx, cone in enumerate(combined_df["cone"].unique()):
    cone_df = combined_df.query(f"cone == '{cone}'")
    axs[1].plot(
        cone_df["wavelength"],
        cone_df["absorption"],
        color=cone_color_dict[cone],
        label=cone,
        linewidth=3,
    )
    # axs[1].fill_between(
    #     led_x[idx],
    #     window,
    #     color=CT.colours[::2][idx],
    #     # fill to the y=0 line
    # )
    # axs[0].fill_between(
    #     led_x[idx],
    #     window,
    #     color=CT.colours[::2][idx],
    #     # fill to the y=0 line
    # )
for idx, template in enumerate(templates.T):
    axs[0].plot(
        cone_df["wavelength"],
        template,
        color=cone_colours[idx],
        linewidth=3,
    )

axs[1].set_xlabel("Wavelength (nm)", fontsize=20)
axs[1].set_ylabel("Absorption", fontsize=20)
# remove the top and right spines from plot
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
# change tick mark size to 12
axs[1].tick_params(axis="both", which="major", labelsize=14)
# remove tick labels from top subplots
axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
#
fig.savefig(
    r"C:\Users\Marvin\Documents\ISER_presentation\opsins_absorption_human.png",
    dpi=300,
)
