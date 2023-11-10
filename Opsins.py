import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly_templates


class Opsin_template:
    opsins_peaks = {}
    opsins_peaks["Chicken"] = np.array([570, 508, 455, 419])
    opsins_peaks["Zebrafish"] = np.array([548, 476, 416, 355])
    opsins_peaks["Human"] = np.array([564, 534, 420])
    opsins_peaks["Mouse"] = np.array([508, 360])
    opsins_peaks["Frog"] = np.array([611, 523, 444])

    opsins_types = {}
    opsins_types["Chicken"] = ["LWS", "MWS", "SWS2", "SWS1"]
    opsins_types["Zebrafish"] = ["LWS", "MWS", "SWS2", "SWS1"]
    opsins_types["Human"] = ["LWS", "MWS", "SWS1"]
    opsins_types["Mouse"] = ["MWS", "SWS1"]
    opsins_types["Frog"] = ["LWS", "MWS", "SWS1"]

    cone_numbers = {}
    cone_numbers["Chicken"] = int(4)
    cone_numbers["Zebrafish"] = int(4)
    cone_numbers["Human"] = int(3)
    cone_numbers["Mouse"] = int(2)
    cone_numbers["Frog"] = int(3)

    cone_colours = {}
    cone_colours["LWS"] = "#fe7c7c"
    cone_colours["MWS"] = "#8afe7c"
    cone_colours["SWS2"] = "#7c86fe"
    cone_colours["SWS1"] = "#fe7cfe"

    rod_peaks = {}
    rod_peaks["Chicken"] = np.array([500])
    rod_peaks["Zebrafish"] = np.array([500])
    rod_peaks["Human"] = np.array([498])
    rod_peaks["Mouse"] = np.array([500])
    rod_peaks["Frog"] = np.array([489])

    def govardovskii_animal(self, animal_name):
        return govardovskii(
            self.opsins_peaks[animal_name], self.cone_numbers[animal_name]
        )

    def plot_overview(self, animal_list):
        fig = make_subplots(rows=1, cols=len(animal_list), subplot_titles=animal_list)
        col = 1
        for animal in animal_list:
            opsin = self.govardovskii_animal(animal_name=animal)

            for cone in range(self.cone_numbers[animal]):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(300, 700, 1),
                        y=opsin[:, cone],
                        mode="lines",
                        line=dict(
                            color=self.cone_colours[self.opsins_types[animal][cone]]
                        ),
                        name=self.opsins_types[animal][cone],
                        fill="tozeroy",
                    ),
                    row=1,
                    col=col,
                )
            opsin = govardovskii(self.rod_peaks[animal], 1)

            fig.add_trace(
                go.Scatter(
                    x=np.arange(300, 700, 1),
                    y=opsin.flatten(),
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    name="Rod",
                ),
                row=1,
                col=col,
            )
            col = col + 1
        fig.update_layout(template="scatter_template")
        fig.update_yaxes(title="Spike count")
        fig.update_xaxes(title="Wavelength")
        return fig


def govardovskii(peakwvss, conenum):
    Ops = np.zeros((1025, conenum))
    for gg in range(0, conenum):
        awvs = 0.8795 + 0.0459 * np.exp(-1 * (peakwvss[gg] - 300) ** 2 / 11940)
        constA = np.asarray([awvs, 0.922, 1.104, 69.7, 28, -14.9, 0.674, peakwvss[gg]])
        Lamb = 189 + 0.315 * constA[7]  # Beta peak
        b = -40.5 + 0.195 * constA[7]  # Beta bandwidth
        Ab = 0.26  # Beta value at peak
        constB = np.asarray([Ab, Lamb, b])
        awvs = 0.8795 + 0.0459 * np.exp(-1 * (constA[7] - 300) ** 2 / 11940)
        constAB = np.concatenate([constA[:], constB[:]])

        lalax = np.asarray(list(range(300, 700, 1)))
        lalay = temple(lalax, constAB)  # Govardovskii guess

        Ops[300:700:1, gg] = lalay
    return Ops[300:700, :]


def temple(x, constAB):
    # constAB = a,b,c,A,B,C,D,lamA,Ab,Lamb,bB
    constA = constAB[0:8:1]
    constB = constAB[8:11:1]
    S = alphaband(x, constA) + betaband(x, constB)
    return S


def alphaband(x, constA):
    # a,b,c,A,B,C,D,lamA = constA
    x = constA[7] / x
    ##A = b/(a+b)*numpy.exp(a/n)
    ##B = a/(a+b)*numpy.exp(-1*b/n)
    alpha = 1 / (
        np.exp(constA[3] * (constA[0] - x))
        + np.exp(constA[4] * (constA[1] - x))
        + np.exp(constA[5] * (constA[2] - x))
        + constA[6]
    )
    return alpha


def betaband(x, constB):
    beta = constB[0] * np.exp(-1 * ((x - constB[1]) / constB[2]) ** 2)
    return beta
