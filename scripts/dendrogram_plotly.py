import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage


# %% get data
feature_df = pd.read_pickle(r"A:\Marvin\fff_clustering\features_coeffs_pca")
data_array = feature_df.drop(
    columns=["pca_1", "pca_2", "labels", "tsne_1", "tsne_2"]
).values
labels = feature_df.index.values

# Initialize figure by creating upper dendrogram
fig = ff.create_dendrogram(
    data_array, orientation="bottom", labels=labels, color_threshold=50
)
for i in range(len(fig["data"])):
    fig["data"][i]["yaxis"] = "y2"

# Create Side Dendrogram
dendro_side = ff.create_dendrogram(data_array, orientation="right", color_threshold=50)
for i in range(len(dendro_side["data"])):
    dendro_side["data"][i]["xaxis"] = "x2"

# Add Side Dendrogram Data to Figure
for data in dendro_side["data"]:
    fig.add_trace(data)

# Create Heatmap
dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
dendro_leaves = list(map(int, dendro_leaves))
data_dist = pdist(data_array)
heat_data = squareform(data_dist)
heat_data = heat_data[dendro_leaves, :]
heat_data = heat_data[:, dendro_leaves]

heatmap = [go.Heatmap(x=dendro_leaves, y=dendro_leaves, z=heat_data, colorscale="deep")]

heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

# Add Heatmap Data to Figure
for data in heatmap:
    fig.add_trace(data)

# Edit Layout
fig.update_layout(
    {
        "width": 800,
        "height": 800,
        "showlegend": False,
        "hovermode": "closest",
    }
)
# Edit xaxis
fig.update_layout(
    xaxis={
        "domain": [0.15, 1],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "ticks": "",
    }
)
# Edit xaxis2
fig.update_layout(
    xaxis2={
        "domain": [0, 0.15],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
        "ticks": "",
    }
)

# Edit yaxis
fig.update_layout(
    yaxis={
        "domain": [0, 0.85],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
        "ticks": "",
    }
)
# Edit yaxis2
fig.update_layout(
    yaxis2={
        "domain": [0.825, 0.975],
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
    }
)

# Plot!
fig.update_layout(width=1000, height=1000)
fig.update_xaxes(showticklabels=False)


# %%
labels = fcluster(linkage(data_dist, method="complete"), criterion="distance", t=50)
print(np.max(labels) + 1)
max_label = np.max(labels) + 1
label_sums = []
previous = 0
for i in range(max_label):
    label_sums.append(previous + np.sum(labels == i))
    previous = label_sums[-1]
