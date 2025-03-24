import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from polarspike import colour_template
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# %%
def extract_spike_data(df, reps=None):
    if reps is None:
        reps = np.arange(0, 20, 2)
    y = df.select([f"first_spike_{rep}" for rep in reps]).to_numpy().flatten()
    x = np.repeat(np.arange(0, reps.shape[0] * 10, 10), len(df))
    weights = (
        df.select([f"first_spike_posterior_{rep}" for rep in reps]).to_numpy().flatten()
    )
    nans = np.isnan(y)
    y = y[~nans]
    x = x[~nans]
    weights = weights[~nans]
    return x, y, weights


# %%
CT = colour_template.Colour_template()
CT.list_stimuli()
CT_contrast = colour_template.Colour_template()
CT_contrast.pick_stimulus("FFF_6_MC")
colours = CT_contrast.colours[[4, 2, 0, 1]]
# %%
contrasts = {
    "535": pl.scan_parquet(
        r"D:\chicken_analysis\changepoint_df_csteps535.parquet"
    ).collect(),
    "560": pl.scan_parquet(
        r"D:\chicken_analysis\changepoint_df_csteps560.parquet"
    ).collect(),
    "610": pl.scan_parquet(
        r"D:\chicken_analysis\changepoint_df_csteps610.parquet"
    ).collect(),
    "white": pl.scan_parquet(
        r"D:\chicken_analysis\changepoint_df_csteps.parquet"
    ).collect(),
}
# %%
contrasts_flat = {}
for key in contrasts.keys():
    (
        contrasts_flat[f"{str(key)}_x"],
        contrasts_flat[f"{str(key)}_y"],
        contrasts_flat[f"{str(key)}_weights"],
    ) = extract_spike_data(contrasts[key], reps=np.arange(1, 20, 2))
# %% Linear regression
contrast_models = {}
for key in contrasts.keys():
    model = RANSACRegressor()
    model.fit(
        contrasts_flat[f"{str(key)}_x"].reshape(-1, 1),
        contrasts_flat[f"{str(key)}_y"],
        sample_weight=contrasts_flat[f"{str(key)}_weights"],
    )
    contrast_models[key] = model
# %% Plot only the regression lines for all models
fig, ax = plt.subplots()
for c_idx, key in enumerate(contrast_models.keys()):
    x = np.linspace(0, 100, 1000)
    y = contrast_models[key].predict(x.reshape(-1, 1))
    ax.plot(x, y, label=key, c=colours[c_idx])
fig.show()
# %% sanity plot of white values
fig, ax = plt.subplots(nrows=len(contrasts.keys()), ncols=1, figsize=(7, 15))
for idx, key in enumerate(contrasts.keys()):
    ax[idx].scatter(
        contrasts_flat[f"{str(key)}_x"],
        contrasts_flat[f"{str(key)}_y"],
        c="black",
        alpha=0.01,
        s=contrasts_flat[f"{str(key)}_weights"] * 10,
    )
    x = np.linspace(0, 100, 1000)
    y = contrast_models[key].predict(x.reshape(-1, 1))
    ax[idx].plot(x, y, label=key, c=colours[idx])

fig.show()
# print all slopes
for key in contrast_models.keys():
    print(key, contrast_models[key].estimator_.coef_)
# %% Apply regression per cell
reps = np.arange(0, 20, 2)
contrast_levels = np.arange(1, 101, 10)
contrast_cells = {}
for key in contrasts.keys():
    contrast_cells[key] = (
        contrasts[key]
        .lazy()
        .group_by("recording", "cell_index")
        .agg(
            pl.concat_list([f"first_spike_{rep}" for rep in reps]).alias("y"),
            pl.concat_list([f"first_spike_posterior_{rep}" for rep in reps]).alias(
                "weights"
            ),
        )
        .explode("y", "weights")
        .explode("y", "weights")
        .with_columns(
            x=np.tile(contrast_levels, len(contrasts[key])),
        )
        .collect()
    )


# %%
def reg_apply_func(df):
    df = df.drop_nulls()
    if len(df) > 2 and np.sum(df["weights"].to_numpy()) > 0:
        results = classify_relationship(df["x"], df["y"], weights=df["weights"])

        df = pl.DataFrame(
            {
                "recording": df.head(1)["recording"],
                "cell_index": df.head(1)["cell_index"],
                "slope": results["slope"],
                "score": results["score"],
                "intercept": results["intercept"],
                "model_type": results["relationship"],
            },
            schema={
                "recording": pl.String,
                "cell_index": pl.Int64,
                "slope": pl.Float64,
                "intercept": pl.Float64,
                "score": pl.Float64,
                "model_type": pl.String,
            },
        )
    else:
        df = pl.DataFrame(
            {
                "recording": df.head(1)["recording"],
                "cell_index": df.head(1)["cell_index"],
                "slope": np.nan,
                "intercept": np.nan,
                "score": np.nan,
                "model_type": "insufficient_data",
            },
            schema={
                "recording": pl.String,
                "cell_index": pl.Int64,
                "slope": pl.Float64,
                "intercept": pl.Float64,
                "score": pl.Float64,
                "model_type": pl.String,
            },
        )
    return df


def classify_relationship(x, y, weights=None):
    # Convert to numpy arrays and reshape for scikit-learn
    x = np.array(x)
    y = np.array(y)
    X = x.reshape(-1, 1)

    # Fit linear model: y ~ x
    linear_model = RANSACRegressor().fit(X, y, sample_weight=weights)
    linear_score = linear_model.score(X, y)

    # Check for positivity before log transformation
    if np.any(x <= 0):
        raise ValueError(
            "All x values must be positive for a logarithmic transformation."
        )

    # Fit log model: y ~ log(x)
    # We build a small pipeline to perform the log transformation before regression.
    log_pipeline = Pipeline(
        [
            ("log_transform", FunctionTransformer(np.log, validate=True)),
            ("linear_regression", RANSACRegressor()),
        ]
    )
    log_pipeline.fit(X, y, linear_regression__sample_weight=weights)
    log_score = log_pipeline.score(X, y)

    # compare to mean fit
    if np.abs(linear_score) < 0.1 and np.abs(log_score) < 0.1:
        # calculate r2 between mean and y
        mean = np.nanmean(y)
        y_pred = np.full_like(y, fill_value=mean)
        ss_tot = np.sum((y - mean) ** 2)
        ss_res = np.sum(
            (y - y_pred) ** 2
        )  # This should ideally be the same as ss_tot for a perfect constant model

        score = 1 - ss_res / ss_tot
        if score == 0:
            score = 1
        if score > 0.5:
            return {
                "relationship": "steady",
                "model": None,
                "score": score,
                "slope": 0,
                "intercept": mean,
            }
        else:
            return {
                "relationship": "no fit",
                "model": None,
                "score": score,
                "slope": 0,
                "intercept": 0,
            }
    # Decide which model fits better
    if linear_score > log_score:
        return {
            "relationship": "linear",
            "model": linear_model,
            "score": linear_score,
            "slope": linear_model.estimator_.coef_[0],
            "intercept": linear_model.estimator_.intercept_,
        }
    else:
        return {
            "relationship": "logarithmic",
            "model": log_pipeline,
            "score": log_score,
            "slope": log_pipeline.named_steps["linear_regression"].estimator_.coef_[0],
            "intercept": log_pipeline.named_steps[
                "linear_regression"
            ].estimator_.intercept_,
        }


# %%
contrast_cells_results = {}
for key in contrast_cells.keys():
    contrast_cells_results[key] = (
        contrast_cells[key]
        .group_by("recording", "cell_index")
        .map_groups(reg_apply_func)
    )
    contrast_cells_results[key] = (
        contrast_cells_results[key]
        .with_columns(pl.all().name.suffix("_" + key))
        .select(
            "recording",
            "cell_index",
            "slope_" + key,
            "intercept_" + key,
            "score_" + key,
            "model_type_" + key,
        )
    )

# %%
all_results = pl.concat(list(contrast_cells_results.values()), how="align")
all_results = all_results.fill_nan(None)
# %% plot intercepts
fig, ax = plt.subplots()
for c_idx, key in enumerate(contrast_cells_results.keys()):
    ax.scatter(
        [key] * len(all_results),
        all_results["score_" + key],
        c=colours[c_idx],
    )
    average = np.average(
        all_results.drop_nulls()["score_" + key].to_numpy(),
        # weights=all_results.drop_nulls()["intercept_" + key].to_numpy(),
    )
    print(key, average)
    ax.scatter(
        key,
        average,
    ),

fig.show()
# %% histogram of model types
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 10), sharey=True)
for idx, key in enumerate(contrast_cells_results.keys()):
    names, counts = np.unique(
        all_results["model_type_" + key].drop_nulls().to_numpy(),
        return_counts=True,
    )
    counts = counts / len(all_results["model_type_" + key].drop_nulls())
    # check how many values are close to 0
    ax[idx].bar(names, counts, color=colours[idx])
ax[0].set_ylabel("Proportion")
ax[0].set_ylim([0, 1])
# despine(ax)
for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
fig.show()
