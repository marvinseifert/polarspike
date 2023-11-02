import numpy as np
import polars as pl
import math


def compute_vector_magnitude_direction(observations):
    """
    Compute the magnitude and direction of the resulting vector based on observations.

    Parameters:
    - observations (list): List of observations for each direction.

    Returns:
    - tuple: (magnitude, direction) of the resulting vector.
    """
    # Calculate the x and y components for each direction
    x_components = [
        obs * math.cos(math.radians(angle))
        for angle, obs in zip(range(0, 360, 45), observations)
    ]
    y_components = [
        obs * math.sin(math.radians(angle))
        for angle, obs in zip(range(0, 360, 45), observations)
    ]

    # Sum up the x and y components to get the total x and y components of the resulting vector
    x_total = sum(x_components)
    y_total = sum(y_components)

    # Calculate the magnitude and direction of the resulting vector
    magnitude = math.sqrt(x_total**2 + y_total**2)
    direction = math.degrees(math.atan2(y_total, x_total))

    return magnitude, direction


def cal_dir_vector_on_df(df):
    directions = np.array([0, 180, 270, 90, 315, 225, 45, 135])
    directions = np.repeat(directions, 400)
    trigger = df["trigger"].to_numpy()
    directions_spikes = directions[trigger]
    direction_counts = np.unique(directions_spikes, return_counts=True)
    magnitude, direction = compute_vector_magnitude_direction(direction_counts[1])
    magnitude = magnitude / np.sum(direction_counts[1])
    cell_index = df["cell_index"][0]
    df_out = pl.DataFrame(
        data={
            "cell_index": [cell_index],
            "magnitude": [magnitude],
            "direction": [direction],
            "sum_spikes": [np.sum(direction_counts[1])],
        }
    )
    return df_out


def cal_ori_vector_on_df(df):
    # directions = np.array([0, 180, 270, 90, 315, 225, 45, 135])
    mapped_directions = np.array([0, 0, 180, 180, 90, 270, 270, 90])
    mapped_directions = np.repeat(mapped_directions, 400)
    trigger = df["trigger"].to_numpy()
    directions_spikes = mapped_directions[trigger]
    direction_counts = np.unique(directions_spikes, return_counts=True)
    magnitude, direction = compute_vector_magnitude_direction(direction_counts[1])
    magnitude = magnitude / np.sum(direction_counts[1])
    cell_index = df["cell_index"][0]
    df_out = pl.DataFrame(
        data={
            "cell_index": [cell_index],
            "magnitude": [magnitude],
            "direction": [direction],
            "sum_spikes": [np.sum(direction_counts[1])],
        }
    )
    return df_out
