import numpy as np
import polars as pl
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


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
    magnitude = math.sqrt(x_total ** 2 + y_total ** 2)
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


def add_arrows_to_matplotlib(initial_fig, degrees, arrow_spacing, names=None):
    """
    Adds a subplot with arrows to an existing Matplotlib figure.

    Parameters:
        - initial_fig: The initial Matplotlib figure to which the subplot will be added.
        - degrees: A list of degrees for the arrows.
        - arrow_spacing: A list of durations in seconds indicating the x-position for each arrow.
        - names: (Optional) A list of names for the arrows.

    Returns:
        - The modified figure with the original plot and the added arrow subplot.
    """
    # Get the original axes
    original_axes = initial_fig.get_axes()
    n_original_subplots = len(original_axes)

    # Create a new GridSpec layout with an additional row for the arrows subplot
    gs = gridspec.GridSpec(n_original_subplots + 1, 1, figure=initial_fig,
                           height_ratios=[1] * n_original_subplots + [0.1])

    # Create the new subplot for the arrows below the original subplots
    ax_arrows = initial_fig.add_subplot(gs[-1, :], sharex=original_axes[0])
    ax_arrows.axis('off')  # Turn off the axis

    # The x_positions are the cumulative sum of the spacings, indicating where along the x-axis the arrows should be
    x_positions = np.cumsum(arrow_spacing) - np.asarray(
        arrow_spacing) / 2  # subtract half, to put arrow into the middle

    # Add arrows to the new subplot
    for x_pos, degree in zip(x_positions, degrees):
        # Calculate the arrow direction
        dx = np.cos(np.radians(degree))
        dy = np.sin(np.radians(degree))

        # Define tail and head of the arrow
        x_tail = x_pos
        y_tail = 0.5
        x_head = x_tail + dx * 0.1
        y_head = y_tail + dy * 0.1

        # Create and add the FancyArrowPatch to the subplot
        arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head), arrowstyle="Fancy",
                                         mutation_scale=20, color='k')
        ax_arrows.add_patch(arrow)

    # Add text if names are provided
    if names is not None and len(names) == len(degrees):
        for x_pos, name in zip(x_positions, names):
            ax_arrows.text(x_pos, 0.3, name, ha='center', va='center', fontsize=14)

    # Set the limits for the arrow subplot to match the x-limits of the original axes
    ax_arrows.set_xlim(original_axes[0].get_xlim())
    ax_arrows.set_ylim(0, 1)

    # Adjust the layout
    initial_fig.tight_layout()

    return initial_fig
