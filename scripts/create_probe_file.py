def create_channel_map(spacing: int) -> dict[int, list[int]]:
    """
    Return dict {channel_index: [x, y]} for a 16x16 spaced grid excluding 4 corners.
    spacing: distance between adjacent positions.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    max_coord = 15 * spacing
    mapping = {}
    idx = 0
    for col in range(16):
        x = col * spacing
        if col in (0, 15):
            ys = range(max_coord - spacing, spacing - 1, -spacing)
        else:
            ys = range(max_coord, -spacing, -spacing)
        for y in ys:
            mapping[idx] = [x, y]
            idx += 1
    return mapping


# %%
if __name__ == "__main__":
    mapping = create_channel_map(30)
    print(str(mapping).replace("],", "],\n"))
