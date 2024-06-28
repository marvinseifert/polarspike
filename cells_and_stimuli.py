def sort(stimuli, cells, all_cells):
    """
    Sorts the stimuli and cells list. If cells is of length 1 and contains the string "all", it will be replaced by all_cells.

    Parameters
    ----------
    stimuli : list
        List of lists of stimuli.
    cells : list
        List of lists of cells.
    all_cells : np.ndarray
        Array containing the integer indices of all cells.

    Returns
    -------
    stimuli : list
        List of lists of stimuli.
    cells : list
        List of lists of cells.
    """
    nr_stimuli = len(stimuli)
    if len(cells) != len(stimuli):
        if len(cells) == 1:
            if type(cells[0]) is str and cells[0] == "all":
                cells = all_cells
            cells = cells * nr_stimuli
        else:
            raise ValueError(
                "Length of stimulus list and cell list must be equal if cell list is not of length 1"
            )

    for idx, entry in enumerate(cells):
        if entry[0] == "all":
            cells[idx] = all_cells

    return stimuli, cells
