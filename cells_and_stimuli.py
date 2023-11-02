import numpy as np


def sort(stimuli, cells, nr_cells):
    assert len(stimuli) == len(cells), "Length of stimulus list and cell list must be equal"
    nr_stimuli = len(stimuli)
    if len(cells) != len(stimuli):
        if len(cells) == 1:
            if cells[0] == "all":
                cells = np.arange(nr_cells)
            cells = cells * nr_stimuli
        else:
            raise ValueError("Length of stimulus list and cell list must be equal if cell list is not of length 1")
    else:
        for idx, entry in enumerate(cells):
            if entry[0] == "all":
                cells[idx] = np.arange(nr_cells)

    return stimuli, cells
