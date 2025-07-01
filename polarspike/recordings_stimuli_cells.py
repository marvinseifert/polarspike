def sort(input, all_recordings):
    """
    Sorts the input into a list of recordings, stimuli and cells. Checks if the input is valid.

    Parameters
    ----------
    input : list
        A list containing the recordings, stimuli and cells.
    all_recordings : list of lists
        A list of lists containing all recordings.

    Returns
    -------
    input : list
        A list containing the recordings, stimuli and cells in the correct format.
    """
    nr_recordings = len(input[0])

    if nr_recordings == 1:
        if input[0][0][0] == "all":
            input[0] = all_recordings
            nr_recordings = len(input[0])
        else:
            assert (
                nr_recordings == len(input[1]) or len(input[1]) == 1
            ), "Nr of stimuli must match number of recordings or be equal to 1."

    nr_rec_stimuli = len(input[1])
    if (nr_rec_stimuli == 1) & (nr_recordings != 1):
        input[1] = input[1] * nr_recordings
        nr_rec_stimuli = len(input[1])

    nr_rec_cells = len(input[2])

    if (nr_rec_cells == 1) & nr_recordings > 1:
        input[2][0] = input[2][0] * nr_recordings
    else:
        assert (nr_rec_cells == nr_recordings) or (
            nr_rec_cells == 1
        ), "Lists of cells must have equal length to lists of recordings or be len == 1."

    if len(input[2]) != nr_recordings:
        assert (
            len(input[2]) == 1
        ), "Length of cell lists must match length of recordings and stimuli or be 1."
        input[2] = input[2] * nr_recordings

    for idx, stimuli in zip(range(nr_recordings * nr_rec_stimuli), input[1]):
        if len(stimuli) != len(input[2][idx]):
            input[2][idx] = input[2][idx] * len(stimuli)

    return input
