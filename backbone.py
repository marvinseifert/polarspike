"""
Backbone functions, which dont do the actual analysis but are used during the analysis by many other
functions in this project.
@ Marvin Seifert 2024
"""
# Imports
from ipywidgets import widgets
import traitlets
from sympy import isprime
from math import gcd
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
import os


def get_file_ending(any_file: str) -> str:
    """
    File ending function
    This function finds the ending of a file and returns the file suffix.

    Parameters
    ----------
    any_file : str:
        The file you want to get the file ending from
    Returns
    -------
    format: str: The file suffix including the dot, or 0 if no suffix is found.
    """
    path = Path(any_file)
    suffix = path.suffix

    if not suffix:
        print("Named file is not a file, (maybe folder?), return 0")
        return 0

    return suffix

class SelectFilesButton(widgets.Button):
    """
    File selection Button using PyQt5
    Opens a native OS file selection dialog and returns selected file paths.
    """

    out = widgets.Output()

    def __init__(self, text: str, button_width: str = "200px") -> None:
        super().__init__()
        self.add_traits(files=traitlets.List())
        self.description = "Select " + text
        self.icon = "square-o"
        self.style.button_color = "red"
        self.layout = widgets.Layout(width=button_width)
        self.on_click(self.select_files)
        self.loaded = False

        # Ensure QApplication exists
        if not QApplication.instance():
            self._app = QApplication(sys.argv)
        else:
            self._app = QApplication.instance()

    def select_files(self, b):
        """Open PyQt5 file dialog and store selected files."""
        with self.out:
            try:
                dialog = QFileDialog()
                dialog.setFileMode(QFileDialog.ExistingFiles)
                dialog.setWindowTitle("Select files")
                if dialog.exec_():
                    selected_files = dialog.selectedFiles()
                    self.files = list(selected_files)

                    if self.files:
                        self.description = "Files Selected"
                        self.icon = "check-square-o"
                        self.style.button_color = "lightgreen"
                        self.loaded = True
                    else:
                        self.loaded = False
                else:
                    self.loaded = False

            except Exception as e:
                self.loaded = False
                print(f"Error: {e}")



# def bisection(array, value):
#     """
#     Bisection function
#     Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
#         and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
#         to indicate that ``value`` is out of range below and above respectively.
#
#     Parameters
#     ----------
#     array: np.array: The array containing a sequence of increasing values.
#
#     value: int, float: The value that the array is compared against
#
#     Returns
#     -------
#     position : The position of the nearest array value to value or zero if the
#     value is out of range
#     """
#     n = len(array)
#     if value < array[0]:
#         return -1
#     elif value > array[n - 1]:
#         return -1
#     jl = 0  # Initialize lower
#     ju = n - 1  # and upper limits.
#     while ju - jl > 1:  # If we are not yet done,
#         jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
#         if value >= array[jm]:
#             jl = jm  # and replace either the lower limit
#         else:
#             ju = jm  # or the upper limit, as appropriate.
#         # Repeat until the test condition is satisfied.
#     if value == array[0]:  # edge cases at bottom
#         return 0
#     elif value == array[n - 1]:  # and top
#         return n - 1
#     else:
#         return jl + 1


# def select_stimulus(max):
#     """
#     Creates a Jupyter widget that allows to select an integer between zeros and zero
#     a maximum
#
#     Parameters
#     ----------
#     max: integer: the maximum to which a number can be selected
#
#     Returns
#     -------
#     A widget object
#     """
#     Select_stimulus_w = widgets.BoundedIntText(
#         value=0,
#         min=0,
#         max=max,
#         step=1,
#         description="Which stimulus do you want to analyse?",
#         disabled=False,
#     )
#     return Select_stimulus_w


# def parallelize_dataframe(df, func, n_cores=4):
#     """
#     Function to split a dataframe into n parts and run the same function on all the
#     parts of the dataframe on different cores in parallel.
#
#     Parameters
#     ----------
#     df: pandas DataFrame: The dataframe
#     func: function: the function that shall be maped onto the dataframe
#     n_cores: int: default: 4: The number of cores which will be assigned with a
#     part of the dataframe
#
#     Returns
#     -------
#     df: pandas DataFrame: The recombined dataframe containing the results
#
#     """
#
#     df_split = np.array_split(df, n_cores)
#     pool = Pool(n_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df


def nr_rowcol_subplt(nr_plots: int) -> tuple:
    """Returns an optimal arangement of subplot rows and columns, depending on
    the nr of total plots. Example: If nr_plots = 20, returns col=4, row=5

    Parameters
    ----------
    nr_plots : int
        Nr of total plots that shall be plotted.

    Returns
    -------
    out/p = list indicating [0] =columns and [1] rows
    nr_plots = nr_of total plots (used for recursive function call)
        Description of returned object.

    """
    # This is a quick and dirty matlab translation....
    while isprime(nr_plots) & nr_plots > 4:
        nr_plots = nr_plots + 1

    p = factorization(nr_plots)
    p = sorted(p)
    if len(p) == 1:
        out = [1, p[0]]
        return out, nr_plots

    while len(p) > 2:
        if len(p) >= 4:
            p[0] = p[0] * p[-2]
            p[1] = p[1] * p[-1]
            p.pop()
            p.pop()
        else:
            p[0] = p[0] * p[1]
            p.pop(1)
        p = sorted(p)
        # print(p)

    while p[1] / p[0] > 2.5:
        N = nr_plots + 1
        p, nr_plots = nr_rowcol_subplt(N)
    return p, nr_plots


def factorization(n: int) -> list:
    """Returns factorization of the input integer

    Parameters
    ----------
    n : Int
        integer that shall be factorized

    Returns
    -------
    factors: :List
        List of factors

    """

    factors = []

    def get_factor(n):
        """Returns the factor of the input integer
        Parameters
        ----------
        n : int
            Integer that shall be factorized
        """
        x_fixed = 2
        cycle_size = 2
        x = 2
        factor = 1

        while factor == 1:
            for count in range(cycle_size):
                if factor > 1:
                    break
                x = (x * x + 1) % n
                factor = gcd(x - x_fixed, n)

            cycle_size *= 2
            x_fixed = x

        return factor

    while n > 1:
        next = get_factor(n)
        factors.append(next)
        n //= next

    return factors


# def multipop(yourlist, itemstopop):
#     """Short summary.
#
#     Parameters
#     ----------
#     yourlist : List
#         Old list from which items shall be removed
#     itemstopop : List
#         Indices you want to remove
#
#     Returns
#     -------
#     List
#         new list without the item that shall be removed
#
#     """
#     result = []
#     itemstopop.sort()
#     itemstopop = itemstopop[::-1]
#     for x in itemstopop:
#         result.append(yourlist.pop(x))
#     return result, yourlist


def hex_to_rgb(hex_color: str) -> tuple:
    """Translate a hex code colour to a rgb value

    Parameters
    ----------
    hex_color : str
        The hex colour code

    Returns
    -------
    tuple
        The resulting rgb value as tuple

    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def enumerate2(xs: list, start: int = 0, step: int = 1):
    """Enumerate function that allows to set a start and step value
    Parameters
    ----------
    xs : list
        The list that shall be enumerated
    start : int
        The starting value
    step : int
        The step value

    """
    for x in xs:
        yield start, x
        start += step
