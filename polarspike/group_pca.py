"""
GroupPCA is a class that performs PCA on multiple views of data and then concatenates the transformed data to perform PCA
on the concatenated data. This can be useful for analyzing multi-view data where each view may have different dimensions
and the views may be correlated.
@ Marvin Seifert 2024
"""


from sklearn.decomposition import PCA
import numpy as np


class GroupPCA:
    """
    GroupPCA is a class that performs PCA on multiple views of data and then concatenates the transformed data to perform PCA
    on the concatenated data. This can be useful for analyzing multi-view data where each view may have different dimensions
    and the views may be correlated.

    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pcas = []
        self.final_pca = PCA(n_components=n_components)

    def fit(self, Xs: list[np.ndarray]):
        """
        Fit the GroupPCA model to the data.
        Parameters
        ----------
        Xs : list of numpy arrays
            List of views of the data where each view is a numpy array.
        """
        # Step 1: Perform PCA on each view
        transformed_views = []
        for X in Xs:
            pca = PCA(n_components=self.n_components)
            transformed = pca.fit_transform(X)
            self.pcas.append(pca)
            transformed_views.append(transformed)

        # Step 2: Concatenate the transformed data
        concatenated = np.hstack(transformed_views)

        # Step 3: Perform PCA on the concatenated data
        self.final_pca.fit(concatenated)

    def transform(self, Xs: list[np.ndarray]) -> np.ndarray:
        """
        Transform the data using the fitted GroupPCA model.
        Parameters
        ----------
        Xs : list of numpy arrays
            List of views of the data where each view is a numpy array.
        """
        # Step 1: Transform each view using the fitted PCAs
        transformed_views = [pca.transform(X) for pca, X in zip(self.pcas, Xs)]

        # Step 2: Concatenate the transformed data
        concatenated = np.hstack(transformed_views)

        # Step 3: Transform the concatenated data using the final PCA
        return self.final_pca.transform(concatenated)

    def fit_transform(self, Xs) -> np.ndarray:
        """
        Fit the GroupPCA model to the data and transform the data. This is equivalent to calling fit() and then transform().
        Parameters
        ----------
        Xs : list of numpy arrays
            List of views of the data where each view is a numpy array.


        """

        self.fit(Xs)
        return self.transform(Xs)
