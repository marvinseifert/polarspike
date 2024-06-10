from sklearn.decomposition import PCA
import numpy as np


class GroupPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pcas = []
        self.final_pca = PCA(n_components=n_components)

    def fit(self, Xs):
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

    def transform(self, Xs):
        # Step 1: Transform each view using the fitted PCAs
        transformed_views = [pca.transform(X) for pca, X in zip(self.pcas, Xs)]

        # Step 2: Concatenate the transformed data
        concatenated = np.hstack(transformed_views)

        # Step 3: Transform the concatenated data using the final PCA
        return self.final_pca.transform(concatenated)

    def fit_transform(self, Xs):
        self.fit(Xs)
        return self.transform(Xs)
