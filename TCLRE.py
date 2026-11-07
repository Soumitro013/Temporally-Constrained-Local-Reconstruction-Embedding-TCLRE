import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness


class TCLRE:
    """
    Temporally Constrained Local Reconstruction Embedding (TCLRE).

    Designed for sequential, nonlinear sensor data such as
    wind turbine SCADA measurements.
    """

    def __init__(
        self,
        n_neighbors=6,
        temporal_window=10,
        temporal_strength=0.5,
        regime_penalty=5.0,
        transition_weight=0.1,
        max_dim=2,
        min_trustworthiness=0.90,
        min_temporal_corr=0.75,
        reg=1e-3
    ):
        self.n_neighbors = n_neighbors
        self.temporal_window = temporal_window
        self.temporal_strength = temporal_strength
        self.regime_penalty = regime_penalty
        self.transition_weight = transition_weight
        self.max_dim = max_dim
        self.min_trustworthiness = min_trustworthiness
        self.min_temporal_corr = min_temporal_corr
        self.reg = reg

    def _robust_scale(self, X):
        """Scale features using median and MAD."""
        self.median_ = np.median(X, axis=0)

        mad = np.median(
            np.abs(X - self.median_),
            axis=0
        )

        scale = 1.4826 * mad
        fallback = np.std(X, axis=0)

        scale = np.where(
            scale > 1e-9,
            scale,
            np.where(fallback > 1e-9, fallback, 1.0)
        )

        self.scale_ = scale

        return (X - self.median_) / self.scale_

    def _local_weights(self, target, neighbors):
        """
        Calculate regularised local reconstruction weights
        subject to their sum being one.
        """
        differences = neighbors - target

        C = differences @ differences.T

        ridge = self.reg * max(
            np.trace(C), 1e-12
        )

        C += ridge * np.eye(len(neighbors))

        try:
            weights = np.linalg.solve(
                C, np.ones(len(neighbors))
            )
        except np.linalg.LinAlgError:
            weights = np.ones(len(neighbors))

        total = np.sum(weights)

        if abs(total) < 1e-12:
            distances = np.linalg.norm(
                differences, axis=1
            )
            weights = 1.0 / (distances + 1e-9)
            total = np.sum(weights)

        return weights / total

    def _temporal_score(self, Z, Y):
        """Measure preservation of consecutive-distance ordering."""
        original_steps = np.linalg.norm(
            np.diff(Z, axis=0), axis=1
        )

        embedded_steps = np.linalg.norm(
            np.diff(Y, axis=0), axis=1
        )

        if (
            np.std(original_steps) < 1e-12
            or np.std(embedded_steps) < 1e-12
        ):
            return 0.0

        correlation = spearmanr(
            original_steps, embedded_steps
        )[0]

        return (
            float(correlation)
            if np.isfinite(correlation)
            else 0.0
        )

    def fit_transform(self, X, regimes=None):
        """
        Fit TCLRE on chronologically ordered observations
        and return their low-dimensional representation.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n, p = X.shape

        if n < 6 or p < 2:
            raise ValueError(
                "At least 6 observations and 2 features are required."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains missing or infinite values. "
                "Clean or impute them before fitting."
            )

        if regimes is not None:
            regimes = np.asarray(regimes)

            if len(regimes) != n:
                raise ValueError(
                    "The number of regime labels must match X."
                )

            self.regimes_ = regimes.copy()
        else:
            self.regimes_ = None

        # Step 1: Robust normalisation
        self.Z_ = self._robust_scale(X)

        # Step 2 and 3: Construct local reconstruction matrix
        W = np.zeros((n, n))

        for i in range(n):

            indices = np.arange(n)

            # Restrict candidates to a local time window
            candidates = indices[
                np.abs(indices - i) <= self.temporal_window
            ]

            candidates = candidates[candidates != i]

            if len(candidates) == 0:
                candidates = indices[indices != i]

            distances = np.sum(
                (self.Z_[candidates] - self.Z_[i]) ** 2,
                axis=1
            )

            # Penalise neighbours belonging to another regime
            if self.regimes_ is not None:
                different_regime = (
                    self.regimes_[candidates] != self.regimes_[i]
                )

                distances += (
                    self.regime_penalty * different_regime
                )

            k = min(self.n_neighbors, len(candidates))

            nearest = candidates[
                np.argsort(distances)[:k]
            ]

            weights = self._local_weights(
                self.Z_[i],
                self.Z_[nearest]
            )

            W[i, nearest] = weights

        # Step 4: Build the temporal graph Laplacian
        steps = np.linalg.norm(
            np.diff(self.Z_, axis=0),
            axis=1
        )

        positive_steps = steps[steps > 1e-12]

        sigma = (
            np.median(positive_steps)
            if len(positive_steps) > 0
            else 1.0
        )

        sigma = max(sigma, 1e-12)

        q = np.exp(
            -(steps ** 2) / (2 * sigma ** 2)
        )

        if self.regimes_ is not None:
            transitions = (
                self.regimes_[:-1] != self.regimes_[1:]
            )

            q[transitions] *= self.transition_weight

        L = np.zeros((n, n))

        for i, weight in enumerate(q):
            L[i, i] += weight
            L[i + 1, i + 1] += weight
            L[i, i + 1] -= weight
            L[i + 1, i] -= weight

        # Step 5: Construct the optimisation matrix
        I = np.eye(n)

        M = (
            (I - W).T @ (I - W)
            + self.temporal_strength * L
        )

        M = (M + M.T) / 2.0

        # Step 6: Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Exclude the constant eigenvector
        constant_idx = np.argmax(
            np.abs(eigenvectors.T @ np.ones(n))
        )

        order = [
            j for j in np.argsort(eigenvalues)
            if j != constant_idx
        ]

        max_d = min(
            self.max_dim,
            p - 1,
            n - 2
        )

        if max_d < 1:
            raise ValueError(
                "Unable to obtain a lower-dimensional representation."
            )

        Y_all = eigenvectors[:, order[:max_d]]

        # Step 7: Select the output dimension
        k_eval = max(
            1,
            min(self.n_neighbors, (n - 1) // 2)
        )

        self.dimension_scores_ = {}
        suitable_dimensions = []

        for d in range(1, max_d + 1):

            Y_d = Y_all[:, :d]

            trust = trustworthiness(
                self.Z_,
                Y_d,
                n_neighbors=k_eval
            )

            temporal = self._temporal_score(
                self.Z_, Y_d
            )

            score = (
                0.65 * trust
                + 0.35 * max(0.0, temporal)
            )

            self.dimension_scores_[d] = {
                "trustworthiness": float(trust),
                "temporal_correlation": float(temporal),
                "combined_score": float(score)
            }

            if (
                trust >= self.min_trustworthiness
                and temporal >= self.min_temporal_corr
            ):
                suitable_dimensions.append(d)

        if suitable_dimensions:
            best_d = min(suitable_dimensions)
        else:
            best_d = max(
                self.dimension_scores_,
                key=lambda d:
                    self.dimension_scores_[d]["combined_score"]
            )

        self.n_components_ = best_d
        self.embedding_ = Y_all[:, :best_d].copy()

        return self.embedding_

    def transform_one(self, x, regime=None):
        """
        Embed one new incoming observation using
        previously learned historical observations.
        """
        if not hasattr(self, "embedding_"):
            raise RuntimeError(
                "Call fit_transform() before transform_one()."
            )

        x = np.asarray(x, dtype=float).reshape(-1)

        if x.size != self.Z_.shape[1]:
            raise ValueError(
                "The new observation has an incorrect number of features."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError(
                "The new observation contains missing or infinite values."
            )

        # Apply the training-data scaling parameters
        z = (x - self.median_) / self.scale_

        n = len(self.Z_)

        # Use the most recent historical observations
        if self.temporal_window is None:
            candidates = np.arange(n)
        else:
            start = max(0, n - self.temporal_window)
            candidates = np.arange(start, n)

        distances = np.sum(
            (self.Z_[candidates] - z) ** 2,
            axis=1
        )

        if self.regimes_ is not None and regime is not None:
            different_regime = (
                self.regimes_[candidates] != regime
            )

            distances += (
                self.regime_penalty * different_regime
            )

        k = min(self.n_neighbors, len(candidates))

        nearest = candidates[
            np.argsort(distances)[:k]
        ]

        weights = self._local_weights(
            z,
            self.Z_[nearest]
        )

        # Out-of-sample barycentric embedding
        return weights @ self.embedding_[nearest]

    def transform(self, X, regimes=None):
        """Embed multiple new observations."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.Z_.shape[1]:
            raise ValueError(
                "The new data has an incorrect number of features."
            )

        if regimes is not None:
            regimes = np.asarray(regimes)

            if len(regimes) != len(X):
                raise ValueError(
                    "The number of regime labels must match X."
                )

            return np.vstack([
                self.transform_one(x, regime=r)
                for x, r in zip(X, regimes)
            ])

        return np.vstack([
            self.transform_one(x)
            for x in X
        ])
