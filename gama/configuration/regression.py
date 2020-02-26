import numpy as np

from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    Binarizer,
)
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import (
    SelectFwe,
    SelectPercentile,
    VarianceThreshold,
    f_regression,
)


from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

# For comparison, this selection of operators and hyperparameters is
# currently most of what TPOT supports.

reg_config = {
    ElasticNetCV: {
        "l1_ratio": np.arange(0.0, 1.01, 0.05),
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    ExtraTreesRegressor: {
        "n_estimators": [100],
        "max_features": np.arange(0.05, 1.01, 0.05),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "bootstrap": [True, False],
    },
    GradientBoostingRegressor: {
        "n_estimators": [100],
        "loss": ["ls", "lad", "huber", "quantile"],
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "subsample": np.arange(0.05, 1.01, 0.05),
        "max_features": np.arange(0.05, 1.01, 0.05),
        "alpha": [0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
    },
    AdaBoostRegressor: {
        "n_estimators": [100],
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "loss": ["linear", "square", "exponential"],
        # 'max_depth': range(1, 11) not available in sklearn==0.19.1
    },
    DecisionTreeRegressor: {
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    },
    KNeighborsRegressor: {
        "n_neighbors": range(1, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    LassoLarsCV: {"normalize": [True, False]},
    LinearSVR: {
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "dual": [True, False],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    },
    RandomForestRegressor: {
        "n_estimators": [100],
        "max_features": np.arange(0.05, 1.01, 0.05),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "bootstrap": [True, False],
    },
    # Preprocesssors
    Binarizer: {"threshold": np.arange(0.0, 1.01, 0.05)},
    FastICA: {"tol": np.arange(0.0, 1.01, 0.05)},
    FeatureAgglomeration: {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
        "param_check": [
            lambda params: (not params["linkage"] == "ward")
            or params["affinity"] == "euclidean"
        ],
    },
    MaxAbsScaler: {},
    MinMaxScaler: {},
    Normalizer: {"norm": ["l1", "l2", "max"]},
    Nystroem: {
        "kernel": [
            "rbf",
            "cosine",
            "chi2",
            "laplacian",
            "polynomial",
            "poly",
            "linear",
            "additive_chi2",
            "sigmoid",
        ],
        "gamma": np.arange(0.0, 1.01, 0.05),
        "n_components": range(1, 11),
    },
    PCA: {"svd_solver": ["randomized"], "iterated_power": range(1, 11)},
    PolynomialFeatures: {
        "degree": [2],
        "include_bias": [False],
        "interaction_only": [False],
    },
    RBFSampler: {"gamma": np.arange(0.0, 1.01, 0.05)},
    RobustScaler: {},
    StandardScaler: {},
    # Selectors
    SelectFwe: {"alpha": np.arange(0, 0.05, 0.001), "score_func": {f_regression: None}},
    SelectPercentile: {"percentile": range(1, 100), "score_func": {f_regression: None}},
    VarianceThreshold: {"threshold": np.arange(0.05, 1.01, 0.05)},
}
