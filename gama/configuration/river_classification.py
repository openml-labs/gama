import numpy as np

# classifiers
from river import neighbors

# from river.neighbors import KNNADWINClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.ensemble import LeveragingBaggingClassifier
from river.ensemble import ADWINBaggingClassifier
from river.ensemble import AdaBoostClassifier
from river.ensemble import AdaptiveRandomForestClassifier
from river import tree
from river import linear_model

# preprocessing
from river.preprocessing import (
    AdaptiveStandardScaler,
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

# feature extraction
from river.feature_extraction import PolynomialExtender

# feature selection
# from river.feature_selection import SelectKBest

clf_config_online = {
    # KNNADWINClassifier: {
    #     "n_neighbors": range(1, 15),
    #     "window_size": [100, 500, 750, 1000],
    #     "leaf_size": range(5, 50, 5),
    #     "p": np.arange(1, 2, 0.2)
    # },
    HoeffdingAdaptiveTreeClassifier: {
        "grace_period": range(50, 350),
        "split_criterion": ["info_gain", "gini", "hellinger"],
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "tie_threshold": np.arange(0.02, 0.08, 0.01),
        "leaf_prediction": ["mc", "nb", "nba"],
        "nb_threshold": range(0, 50, 10),
        # "splitter": ["tree.splitter.EBSTSplitter", "t
        # ree.splitter.HistogramSplitter",
        #             "tree.splitter.TEBSTSplitter",
        #             "tree.splitter.GaussianSplitter"],
        "bootstrap_sampling": [True, False],
        "drift_window_threshold": range(100, 500, 100),
        "adwin_confidence": [2e-4, 2e-3, 2e-2],
        # "max_size": [16],
        # "memory_estimate_period":[1000],
        # "stop_mem_management": [True],
        # "remove_poor_attrs": [True],
    },
    AdaptiveRandomForestClassifier: {
        "n_models": range(1, 20),
        "max_features": [0.2, 0.5, 0.7, 1.0, "sqrt", "log2", None],
        "lambda_value": range(2, 10),
        "grace_period": range(50, 350),
        "split_criterion": ["info_gain", "gini", "hellinger"],
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "tie_threshold": np.arange(0.02, 0.08, 0.01),
        "leaf_prediction": ["mc", "nb", "nba"],
        "nb_threshold": range(0, 50, 10),
        # "max_size": [16],
        # "stop_mem_management": [True],
        # "remove_poor_attrs": [True],
        # "memory_estimate_period": [1000]
    },
    LeveragingBaggingClassifier: {
        "model": [
            linear_model.LogisticRegression(),
            neighbors.KNNClassifier(),
            linear_model.Perceptron(),
            tree.HoeffdingTreeClassifier(),
        ],
        "n_models": range(1, 20),
        "w": range(1, 10),
        "adwin_delta": [0.001, 0.002, 0.005, 0.01],
        "bagging_method": ["bag", "me", "half", "wt", "subag"],
    },
    ADWINBaggingClassifier: {
        "model": [
            linear_model.LogisticRegression(),
            neighbors.KNNClassifier(),
            linear_model.Perceptron(),
            tree.HoeffdingTreeClassifier(),
        ],
        "n_models": range(1, 20),
    },
    AdaBoostClassifier: {
        "model": [
            linear_model.LogisticRegression(),
            neighbors.KNNClassifier(),
            linear_model.Perceptron(),
            tree.HoeffdingTreeClassifier(),
        ],
        "n_models": range(1, 20),
    },
    RobustScaler: {
        "with_centering": [True, False],
        "with_scaling": [True, False],
        "q_inf": np.arange(0, 0.5, 0.05),
        "q_sup": np.arange(0.55, 1, 0.05),
    },
    StandardScaler: {},
    AdaptiveStandardScaler: {"alpha": np.arange(0.1, 1, 0.1)},
    MaxAbsScaler: {},
    MinMaxScaler: {},
    Normalizer: {
        "order": [1, 2],
    },
    Binarizer: {"threshold": np.arange(0.0, 1.01, 0.05)},
    PolynomialExtender: {
        "degree": [2, 3, 4],
        "interaction_only": [True, False],
        "include_bias": [True, False],
    },
}
