import numpy as np

#classifiers
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.ensemble import LeveragingBaggingClassifier
from river.neighbors import KNNADWINClassifier

#preprocessing
from river.preprocessing import (
    AdaptiveStandardScaler,
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler)

#feature extraction
from river.feature_extraction import PolynomialExtender

#feature selection
from river.feature_selection import SelectKBest

clf_config_online = {
    "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    "fit_prior": [True, False],
    "min_samples_split": range(2, 21),
    "min_samples_leaf": range(1, 21),
    KNNADWINClassifier: {
        "n_neighbors": range(1,15),
        "window_size":[100, 500, 1000, 1500, 2000],
        "leaf_size":range(5,50,5),
        "p": np.arange(1,2,0.2)
    },
    RobustScaler: {
        "with_centering": [True, False],
        "with_scaling": [True, False],
        "q_inf": np.arange(0,1,0.05),
        "q_sup": np.arange(0,1,0.05)
    },

}
