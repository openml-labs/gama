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

clf_config = {
    "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    "fit_prior": [True, False],
    "min_samples_split": range(2, 21),
    "min_samples_leaf": range(1, 21),
    HoeffdingAdaptiveTreeClassifier: {
        "grace_period": range(50, 350),
        "split_criterion": ["info_gini", "gini", "hellinger"],
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "tie_threshold": np.arange(0.02, 0.08, 0.01),
        "leaf_prediction": ["mc", "nb", "nba"],
        "nb_threshold": range(0,50,10),
        "splitter": ["tree.splitter.EBSTSplitter", "tree.splitter.HistogramSplitter",
                     "tree.splitter.TEBSTSplitter", "tree.splitter.GaussianSplitter"],
        "bootstrap_sampling": [True, False],
        "drift_window_threshold": range(100,500,100),
        "adwin_confidence": [2e-4, 2e-3, 2e-2]
    },
    LeveragingBaggingClassifier: {},
    KNNADWINClassifier: {
        "n_neighbors": range(1,15),
        "window_size":[100, 500, 1000, 1500, 2000],
        "leaf_size":range(5,50,5),
        "p": np.arange(1,2,0.2)
    },
    AdaptiveStandardScaler:{"alpha":np.arange(0.1,1,0.1)},
    Binarizer: {"threshold": np.arange(0.0, 1.01, 0.05)},
    MaxAbsScaler: {},
    MinMaxScaler: {},
    Normalizer: {"order": [1,2]},
    #Can we use onehotencoder or imputer  in gama? requires learn before transform
    #OneHotEncoder: {
    #    sparse: [True, False]
    #},
    RobustScaler: {
        "with_centering": [True, False],
        "with_scaling": [True, False],
        "q_inf": np.arange(0,1,0.05),
        "q_sup": np.arange(0,1,0.05)
    },
    StandardScaler: {},
    PolynomialExtender:{
        "degree": [2,3,4],
        "interaction_only": [True, False],
        "include_bias": [True, False]
    },
    SelectKBest:{
        "k": range(5,20,5)
    }

}
