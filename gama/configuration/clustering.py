import numpy as np

from sklearn.cluster import (KMeans,
                             SpectralClustering,
                             MeanShift,
                             AgglomerativeClustering,
                             DBSCAN,
                             OPTICS,
                             Birch,
                             FeatureAgglomeration)

from sklearn.preprocessing import (MaxAbsScaler,
                                   MinMaxScaler,
                                   RobustScaler,
                                   Normalizer,
                                   PolynomialFeatures)

from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import VarianceThreshold


cluster_config = {

    'n_clusters': np.arange(2, 21),

    KMeans: {'n_clusters': [],
             "n_init": [10, 15],
             "max_iter": [100, 200, 300],
             },

    MeanShift: {"bin_seeding": [True, False],
                "min_bin_freq": np.arange(1, 5),
                "max_iter": [100, 200, 300]
                },

    SpectralClustering: {'n_clusters': [],
                         "gamma": np.arange(0.8, 1.1, 0.1),
                         "n_init": [10, 15]
                         },

    AgglomerativeClustering: {'n_clusters': [],
                              "affinity": ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
                              "linkage": ['ward', 'complete', 'average', 'single'],
                              "param_check": [lambda params: (not params["linkage"] == "ward") or params["affinity"] == "euclidean"]
                              },

    DBSCAN: {'eps': np.arange(0.1, 0.5, 0.05),
             "min_samples": np.arange(3, 8),
             "p": [1, 2],
             },

    OPTICS: {'min_samples': np.arange(3, 8),
             "p": [1, 2],
             'xi': np.arange(0.05, 0.5, 0.05),
             },

    Birch: {"n_clusters": [],
            'threshold': np.arange(0.2, 0.8, 0.1),
            "branching_factor": [25, 50, 75]
            },

    # Preprocesssors
    FeatureAgglomeration: {"linkage": ["ward", "complete", "average"],
                           "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
                           "param_check": [lambda params: (not params["linkage"] == "ward") or params["affinity"] == "euclidean"],
                           },

    FastICA: {"tol": np.arange(0.0, 1.01, 0.05)},

    PCA: {"svd_solver": ["randomized"], "iterated_power": range(1, 11)},

    PolynomialFeatures: {"degree": [2],
                         "include_bias": [False],
                         "interaction_only": [False]
                         },

    MaxAbsScaler: {},

    MinMaxScaler: {},

    RobustScaler: {},

    Normalizer: {"norm": ["l1", "l2", "max"]},

    Nystroem: {"kernel": ["rbf",
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

    RBFSampler: {"gamma": np.arange(0.0, 1.01, 0.05)},

    # Selectors
    VarianceThreshold: {"threshold": np.arange(0.05, 1.01, 0.05)}
    }