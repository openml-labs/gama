import numpy as np

# SKAdaBoostRegressor
from d3m.primitives.sklearn_wrap import SKExtraTreesRegressor, SKGradientBoostingRegressor, \
                                        SKDecisionTreeRegressor, SKKNeighborsRegressor, SKLinearSVR,\
                                        SKRandomForestRegressor, SKFastICA, SKFeatureAgglomeration, SKMinMaxScaler, \
                                        SKNystroem, SKPCA, SKPolynomialFeatures, SKRBFSampler, SKStandardScaler,\
                                        SKOneHotEncoder, SKSelectPercentile

regressor_config_dict = {

    SKExtraTreesRegressor: {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    SKGradientBoostingRegressor: {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    #SKAdaBoostRegressor: {
    #    'n_estimators': [100],
    #    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #    'loss': ["linear", "square", "exponential"],
    #    'max_depth': range(1, 11)
    #},

    SKDecisionTreeRegressor: {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    #SKKNeighborsRegressor: {
    #    'n_neighbors': range(1, 101),
    #    'weights': ["uniform", "distance"],
    #    'p': [1, 2]
    #},

    SKLinearSVR: {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    SKRandomForestRegressor: {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

   SKFastICA: {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    SKFeatureAgglomeration: {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    },

    SKMinMaxScaler: {
    },

    SKNystroem: {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    SKPCA: {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    SKPolynomialFeatures: {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    SKRBFSampler: {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    SKStandardScaler: {
    },
    
    SKOneHotEncoder: {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False]
    },

    SKSelectPercentile: {
        'percentile': range(1, 100),
        'score_func': ['f_regression']
    }
}
