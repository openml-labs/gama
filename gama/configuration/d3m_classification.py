import numpy as np

from d3m.primitives.sklearn_wrap import SKGaussianNB, SKBernoulliNB, SKMultinomialNB, SKDecisionTreeClassifier, \
                                        SKExtraTreesClassifier, SKRandomForestClassifier, SKGradientBoostingClassifier,\
                                        SKKNeighborsClassifier, SKLinearSVC, SKLogisticRegression, \
                                        SKFastICA, SKFeatureAgglomeration, SKMinMaxScaler, SKNystroem, \
                                        SKPCA, SKPolynomialFeatures, SKRBFSampler, SKStandardScaler, \
                                        SKOneHotEncoder, SKSelectPercentile

classifier_config_dict = {

    # Classifiers
    SKGaussianNB: {
    },

    SKBernoulliNB: {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    SKMultinomialNB: {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    SKDecisionTreeClassifier: {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    SKExtraTreesClassifier: {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    SKRandomForestClassifier: {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    SKGradientBoostingClassifier: {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    #SKKNeighborsClassifier: {
    #    'n_neighbors': range(1, 101),
    #    'weights': ["uniform", "distance"],
    #    'p': [1, 2]
    #},

    SKLinearSVC: {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

#    Disabled for testing.  To slow for debugging!
    SKLogisticRegression: {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
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
        'score_func': ['f_classif']
    }

}
