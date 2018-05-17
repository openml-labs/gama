import numpy as np

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler, StandardScaler, Binarizer
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_classif, VarianceThreshold, RFE, SelectFromModel, f_regression

# Default TPOT Classifier config for comparison

clf_config = {

    'alpha' : [1e-3, 1e-2, 1e-1, 1., 10., 100.],
    'fit_prior': [True, False],
    'min_samples_split': range(2, 21),
    'min_samples_leaf': range(1, 21),  
    
    # Classifiers
    GaussianNB: {
    },

    BernoulliNB: {
        'alpha': [],
        'fit_prior': []
    },

    MultinomialNB: {
        'alpha': [],
        'fit_prior': []
    },

    DecisionTreeClassifier: {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': [],
        'min_samples_leaf': []
    },

    ExtraTreesClassifier: {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': [],
        'min_samples_leaf': [],
        'bootstrap': [True, False]
    },
    
    RandomForestClassifier: {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    GradientBoostingClassifier: {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    KNeighborsClassifier: {
        'n_neighbors': range(1, 51),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    LinearSVC: {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [False, True], 
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'param_check': [lambda params: (not params['dual'] or params['penalty'] == "l2")
                                        and not (params['penalty'] == "l1" and params['loss'] == "hinge")
                                        and not (params['penalty'] == "l2" and params['loss'] == "hinge" and not params['dual']) ]
        #'param_check': [lambda params: ( not params['dual']
         #       (not params['dual'] or params['penalty'] == "l2")
         #                                and (params['penalty'] == "l1" and params['penalty'] == "squared_hinge")]
    },
    
    LogisticRegression: {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [False, True],
        'param_check': [lambda params: not params['dual'] or params['penalty'] == "l2"]
    },

    #'xgboost.XGBClassifier': {
    #    'n_estimators': [100],
    #    'max_depth': range(1, 11),
    #    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #    'subsample': np.arange(0.05, 1.01, 0.05),
    #    'min_child_weight': range(1, 21),
    #    'nthread': [1]
    #},
    
    # Preprocesssors
    Binarizer: {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    FastICA: {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    FeatureAgglomeration: {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'],
        'param_check': [lambda params: (not params['linkage'] == "ward") or params['affinity'] == "euclidean"]
    },
    
    MaxAbsScaler: {
    },
    
    MinMaxScaler: {
    },
    
    Normalizer: {
        'norm': ['l1', 'l2', 'max']
    },
    
    Nystroem: {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },
    
    PCA: {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },
    
    PolynomialFeatures: {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },
    
    RBFSampler: {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },
    
    RobustScaler: {
    },
    
    StandardScaler: {
    },
    
    #'tpot.builtins.ZeroCount': {
    #},
    
    #'tpot.builtins.OneHotEncoder': {
    #    'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
    #    'sparse': [False]
    #},
    
    # Selectors
    SelectFwe: {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            f_classif: None
        }
    },
    
    SelectPercentile: {
        'percentile': range(1, 100),
        'score_func': {
            f_classif: None
        }
    },
    
   VarianceThreshold: {
        'threshold': np.arange(0.05, 1.01, 0.05)
    }
#   RFE: {
#    'step': np.arange(0.05, 1.01, 0.05),
#    'estimator': {
#            ExtraTreesClassifier : []
#            }
#   },
#   SelectFromModel: {
#       'threshold': np.arange(0, 1.01, 0.05),
#       'estimator': {
#            ExtraTreesClassifier : []
#            }
#    }
#           RFE: {
#    'step': np.arange(0.05, 1.01, 0.05),
#    'estimator': {
#        ExtraTreesClassifier: {
#            'n_estimators': [100],
#            'criterion': ['gini', 'entropy'],
#            'max_features': np.arange(0.05, 1.01, 0.05)
#        }
#    }
#   },
#   SelectFromModel: {
#       'threshold': np.arange(0, 1.01, 0.05),
#       'estimator': {
#           ExtraTreesClassifier: {
#            'n_estimators': [100],
#               'criterion': ['gini', 'entropy'],
#               'max_features': np.arange(0.05, 1.01, 0.05)
#           }
#       }
#    }
}
   
""" 
RFE: {
    'step': np.arange(0.05, 1.01, 0.05),
    'estimator': {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ['gini', 'entropy'],
            'max_features': np.arange(0.05, 1.01, 0.05)
        }
    }
},
SelectFromModel: {
    'threshold': np.arange(0, 1.01, 0.05),
    'estimator': {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ['gini', 'entropy'],
            'max_features': np.arange(0.05, 1.01, 0.05)
        }
    }
"""

from sklearn.linear_model import ElasticNetCV, LassoLarsCV, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

# TPOT Regressor configuration

reg_config = {

    ElasticNetCV: {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    ExtraTreesRegressor: {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    GradientBoostingRegressor: {
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

    AdaBoostRegressor: {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"],
        # 'max_depth': range(1, 11) not available in sklearn==0.19.1
    },

    DecisionTreeRegressor: {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    KNeighborsRegressor: {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    LassoLarsCV: {
        'normalize': [True, False]
    },

    LinearSVR: {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    RandomForestRegressor: {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    RidgeCV: {
    },

#    'xgboost.XGBRegressor': {
#        'n_estimators': [100],
#        'max_depth': range(1, 11),
#        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
#        'subsample': np.arange(0.05, 1.01, 0.05),
#        'min_child_weight': range(1, 21),
#        'nthread': [1]
#    },

    # Preprocesssors
    Binarizer: {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    FeatureAgglomeration: {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    MaxAbsScaler: {
    },

    MinMaxScaler: {
    },

    Normalizer: {
        'norm': ['l1', 'l2', 'max']
    },

    Nystroem: {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    PCA: {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    PolynomialFeatures: {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    RBFSampler: {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    RobustScaler: {
    },

    StandardScaler: {
    },

 #   'tpot.builtins.ZeroCount': {
 #   },

 #   'tpot.builtins.OneHotEncoder': {
 #       'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
 #       'sparse': [False]
 #   },

    # Selectors
    SelectFwe: {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            f_regression: None
        }
    },

    SelectPercentile: {
        'percentile': range(1, 100),
        'score_func': {
            f_regression: None
        }
    },

    VarianceThreshold: {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

#    SelectFromModel: {
#        'threshold': np.arange(0, 1.01, 0.05),
#        'estimator': {
#           'sklearn.ensemble.ExtraTreesRegressor': {
#                'n_estimators': [100],
#                'max_features': np.arange(0.05, 1.01, 0.05)
#            }
#        }
#    }

}
