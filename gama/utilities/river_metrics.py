from river.metrics import Accuracy
from river.metrics import BalancedAccuracy

# from river.metrics import BinaryMetric
# from river.metrics import ClassificationMetric
# from river.metrics import ClassificationReport
# from river.metrics import CohenKappa
# from river.metrics import Completeness
from river.metrics import ConfusionMatrix

# from river.metrics import CrossEntropy
# from river.metrics import ExactMatch
# from river.metrics import ExampleF1
# from river.metrics import ExampleFBeta
# from river.metrics import ExamplePrecision
# from river.metrics import ExampleRecall
from river.metrics import F1

# from river.metrics import FBeta
# from river.metrics import FowlkesMallows
# from river.metrics import GeometricMean
# from river.metrics import Hamming
# from river.metrics import HammingLoss
# from river.metrics import Homogeneity
# from river.metrics import Jaccard
# from river.metrics import KappaM
# from river.metrics import KappaT
# from river.metrics import LogLoss
# from river.metrics import MAE
# from river.metrics import MCC
# from river.metrics import MSE
# from river.metrics import MacroF1
# from river.metrics import MacroFBeta
# from river.metrics import MacroPrecision
# from river.metrics import MacroRecall
# from river.metrics import MatthewsCorrCoef
# from river.metrics import Metric
# from river.metrics import Metrics
# from river.metrics import MicroF1
# from river.metrics import MicroFBeta
# from river.metrics import MicroPrecision
# from river.metrics import MicroRecall
# from river.metrics import MultiClassMetric
# from river.metrics import MultiFBeta
# from river.metrics import MultiLabelConfusionMatrix
# from river.metrics import MultiOutputClassificationMetric
# from river.metrics import MultiOutputRegressionMetric
# from river.metrics import MutualInfo
# from river.metrics import NormalizedMutualInfo
# from river.metrics import PairConfusionMatrix
# from river.metrics import Precision
# from river.metrics import PrevalenceThreshold
# from river.metrics import Purity
# from river.metrics import Q0
# from river.metrics import Q2
# from river.metrics import R2
from river.metrics import RMSE

# from river.metrics import RMSLE
from river.metrics import ROCAUC

# from river.metrics import Rand
# from river.metrics import Recall
# from river.metrics import RegressionMetric
# from river.metrics import RegressionMultiOutput
# from river.metrics import Rolling
# from river.metrics import SMAPE
# from river.metrics import SorensenDice
# from river.metrics import TimeRolling
# from river.metrics import VBeta
# from river.metrics import VariationInfo
# from river.metrics import WeightedF1
# from river.metrics import WeightedFBeta
# from river.metrics import WeightedPrecision


def get_metric(metric):
    if metric == "accuracy":
        return Accuracy()
    elif metric == "balanced_accuracy":
        return BalancedAccuracy()
    elif metric == "f1":
        return F1()
    elif metric == "roc_auc":
        return ROCAUC()
    elif metric == "rmse":
        return RMSE()
    elif metric == "confusion_matrix":
        return ConfusionMatrix()
