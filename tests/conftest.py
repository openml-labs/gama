import pytest
from gama import GamaClassifier
from gama.genetic_programming.components import Individual
from gama.configuration.testconfiguration import clf_config
from gama.genetic_programming.compilers.scikitlearn import compile_individual


@pytest.fixture
def pset():
    gc = GamaClassifier(config=clf_config, scoring="accuracy", store="nothing")
    yield gc._pset
    gc.cleanup("all")


@pytest.fixture
def opset():
    gc = GamaClassifier(config=clf_config, scoring="accuracy", store="nothing")
    yield gc._operator_set
    gc.cleanup("all")


@pytest.fixture
def GNB(pset):
    return Individual.from_string("GaussianNB(data)", pset, compile_individual)


@pytest.fixture
def RS_MNB(pset):
    return Individual.from_string(
        "MultinomialNB(RobustScaler(data), alpha=1.0, fit_prior=True)",
        pset,
        compile_individual,
    )


@pytest.fixture
def SS_BNB(pset):
    return Individual.from_string(
        "BernoulliNB(StandardScaler(data), alpha=0.1, fit_prior=True)",
        pset,
        compile_individual,
    )


@pytest.fixture
def SS_RBS_SS_BNB(pset):
    return Individual.from_string(
        "BernoulliNB(StandardScaler(RobustScaler(StandardScaler(data))), alpha=0.1, fit_prior=True)",  # noqa: E501
        pset,
        compile_individual,
    )


@pytest.fixture
def LinearSVC(pset):
    individual_str = """LinearSVC(data,
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l2',
            LinearSVC.tol=1e-05)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")
    return Individual.from_string(individual_str, pset, None)


@pytest.fixture
def ForestPipeline(pset):
    individual_str = """RandomForestClassifier(
            FeatureAgglomeration(
                    data,
                    FeatureAgglomeration.affinity='l2',
                    FeatureAgglomeration.linkage='complete'
                    ),
            RandomForestClassifier.bootstrap=True,
            RandomForestClassifier.criterion='gini',
            RandomForestClassifier.max_features=0.6,
            RandomForestClassifier.min_samples_leaf=7,
            RandomForestClassifier.min_samples_split=6,
            RandomForestClassifier.n_estimators=100)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")

    return Individual.from_string(individual_str, pset, None)


@pytest.fixture
def InvalidLinearSVC(pset):
    individual_str = """LinearSVC(data,
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l1',
            LinearSVC.tol=1e-05)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")
    return Individual.from_string(individual_str, pset, compile_individual)
