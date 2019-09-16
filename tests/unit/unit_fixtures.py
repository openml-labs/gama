import pytest
from gama import GamaClassifier
from gama.genetic_programming.components import Individual
from gama.configuration.testconfiguration import clf_config
from gama.genetic_programming.compilers.scikitlearn import compile_individual

@pytest.fixture
def pset():
    gc = GamaClassifier(config=clf_config, scoring='accuracy')
    gc.delete_cache()
    return gc._pset


@pytest.fixture
def GaussianNB(pset):
    return Individual.from_string("GaussianNB(data)",
                                  pset, compile_individual)


@pytest.fixture
def MultinomialNBRobustScaler(pset):
    return Individual.from_string("MultinomialNB(RobustScaler(data), alpha=1.0, fit_prior=True)",
                                  pset, compile_individual)


@pytest.fixture
def BernoulliNBStandardScaler(pset):
    return Individual.from_string("BernoulliNB(StandardScaler(data), alpha=0.1, fit_prior=True)",
                                  pset, compile_individual)


@pytest.fixture
def LinearSVC(pset):
    individual_str = """LinearSVC(data,
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l2',
            LinearSVC.tol=1e-05)"""
    individual_str = ''.join(individual_str.split()).replace(',', ', ')
    return Individual.from_string(individual_str, pset, None)


@pytest.fixture
def RandomForestPipeline(pset):
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
    individual_str = ''.join(individual_str.split()).replace(',', ', ')

    return Individual.from_string(individual_str, pset, None)
