from sklearn.naive_bayes import BernoulliNB, GaussianNB

from gama.configuration.parser import merge_configurations


def test_merge_configuration():
    """ Test merging two simple configurations works as expected. """

    one = {"alpha": [0, 1], BernoulliNB: {"fit_prior": [True, False]}}
    two = {"alpha": [0, 2], GaussianNB: {"fit_prior": [True, False]}}
    expected_merged = {
        "alpha": [0, 1, 2],
        GaussianNB: {"fit_prior": [True, False]},
        BernoulliNB: {"fit_prior": [True, False]},
    }

    actual_merged = merge_configurations(one, two)
    assert expected_merged == actual_merged
