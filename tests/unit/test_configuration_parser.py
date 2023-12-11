from gama.utilities.config_space import merge_configurations

from gama.configuration.testconfiguration import (
    config_space as classification_config_space,
)
from gama.configuration.regression import config_space as regression_config_space


def test_merge_configuration():
    """Test merging two simple configurations works as expected."""

    test_classification_config = classification_config_space
    test_regression_config = regression_config_space

    prefix = "merged"
    delimiter = "_"

    merged_config = merge_configurations(
        test_classification_config,
        test_regression_config,
        prefix=prefix,
        delimiter=delimiter,
    )

    assert (
        test_classification_config.meta["estimators"]
        in merged_config.get_hyperparameters_dict()
    )
    assert (
        prefix + delimiter + test_regression_config.meta["estimators"]
    ) in merged_config.get_hyperparameters_dict()
