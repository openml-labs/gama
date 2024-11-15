import ConfigSpace as cs

from gama.configuration.configuration_task_test import (
    ClassifierConfigTest,
    PreprocessorConfigTest,
)

# A configuration with limited operators for unit tests 🧪

config_space = cs.ConfigurationSpace(
    meta={
        # "gama_system_name": "current_configuration_name",
        "estimators": "classifiers",
        "preprocessors": "preprocessors",
    }
)

classifier_config = ClassifierConfigTest(config_space)
classifier_config.setup_classifiers()

preprocessor_config = PreprocessorConfigTest(config_space)
preprocessor_config.setup_preprocessors()
