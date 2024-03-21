import ConfigSpace as cs

from .regression_task import RegressorConfig, PreprocessorConfig

# Regressors & Preprocessors 🚀

# This script is your ticket to configuring a ConfigSpace object, teeming with
# regressors and preprocessors. We are diving in with the RegressorConfig and
# PreprocessorConfig classes to fill the configuration space with a slew of
# hyperparameters and options.

# Customise Your Space 🔧

# Want just classifiers? No biggie! Just comment out or remove the PreprocessorConfig
# setup. Voila! You're left with a sleek, regressor-only configuration space.

# Want to add more regressors or preprocessors? Easy! Just add them to the
# RegressorConfig or PreprocessorConfig classes, respectively. You can even
# add your own custom regressors or preprocessors. Just make sure they are
# compatible with scikit-learn's API.

# Meta-Parameters 📝

# The meta-parameters are the "estimators" and "preprocessors" keys in the
# configuration space. These are used to identify the regressors and preprocessors
# by the internal system. They are not hyperparameters, and should not be
# changed, except by advanced users. If you do change them, make sure to change
# the corresponding values in the current configuration space, i.e. in RegressorConfig
# and PreprocessorConfig.

# 👩‍💻👨‍💻 Happy configuring, and may your machine learning models shine!

config_space = cs.ConfigurationSpace(
    meta={
        # "gama_system_name": "current_configuration_name",
        "estimators": "regressors",
        "preprocessors": "preprocessors",
    }
)

regressor_config = RegressorConfig(config_space)
regressor_config.setup_regressors()

preprocessor_config = PreprocessorConfig(config_space)
preprocessor_config.setup_preprocessors()
