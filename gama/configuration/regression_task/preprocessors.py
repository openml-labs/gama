import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class PreprocessorConfig:
    """Manages the configuration space for preprocessors in supervised learning contexts

    PreprocessorConfig oversees the configuration space of preprocessors used in
    supervised machine learning tasks. This class facilitates the addition of
    new preprocessors and the modification of existing ones in the configuration space
    via standardised methods. The ConfigSpace library is used to designate the
    configuration space, enabling the creation of complex and adaptable
    configuration setups. For additional information on using constraints and
    various types of hyperparameters with ConfigSpace, refer to
    the ConfigSpace documentation, available at:
    https://automl.github.io/ConfigSpace/main/quickstart.html

    For further details how to add, modify and remove preprocessors, refer to the
    documentation of classification task:
    /configuration/classification_task/preprocessors.py


    Parameters
    ----------
    config_space : cs.ConfigurationSpace
        The ConfigSpace object that will be used to add the preprocessors and their
        respective hyperparameters.

    """

    def __init__(
        self,
        config_space: cs.ConfigurationSpace,
    ):
        if "preprocessors" not in config_space.meta:
            raise ValueError("Expected 'preprocessors' key in meta of config_space")
        self.config_space = config_space
        self.preprocessors_setup_map = {
            "Binarizer": self.setup_binarizer,
            "FastICA": self.setup_fast_ica,
            "FeatureAgglomeration": self.setup_feature_agglomeration,
            "MaxAbsScaler": self.setup_max_abs_scaler,
            "MinMaxScaler": self.setup_min_max_scaler,
            "Normalizer": self.setup_normalizer,
            "Nystroem": self.setup_nystroem,
            "PCA": self.setup_pca,
            "PolynomialFeatures": self.setup_polynomial_features,
            "RBFSampler": self.setup_rbf_sampler,
            "RobustScaler": self.setup_robust_scaler,
            "StandardScaler": self.setup_standard_scaler,
            "SelectFwe": self.setup_select_fwe,
            "SelectPercentile": self.setup_select_percentile,
            "VarianceThreshold": self.setup_variance_threshold,
        }

        self.cs_preprocessors_name = config_space.meta["preprocessors"]

    @property
    def shared_hyperparameters(self):
        return {
            "gamma": {"lower": 0.0, "upper": 1.01, "default_value": 0.05},
            "threshold": {"lower": 0.0, "upper": 1.01, "default_value": 0.05},
        }

    def setup_preprocessors(self):
        preprocessors_choices = list(self.preprocessors_setup_map.keys())

        if not preprocessors_choices:
            raise ValueError("No preprocessors to add to config space")

        preprocessors = csh.CategoricalHyperparameter(
            name=self.cs_preprocessors_name,
            choices=preprocessors_choices,
        )
        self.config_space.add_hyperparameter(preprocessors)

        for preprocessor_name in preprocessors_choices:
            if setup_func := self.preprocessors_setup_map.get(preprocessor_name):
                setup_func(preprocessors)

    def _add_hyperparameters_and_equals_conditions(
        self, local_vars: dict, preprocessor_name: str
    ):
        if "preprocessors" not in local_vars or not isinstance(
            local_vars["preprocessors"], csh.CategoricalHyperparameter
        ):
            raise ValueError(
                "Expected 'preprocessors' key with a CategoricalHyperparameter in local"
                "vars"
            )

        hyperparameters_to_add = [
            hyperparameter
            for hyperparameter in local_vars.values()
            if isinstance(hyperparameter, csh.Hyperparameter)
            and hyperparameter != local_vars["preprocessors"]
        ]

        conditions_to_add = [
            cs.EqualsCondition(
                hyperparameter, local_vars["preprocessors"], preprocessor_name
            )
            for hyperparameter in hyperparameters_to_add
        ]

        self.config_space.add_hyperparameters(hyperparameters_to_add)
        self.config_space.add_conditions(conditions_to_add)

    def setup_binarizer(self, preprocessors: csh.CategoricalHyperparameter):
        threshold = csh.UniformFloatHyperparameter(
            "threshold__Binarizer",
            **self.shared_hyperparameters["threshold"],
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "Binarizer")

    def setup_fast_ica(self, preprocessors: csh.CategoricalHyperparameter):
        tol = csh.UniformFloatHyperparameter(
            "tol__FastICA",
            lower=0.0,
            upper=1.01,
            default_value=0.05,
        )
        whiten = csh.CategoricalHyperparameter(
            "whiten__FastICA",
            choices=["unit-variance"],
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "FastICA")

    def setup_feature_agglomeration(self, preprocessors: csh.CategoricalHyperparameter):
        linkage = csh.CategoricalHyperparameter(
            "linkage__FeatureAgglomeration",
            choices=["ward", "complete", "average"],
        )
        affinity = csh.CategoricalHyperparameter(
            "affinity__FeatureAgglomeration",
            choices=["euclidean", "l1", "l2", "manhattan", "cosine"],
        )

        self._add_hyperparameters_and_equals_conditions(
            locals(), "FeatureAgglomeration"
        )

    def setup_max_abs_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_min_max_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_normalizer(self, preprocessors: csh.CategoricalHyperparameter):
        norm = csh.CategoricalHyperparameter(
            "norm__Normalizer",
            choices=["l1", "l2", "max"],
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "Normalizer")

    def setup_nystroem(self, preprocessors: csh.CategoricalHyperparameter):
        kernel = csh.CategoricalHyperparameter(
            "kernel__Nystroem",
            choices=[
                "rbf",
                "cosine",
                "chi2",
                "laplacian",
                "polynomial",
                "poly",
                "linear",
                "additive_chi2",
                "sigmoid",
            ],
        )
        gamma = csh.UniformFloatHyperparameter(
            "gamma__Nystroem",
            **self.shared_hyperparameters["gamma"],
        )
        n_components = csh.UniformIntegerHyperparameter(
            "n_components__Nystroem",
            lower=1,
            upper=11,
            default_value=1,
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "Nystroem")

    def setup_pca(self, preprocessors: csh.CategoricalHyperparameter):
        svd_solver = csh.CategoricalHyperparameter(
            "svd_solver__PCA",
            choices=["randomized"],
        )
        iterated_power = csh.UniformIntegerHyperparameter(
            "iterated_power__PCA",
            lower=1,
            upper=11,
            default_value=1,
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "PCA")

    def setup_polynomial_features(self, preprocessors: csh.CategoricalHyperparameter):
        degree = csh.CategoricalHyperparameter(
            "degree__PolynomialFeatures",
            choices=[2],
        )
        include_bias = csh.CategoricalHyperparameter(
            "include_bias__PolynomialFeatures",
            choices=[False],
        )
        interaction_only = csh.CategoricalHyperparameter(
            "interaction_only__PolynomialFeatures",
            choices=[False],
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "PolynomialFeatures")

    def setup_rbf_sampler(self, preprocessors: csh.CategoricalHyperparameter):
        gamma = csh.UniformFloatHyperparameter(
            "gamma__RBFSampler",
            **self.shared_hyperparameters["gamma"],
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "RBFSampler")

    def setup_robust_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_standard_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_select_fwe(self, preprocessors: csh.CategoricalHyperparameter):
        alpha = csh.UniformFloatHyperparameter(
            "alpha__SelectFwe",
            lower=0.0,
            upper=0.05,
            default_value=0.001,
        )
        # TODO Score func, how to add this?

        self._add_hyperparameters_and_equals_conditions(locals(), "SelectFwe")

    def setup_select_percentile(self, preprocessors: csh.CategoricalHyperparameter):
        percentile = csh.UniformIntegerHyperparameter(
            "percentile__SelectPercentile",
            lower=1,
            upper=100,
            default_value=1,
        )
        # TODO @Pieter – Score func, how to add this, you reckon?

        self._add_hyperparameters_and_equals_conditions(locals(), "SelectPercentile")

    def setup_variance_threshold(self, preprocessors: csh.CategoricalHyperparameter):
        threshold = csh.UniformFloatHyperparameter(
            "threshold__VarianceThreshold",
            lower=0.05,
            upper=1.01,
            default_value=0.05,
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "VarianceThreshold")
