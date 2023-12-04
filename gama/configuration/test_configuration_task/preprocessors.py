import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class TestPreprocessorConfig:
    def __init__(
        self,
        config_space: cs.ConfigurationSpace,
    ):
        if "preprocessors" not in config_space.meta:
            raise ValueError("Expected 'preprocessors' key in meta of config_space")
        self.config_space = config_space
        self.preprocessors_setup_map = {
            "SelectFwe": self.setup_select_fwe,
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
            "SelectPercentile": self.setup_select_percentile,
            "VarianceThreshold": self.setup_variance_threshold,
        }
        self.cs_preprocessors_name = config_space.meta["preprocessors"]

    @property
    def shared_hyperparameters(self):
        return {
            "gamma": {"lower": 0.01, "upper": 1.01, "default_value": 1.0},
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

    def setup_select_fwe(self, preprocessors: csh.CategoricalHyperparameter):
        alpha = csh.UniformFloatHyperparameter(
            "alpha__SelectFwe", 0, 0.05, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "SelectFwe")

    def setup_binarizer(self, preprocessors: csh.CategoricalHyperparameter):
        threshold = csh.UniformFloatHyperparameter(
            "threshold__binarizer", 0.0, 1.01, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "Binarizer")

    def setup_fast_ica(self, preprocessors: csh.CategoricalHyperparameter):
        whiten = csh.CategoricalHyperparameter("whiten", ["unit-variance"])
        tol = csh.UniformFloatHyperparameter(
            "tol__fast_ica", 0.0, 1.01, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "FastICA")

    def setup_feature_agglomeration(self, preprocessors: csh.CategoricalHyperparameter):
        linkage = csh.CategoricalHyperparameter(
            "linkage__feature_agglomeration", ["ward", "complete", "average"]
        )
        affinity = csh.CategoricalHyperparameter(
            "affinity__feature_agglomeration",
            ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "FeatureAgglomeration"
        )

        # Forbidden clause: Linkage is different from 'ward' and affinity is 'euclidean'
        forbidden_penalty_loss = cs.ForbiddenAndConjunction(
            cs.ForbiddenInClause(
                self.config_space["linkage__feature_agglomeration"],
                ["complete", "average"],
            ),
            cs.ForbiddenEqualsClause(
                self.config_space["affinity__feature_agglomeration"], "euclidean"
            ),
        )
        self.config_space.add_forbidden_clause(forbidden_penalty_loss)

    def setup_max_abs_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_min_max_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_normalizer(self, preprocessors: csh.CategoricalHyperparameter):
        norm = csh.CategoricalHyperparameter("norm", ["l1", "l2", "max"])
        self._add_hyperparameters_and_equals_conditions(locals(), "Normalizer")

    def setup_nystroem(self, preprocessors: csh.CategoricalHyperparameter):
        kernel = csh.CategoricalHyperparameter(
            "kernel",
            [
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
            "gamma__nystroem", **self.shared_hyperparameters["gamma"]
        )
        n_components = csh.UniformIntegerHyperparameter("n_components", 1, 11)
        self._add_hyperparameters_and_equals_conditions(locals(), "Nystroem")

    def setup_pca(self, preprocessors: csh.CategoricalHyperparameter):
        svd_solver = csh.CategoricalHyperparameter("svd_solver", ["randomized"])
        iterated_power = csh.UniformIntegerHyperparameter("iterated_power", 1, 11)
        self._add_hyperparameters_and_equals_conditions(locals(), "PCA")

    def setup_polynomial_features(self, preprocessors: csh.CategoricalHyperparameter):
        include_bias = csh.CategoricalHyperparameter("include_bias", [False])
        interaction_only = csh.CategoricalHyperparameter("interaction_only", [False])
        self._add_hyperparameters_and_equals_conditions(locals(), "PolynomialFeatures")

    def setup_rbf_sampler(self, preprocessors: csh.CategoricalHyperparameter):
        gamma = csh.UniformFloatHyperparameter(
            "gamma__rbf_sampler", **self.shared_hyperparameters["gamma"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "RBFSampler")

    def setup_robust_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_standard_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_select_percentile(self, preprocessors: csh.CategoricalHyperparameter):
        percentile = csh.UniformIntegerHyperparameter("percentile", 1, 100)
        self._add_hyperparameters_and_equals_conditions(locals(), "SelectPercentile")

    def setup_variance_threshold(self, preprocessors: csh.CategoricalHyperparameter):
        threshold = csh.UniformFloatHyperparameter(
            "threshold__variance_threshold", 0.05, 1.01, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "VarianceThreshold")
