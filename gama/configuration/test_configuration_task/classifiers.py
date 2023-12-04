import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class TestClassifierConfig:
    def __init__(
        self,
        config_space: cs.ConfigurationSpace,
    ):
        if "estimators" not in config_space.meta:
            raise ValueError("Expected 'estimators' key in meta of config_space")
        self.config_space = config_space
        self.classifiers_setup_map = {
            "BernoulliNB": self.setup_bernoulliNB,
            "MultinomialNB": self.setup_multinomialNB,
            "GaussianNB": self.setup_gaussianNB,
            "DecisionTreeClassifier": self.setup_decision_tree,
            "ExtraTreesClassifier": self.setup_extra_trees,
            "RandomForestClassifier": self.setup_random_forest,
            "GradientBoostingClassifier": self.setup_gradient_boosting,
            "KNeighborsClassifier": self.setup_k_neighbors,
            "LinearSVC": self.setup_linear_svc,
            "LogisticRegression": self.setup_logistic_regression,
        }
        self.cs_estimators_name = self.config_space.meta["estimators"]

    @property
    def shared_hyperparameters(self):
        return {
            "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            "fit_prior": [True, False],
            "criterion": ["gini", "entropy"],
            "max_depth": {"lower": 1, "upper": 11},
            "min_samples_split": {"lower": 2, "upper": 21},
            "min_samples_leaf": {"lower": 1, "upper": 21},
            "max_features": {"lower": 0.05, "upper": 1.01, "default_value": 1.0},
            "n_estimators": [100],
            "bootstrap": [True, False],
            "dual": [True, False],
            "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        }

    def setup_classifiers(self):
        classifiers_choices = list(self.classifiers_setup_map.keys())

        if not classifiers_choices:
            raise ValueError("No classifiers to add to config space")

        classifiers = csh.CategoricalHyperparameter(
            name=self.cs_estimators_name,
            choices=classifiers_choices,
        )
        self.config_space.add_hyperparameter(classifiers)

        for classifier_name in classifiers_choices:
            if setup_func := self.classifiers_setup_map.get(classifier_name):
                setup_func(classifiers)

    def _add_hyperparameters_and_equals_conditions(
        self, local_vars: dict, estimator_name: str
    ):
        if "classifiers" not in local_vars or not isinstance(
            local_vars["classifiers"], csh.CategoricalHyperparameter
        ):
            raise ValueError(
                "Expected 'classifiers' key with a CategoricalHyperparameter in local"
                "vars"
            )

        hyperparameters_to_add = [
            hyperparameter
            for hyperparameter in local_vars.values()
            if isinstance(hyperparameter, csh.Hyperparameter)
            and hyperparameter != local_vars["classifiers"]
        ]

        conditions_to_add = [
            cs.EqualsCondition(
                hyperparameter, local_vars["classifiers"], estimator_name
            )
            for hyperparameter in hyperparameters_to_add
        ]

        self.config_space.add_hyperparameters(hyperparameters_to_add)
        self.config_space.add_conditions(conditions_to_add)

    def setup_bernoulliNB(self, classifiers: csh.CategoricalHyperparameter):
        alpha_NB = csh.CategoricalHyperparameter(
            "alpha__bernoulliNB", self.shared_hyperparameters["alpha"]
        )
        fit_prior = csh.CategoricalHyperparameter(
            "fit_prior__bernoulliNB", self.shared_hyperparameters["fit_prior"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "BernoulliNB")

    def setup_multinomialNB(self, classifiers: csh.CategoricalHyperparameter):
        alpha_NB = csh.CategoricalHyperparameter(
            "alpha__multinomialNB", self.shared_hyperparameters["alpha"]
        )
        fit_prior = csh.CategoricalHyperparameter(
            "fit_prior__multinomialNB", self.shared_hyperparameters["fit_prior"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "MultinomialNB")

    def setup_gaussianNB(self, classifiers: csh.CategoricalHyperparameter):
        # GaussianNB has no hyperparameters
        pass

    def setup_decision_tree(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__decision_tree", self.shared_hyperparameters["criterion"]
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__decision_tree", **self.shared_hyperparameters["max_depth"]
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__decision_tree",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__decision_tree",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "DecisionTreeClassifier"
        )

    def setup_extra_trees(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__extra_trees", self.shared_hyperparameters["criterion"]
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__extra_trees", **self.shared_hyperparameters["max_depth"]
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__extra_trees",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__extra_trees",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features__extra_trees", **self.shared_hyperparameters["max_features"]
        )
        n_estimators = csh.CategoricalHyperparameter(
            "n_estimators__extra_trees", self.shared_hyperparameters["n_estimators"]
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap__extra_trees", self.shared_hyperparameters["bootstrap"]
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "ExtraTreesClassifier"
        )

    def setup_random_forest(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__random_forest", self.shared_hyperparameters["criterion"]
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__random_forest", **self.shared_hyperparameters["max_depth"]
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split", **self.shared_hyperparameters["min_samples_split"]
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf", **self.shared_hyperparameters["min_samples_leaf"]
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features", **self.shared_hyperparameters["max_features"]
        )
        n_estimators = csh.CategoricalHyperparameter(
            "n_estimators__random_forest", self.shared_hyperparameters["n_estimators"]
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap", self.shared_hyperparameters["bootstrap"]
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "RandomForestClassifier"
        )

    def setup_gradient_boosting(self, classifiers: csh.CategoricalHyperparameter):
        sub_sample = csh.CategoricalHyperparameter(
            "subsample", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        learning_rate = csh.CategoricalHyperparameter(
            "learning_rate", [1e-3, 1e-2, 1e-1, 0.5, 1.0]
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features__gradient_boosting",
            **self.shared_hyperparameters["max_features"],
        )
        n_estimators = csh.CategoricalHyperparameter(
            "n_estimators__gradient_boosting",
            self.shared_hyperparameters["n_estimators"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "GradientBoostingClassifier"
        )

    def setup_k_neighbors(self, classifiers: csh.CategoricalHyperparameter):
        n_neighbors = csh.UniformIntegerHyperparameter("n_neighbors", 1, 51)
        weights = csh.CategoricalHyperparameter("weights", ["uniform", "distance"])
        p = csh.UniformIntegerHyperparameter("p", 1, 2)
        self._add_hyperparameters_and_equals_conditions(
            locals(), "KNeighborsClassifier"
        )

    def setup_linear_svc(self, classifiers: csh.CategoricalHyperparameter):
        loss = csh.CategoricalHyperparameter(
            "loss__linear_svc", ["hinge", "squared_hinge"]
        )
        penalty = csh.CategoricalHyperparameter("penalty__linear_svc", ["l1", "l2"])
        dual = csh.CategoricalHyperparameter(
            "dual__svc", self.shared_hyperparameters["dual"]
        )
        tol = csh.CategoricalHyperparameter("tol__svc", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        C = csh.CategoricalHyperparameter("C__svc", self.shared_hyperparameters["C"])
        self._add_hyperparameters_and_equals_conditions(locals(), "LinearSVC")

        # Forbidden clause: Penalty 'l1' cannot be used with loss 'hinge'
        forbidden_penalty_loss = cs.ForbiddenAndConjunction(
            cs.ForbiddenEqualsClause(self.config_space["penalty__linear_svc"], "l1"),
            cs.ForbiddenEqualsClause(self.config_space["loss__linear_svc"], "hinge"),
        )
        self.config_space.add_forbidden_clause(forbidden_penalty_loss)

    def setup_logistic_regression(self, classifiers: csh.CategoricalHyperparameter):
        penalty = csh.CategoricalHyperparameter(
            "penalty__logistic_regression", ["l1", "l2"]
        )
        C = csh.CategoricalHyperparameter(
            "C__logistic_regression", self.shared_hyperparameters["C"]
        )
        dual = csh.CategoricalHyperparameter(
            "dual__logistic_regression", self.shared_hyperparameters["dual"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "LogisticRegression")
