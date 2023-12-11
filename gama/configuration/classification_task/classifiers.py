import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class ClassifierConfig:
    """Manages the configuration space for classifiers in supervised learning contexts

    ClassifierConfig oversees the configuration space of classifiers utilised for a
    supervised machine learning tasks. This class facilitates the addition of
    new classifiers and the modification of existing ones in the configuration space
    via standardised methods. The ConfigSpace library is utilised to designate the
    configuration space, which enables the creation of intricate and adaptable
    configuration setups. For additional information on utilising constraints and
    various types of hyperparameters, with ConfigSpace, we refer the reader to
    ConfigSpace documentation, available at:
    https://automl.github.io/ConfigSpace/main/quickstart.html

    Add a classifier 💡
    ----------------

    1️⃣ To add a new classifier, define its setup method following the naming convention
    `setup_classifierName`. This method should:
        * Define hyperparameters specific to the classifier.
        * Use `_add_hyperparameters_and_equals_conditions` to add these
        hyperparameters to the config space with appropriate conditions.

    2️⃣ Next, your setup method need to be added to the `classifiers_setup_map` in
    the `__init__` method, where the key should be the Sci-kit learn name of your
    classifier, and the value should be pointing to your newly setup method.

    voila! 🎉 You are done! Your classifier is now added to the config space.

    How to use the shared hyperparameters 🪢
    -------------------------------------

    The shared hyperparameters are hyperparameters that are shared across multiple
    classifiers. These hyperparameters are defined in the `shared_hyperparameters`
    property. To use these hyperparameters, simply add them to the setup method of
    the classifier you are adding. For example, to add the `C` hyperparameter to the
    `LogisticRegression` classifier, add the following line to the
    `setup_logistic_regression` method:

    >>>    C = csh.CategoricalHyperparameter(
    >>>        "C__LogisticRegression", self.shared_hyperparameters["C"]
    >>>    )

    voila! 🎉 The `C` hyperparameter is now added to the LogisticRegression classifier
    with the shared value available in the `shared_hyperparameters` property.

    How to name my hyperparameters ✍️
    ------------------------------

    The hyperparameters you add to the config space should be named in the following
    format if similar hyperparameters names can be found in other classifiers:

    >>>    <hyperparameter_name>__<ClassifierName>

    For example, the `C` hyperparameter for the `LogisticRegression` classifier should
    be named `C__LogisticRegression` given that the `C` hyperparameter is also
    available in the `LinearSVC` classifier. This naming convention is used to ensure
    that the hyperparameters are added to the correct classifier in the config space.

    If your hyperparameter name is unique to your classifier, you can name it as you
    please without the need to have `__<ClassifierName>` at the end of the name.
    Nonetheless, following the naming convention would in any way not cause any issues.

    Modify an existing classifier 💅
    -------------------

    To modify an existing classifier, adjust its respective setup method and the
    shared hyperparameters property as needed by modifying the values of the
    hyperparameters. For example, to change the value of the `C` hyperparameter for
    the `LogisticRegression` classifier, change the value of the `C` hyperparameter
    in the `shared_hyperparameters` property by:

    >>>    "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],

    The `C` hyperparameter will then be added to the config space with the appropriate
    value. However, be cautious, if you change values in the shared hyperparameters
    property, it will be changed for all classifiers that use that hyperparameter.
    If you want this change to only apply to a specific classifier, you should add
    the hyperparameter to the setup method of that classifier. E.g. if you want to
    change the value of the `C` hyperparameter for the `LogisticRegression` classifier,
    and only want this change to apply to the `LogisticRegression` classifier, add
    the following line to the `setup_logistic_regression` method:

    >>>    C = csh.CategoricalHyperparameter(
    >>>        "C__LogisticRegression",
    >>>        [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    >>>    )

    The `C` hyperparameter will be added as-is for the `LogisticRegression` classifier
    and the value of the `C` hyperparameter for other classifiers will be as available
    in the `shared_hyperparameters` property – iff they use the `C`
    hyperparameter of the `shared_hyperparameters` property.

    Parameters
    ----------
    config_space : cs.ConfigurationSpace
        The ConfigSpace object that defines the hyperparameters and their ranges for
        the classifiers.

    """

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
            "n_estimators": {"lower": 10, "upper": 100},
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
            "alpha__BernoulliNB", self.shared_hyperparameters["alpha"]
        )
        fit_prior = csh.CategoricalHyperparameter(
            "fit_prior__BernoulliNB", self.shared_hyperparameters["fit_prior"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "BernoulliNB")

    def setup_multinomialNB(self, classifiers: csh.CategoricalHyperparameter):
        alpha_NB = csh.CategoricalHyperparameter(
            "alpha__MultinomialNB", self.shared_hyperparameters["alpha"]
        )
        fit_prior = csh.CategoricalHyperparameter(
            "fit_prior__MultinomialNB", self.shared_hyperparameters["fit_prior"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "MultinomialNB")

    def setup_gaussianNB(self, classifiers: csh.CategoricalHyperparameter):
        # GaussianNB has no hyperparameters
        pass

    def setup_decision_tree(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__DecisionTreeClassifier",
            self.shared_hyperparameters["criterion"],
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__DecisionTreeClassifier",
            **self.shared_hyperparameters["max_depth"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__DecisionTreeClassifier",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__DecisionTreeClassifier",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "DecisionTreeClassifier"
        )

    def setup_extra_trees(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__ExtraTreesClassifier", self.shared_hyperparameters["criterion"]
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__ExtraTreesClassifier",
            **self.shared_hyperparameters["max_depth"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__ExtraTreesClassifier",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__ExtraTreesClassifier",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features__ExtraTreesClassifier",
            **self.shared_hyperparameters["max_features"],
        )
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__ExtraTreesClassifier",
            **self.shared_hyperparameters["n_estimators"],
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap__ExtraTreesClassifier", self.shared_hyperparameters["bootstrap"]
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "ExtraTreesClassifier"
        )

    def setup_random_forest(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__RandomForestClassifier",
            self.shared_hyperparameters["criterion"],
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__RandomForestClassifier",
            **self.shared_hyperparameters["max_depth"],
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
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators", **self.shared_hyperparameters["n_estimators"]
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap", self.shared_hyperparameters["bootstrap"]
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "RandomForestClassifier"
        )

    def setup_gradient_boosting(self, classifiers: csh.CategoricalHyperparameter):
        sub_sample = csh.UniformFloatHyperparameter(
            "subsample", 0.05, 1.01, default_value=1.0
        )
        learning_rate = csh.CategoricalHyperparameter(
            "learning_rate", [1e-3, 1e-2, 1e-1, 0.5, 1.0]
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features__GradientBoostingClassifier",
            **self.shared_hyperparameters["max_features"],
        )
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__GradientBoostingClassifier",
            **self.shared_hyperparameters["n_estimators"],
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__GradientBoostingClassifier",
            **self.shared_hyperparameters["max_depth"],
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
            "loss__LinearSVC", ["hinge", "squared_hinge"]
        )
        penalty = csh.CategoricalHyperparameter("penalty__LinearSVC", ["l1", "l2"])
        dual = csh.CategoricalHyperparameter(
            "dual__LinearSVC", self.shared_hyperparameters["dual"]
        )
        tol = csh.CategoricalHyperparameter(
            "tol__LinearSVC", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        )
        C = csh.CategoricalHyperparameter(
            "C__LinearSVC", self.shared_hyperparameters["C"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "LinearSVC")

        # Forbidden clause: Penalty 'l1' cannot be used with loss 'hinge'
        forbidden_penalty_loss = cs.ForbiddenAndConjunction(
            cs.ForbiddenEqualsClause(self.config_space["penalty__LinearSVC"], "l1"),
            cs.ForbiddenEqualsClause(self.config_space["loss__LinearSVC"], "hinge"),
        )
        self.config_space.add_forbidden_clause(forbidden_penalty_loss)

    def setup_logistic_regression(self, classifiers: csh.CategoricalHyperparameter):
        solver = csh.CategoricalHyperparameter("solver", ["lbfgs"])
        penalty = csh.CategoricalHyperparameter("penalty__LogisticRegression", ["l2"])
        C = csh.CategoricalHyperparameter(
            "C__LogisticRegression", self.shared_hyperparameters["C"]
        )
        dual = csh.CategoricalHyperparameter(
            "dual__LogisticRegression", self.shared_hyperparameters["dual"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "LogisticRegression")
