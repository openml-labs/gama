import argparse
import logging
import os
import pickle

from gama import GamaClassifier, GamaRegressor
from gama.data import load_feature_metadata_from_arff


def parse_args():
    desc = "An AutoML tool that optimizes machine learning pipelines for your data."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "input_file",
        type=str,
        help="An ARFF file with the data to optimize a model for.",
    )

    io_group = parser.add_argument_group("File I/O")

    # io_group.add_argument(
    #     '-sep',
    #     dest='separator',
    #     type=str,
    #     default=',',
    #     help=("For CSV files only, the character used
    #     to separate columns in the input file. (default=',')")
    # )
    io_group.add_argument(
        "--target",
        dest="target",
        type=str,
        default=None,
        help="The target column in the input file.",
    )
    io_group.add_argument(
        "-o",
        dest="output_file",
        type=str,
        default="gama_model.pkl",
        help="Path to store the final model in. (default=gama_model.pkl)",
    )

    io_group.add_argument(
        "-py",
        dest="export_python",
        type=str,
        default=None,
        help="If set, export python code for the final model to this destination.",
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default=None,
        help="By default, it is determined automatically from the input file."
        "Use this parameter force 'classification' or 'regression' mode. ",
    )
    optimization.add_argument(
        "-m",
        dest="metric",
        type=str,
        default=None,
        help="The metric to optimize the model for. "
        "Default is log loss for classification and RMSE for regression.",
    )
    optimization.add_argument(
        "--long",
        dest="prefer_short",
        action="store_false",
        help="By default GAMA will guide search towards shorter pipelines. "
        "Set this flag to disable the feature.",
    )

    resources = parser.add_argument_group("Resources")
    resources.add_argument(
        "-t",
        dest="time_limit_m",
        type=int,
        default=60,
        help="The maximum time (in minutes) GAMA has for optimization. (default=60)",
    )
    resources.add_argument(
        "--time_pipeline",
        dest="max_eval_time_m",
        type=int,
        default=5,
        help="The maximum time in minutes GAMA may spend "
        "on a single pipeline evaluation. (default=5)",
    )
    resources.add_argument(
        "-n",
        dest="n_jobs",
        type=int,
        default=1,
        help="Jobs to run in parallel for pipeline evaluations. (default=1)",
    )

    # Extra
    parser.add_argument(
        "-log", dest="logpath", default=None, type=str, help="File to store GAMA's log."
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="Report status updates to console.",
    )
    parser.add_argument(
        "-dry",
        dest="dry_run",
        action="store_true",
        help="If True, execute without calling fit or exports.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("CLI: Processing input")
    if args.input_file.lower().endswith(".csv"):
        raise NotImplementedError("CSV currently not supported.")
        # data = pd.read_csv(args.input_file, sep=args.separator)
    elif not os.path.exists(args.input_file.lower()):
        raise FileNotFoundError(args.input_file)

    if args.input_file.lower().endswith(".arff") and args.mode is None:
        # Determine the task type based on the target column in the arff file
        attributes = load_feature_metadata_from_arff(args.input_file)
        target = list(attributes)[-1] if args.target is None else args.target
        target_type = attributes[target]
        if "{" in target_type:
            # Nominal features are denoted by listen all their values, eg.
            # {VALUE_1, VALUE_2, ...}
            args.mode = "classification"
        elif target_type.lower() == "real":
            args.mode = "regression"
        else:
            raise ValueError(
                f"Target column {target} has type {target_type}, which GAMA can't model"
            )
        print(f"Detected a {args.mode} problem.")

    print("CLI: Initializing GAMA")
    log_level = logging.INFO if args.verbose else logging.WARNING
    configuration = dict(
        regularize_length=args.prefer_short,
        max_total_time=args.time_limit_m * 60,
        max_eval_time=args.max_eval_time_m * 60,
        n_jobs=args.n_jobs,
        verbosity=log_level,
        keep_analysis_log=args.logpath,
    )
    if args.metric:
        configuration["scoring"] = args.metric

    if args.mode == "regression":
        automl = GamaRegressor(**configuration)
    elif args.mode == "classification":
        automl = GamaClassifier(**configuration)
    else:
        raise ValueError(f"Mode {args.mode} is not valid (--mode).")

    if not args.dry_run:
        print("CLI: Starting model search")
        if args.input_file.lower().endswith(".arff"):
            automl.fit_arff(args.input_file.lower(), target_column=args.target)
        # else:
        #    automl.fit(x, y)

        # == Model Export ===
        print("CLI: Exporting models.")
        with open(args.output_file, "wb") as fh:
            pickle.dump(automl.model, fh)

        if args.export_python is not None:
            automl.export_script(args.export_python, raise_if_exists=False)
    print("done!")


if __name__ == "__main__":
    main()
