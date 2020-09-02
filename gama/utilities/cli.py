import argparse
import logging
import os
import pickle

from pandas.api.types import is_categorical_dtype

from gama import GamaClassifier, GamaRegressor
from gama.data_loading import X_y_from_file


def parse_args():
    desc = "An AutoML tool that optimizes machine learning pipelines for your data."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "input_file",
        type=str,
        help="A csv or ARFF file with the data to optimize a model for.",
    )

    io_group = parser.add_argument_group("File I/O")

    io_group.add_argument(
        "-sep",
        dest="separator",
        type=str,
        default=None,
        help=(
            "For CSV files: the character used to separate values in the input file."
            "If none is given, the Python parser will be used to infer it."
        ),
    )
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
        "-outdir",
        dest="outdir",
        default=None,
        type=str,
        help="Directory to store GAMA logs",
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
    if not os.path.exists(args.input_file.lower()):
        raise FileNotFoundError(args.input_file)
    if args.input_file.lower().split(".")[-1] not in ["csv", "arff"]:
        raise ValueError("Unknown file extension. Please use csv or arff.")

    kwargs = {}
    if args.input_file.lower().endswith(".csv") and args.separator is not None:
        kwargs["sep"] = args.seperator

    x, y = X_y_from_file(
        file_path=args.input_file.lower(), split_column=args.target, **kwargs,
    )
    if args.mode is None:
        if is_categorical_dtype(y.dtype):
            args.mode = "classification"
        else:
            args.mode = "regression"
        print(f"Detected a {args.mode} problem.")

    print("CLI: Initializing GAMA")
    log_level = logging.INFO if args.verbose else logging.WARNING
    configuration = dict(
        regularize_length=args.prefer_short,
        max_total_time=args.time_limit_m * 60,
        max_eval_time=args.max_eval_time_m * 60,
        n_jobs=args.n_jobs,
        verbosity=log_level,
        output_directory=args.outdir,
        store="nothing" if args.dry_run else "logs",
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
        automl.fit(x, y)

        # == Model Export ===
        print("CLI: Exporting models.")
        with open(args.output_file, "wb") as fh:
            pickle.dump(automl.model, fh)

        if args.export_python is not None:
            automl.export_script(args.export_python, raise_if_exists=False)
    else:
        automl.cleanup("all")
    print("done!")


if __name__ == "__main__":
    main()
