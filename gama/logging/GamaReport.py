import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import pandas as pd

from gama.configuration.classification import clf_config
from gama.configuration.parser import pset_from_config, merge_configurations
from gama.configuration.regression import reg_config
from gama.genetic_programming.components import Individual


pset, _ = pset_from_config(merge_configurations(clf_config, reg_config))


class GamaReport:
    """ Contains information parsed from a search captured by a GAMA analysis log. """

    def __init__(self, log_directory: str):
        """ Parse the logfile or log lines provided.

        Parameters
        ----------
        log_directory: str
            The directory with logs:
                - gama.log
                - evaluations.log
                - resources.log
        """

        with open(os.path.join(log_directory, "gama.log")) as fh:
            gama_lines = fh.readlines()

        self.name = log_directory.split("\\")[-1]
        self.phases: List[Tuple[str, str, datetime, float]] = []
        for line in gama_lines:
            if "INIT:" in line:
                self.hyperparameters = init_to_hps(line)
            if "STOP:" in line:
                time_and_place, activity = line.split(" STOP: ")
                timestamp, place = time_and_place[1:-1].split(" - ")
                end_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
                print(activity)
                phase, algorithm, _, duration = activity.split(" ")
                d = float(duration[:-3])  # e.g. "0.1300s.\n"
                start_time = end_time - timedelta(seconds=d)
                self.phases.append((phase, algorithm, start_time, d))

        self.metrics = self.hyperparameters["scoring"].split(",")
        if self.hyperparameters["regularize_length"]:
            self.metrics += ["length"]

        self.incomplete = len(self.phases) < 3

        # [x] Extract Phases
        # [x] Extract Evaluations
        self.evaluations = evaluations_to_dataframe(
            os.path.join(log_directory, "evaluations.log"), self.metrics
        )

        # This can take a while for long logs (e.g. ~1sec for 10k individuals)
        self.individuals: Dict[str, Individual] = {
            id_: Individual.from_string(pipeline, pset)
            for id_, pipeline in zip(self.evaluations.id, self.evaluations.pipeline)
        }

        # [ ] Extract advanced setup (e.g. method information)
        # [ ] Allow for updates
        # [ ] Dashboard -- how point to directory??

        # parse_method_data: Dict[str, Callable[..., pd.DataFrame]] = defaultdict(
        #     lambda: lambda *args: None,
        #     AsynchronousSuccessiveHalving=_ASHA_data_to_dataframe,
        # )
        # # search_method is formatted like NAME(kwargs)
        # # where kwargs could contain additional parentheses.
        # method_name, _ = self.search_method.split("(", maxsplit=1)
        # method_token = METHOD_TOKENS.get(method_name)
        # self.method_data = parse_method_data[method_name](
        #     events_by_type[method_token], self.metrics
        # )
        self.method_data = None

    # def update(self) -> bool:
    #     new_lines = _find_new_lines(self._filename, start_from=self._lines_read)
    #     if len(new_lines) > 0:
    #         self._lines_read += len(new_lines)
    #         print(f"read {len(new_lines)} new lines")
    #         events_by_type = _lines_to_dict(new_lines)
    #         if len(self.evaluations) == 0:
    #             search_start = None
    #         else:
    #             search_start = self.evaluations.start.min()
    #         start_n = self.evaluations.n.max()
    #         if math.isnan(start_n):
    #             start_n = -1
    #
    #         new_evaluations = _evaluations_to_dataframe(
    #             events_by_type[TOKENS.EVALUATION_RESULT],
    #             metric_names=self.metrics,
    #             search_start=search_start,
    #             start_n=start_n + 1,
    #         )
    #         self.evaluations = pd.concat([self.evaluations, new_evaluations])
    #         for metric in self.metrics:
    #             self.evaluations[f"{metric}_cummax"] = self.evaluations[
    #                 metric
    #             ].cummax()  # noqa: E501
    #         new_individuals = {
    #             id_: Individual.from_string(pipeline, pset)
    #             for id_, pipeline in zip(new_evaluations.id, new_evaluations.pipeline)
    #         }
    #         self.individuals.update(new_individuals)
    #     return len(new_lines) > 0


# new
def init_to_hps(init_line: str) -> Dict[str, str]:
    all_arguments = init_line.split("(", maxsplit=1)[-1].rsplit(")", maxsplit=1)[0]
    return dict(hp.split("=") for hp in all_arguments.split(","))  # type: ignore


# old

# def _lines_to_dict(log_lines: List[str]):
#     # Find the Parseable Log Events and discard their start/end tokens.
#     ple_lines = [
#         line.split(PLE_DELIM)[1:-1]
#         for line in log_lines
#         if line.startswith(PLE_START) and line.endswith(f"{PLE_END}")
#     ]
#
#     events_by_type: Dict[str, List[List[str]]] = defaultdict(list)
#     for token, *event in ple_lines:
#         events_by_type[token].append(event)
#
#     return events_by_type


# def _find_new_lines(logfile: str, start_from: int = 0) -> List[str]:
#     with open(logfile, "r") as fh:
#         log_lines = [line.rstrip() for line in fh.readlines()]
#     new_lines = log_lines[start_from:]
#     return new_lines


# def _find_metric_configuration(
#     init_lines: List[List[str]],
# ) -> Tuple[List[str], str, str, str]:
#     hps = init_lines[0][0].split(",")
#
#     def hyperparameter_value_of(hp_name: str) -> str:
#         return [hp.split("=", maxsplit=1)[-1] for hp in hps if hp_name in hp][0]
#
#     metric = hyperparameter_value_of("scoring")
#     regularize = hyperparameter_value_of("regularize_length")
#     search = hyperparameter_value_of("search_method")
#     filename = hyperparameter_value_of("keep_analysis_log")
#     postprocessing = hyperparameter_value_of("post_processing_method")
#     if bool(regularize):
#         return [metric, "length"], search, postprocessing, filename
#     else:
#         return [metric], search, postprocessing, filename


# def _find_phase_information(
#     events_by_type: Dict[str, List[str]]
# ) -> List[Tuple[str, str, datetime, float]]:
#     """ For each phase (e.g. search), find the type (e.g. ASHA) and its duration. """
#     phases = ["preprocessing", "search", "postprocess"]
#     phase_info = []
#     # Events as phase;algorithm;logtime
#     for phase in phases:
#         phase_starts = [e for e in events_by_type[TOKENS.PHASE_START] if phase in e]
#         phase_ends = [e for e in events_by_type[TOKENS.PHASE_END] if phase in e]
#         if phase_starts == [] or phase_ends == []:
#             # The phase has either not yet started, or not yet completed.
#             # Then this is also true for later phases.
#             break
#         start_phase, end_phase = phase_starts[0], phase_ends[0]
#         _, _, start_time = start_phase
#         start_time_dt = datetime.strptime(start_time, TIME_FORMAT)
#         _, algorithm, end_time = end_phase
#         duration = datetime.strptime(end_time, TIME_FORMAT) - start_time_dt
#         phase_info.append((phase, algorithm, start_time_dt, duration.total_seconds()))
#     return phase_info


def evaluations_to_dataframe(file: str, metric_names: List[str]):
    df = pd.read_csv(file, sep=";")
    df["n"] = df.index
    df = df.rename(columns=dict(t_start="start", t_wallclock="duration"))

    def tuple_to_metrics(tuple_str):
        return pd.Series([float(value) for value in tuple_str[1:-1].split(",")])

    df[metric_names] = df.score.apply(tuple_to_metrics)
    for metric in metric_names:
        df[f"{metric}_cummax"] = df[metric].cummax()

    df.start = pd.to_datetime(df.start)  # needed?
    df.duration = pd.to_timedelta(df.duration, unit="s")
    search_start = df.start.min()
    if len(df.start) > 0:
        df["relative_end"] = (
            (df.start + df.duration) - search_start
        ).dt.total_seconds()
    else:
        df["relative_end"] = pd.Series()
    return df


# def _evaluations_to_dataframe(
#     evaluation_lines: List[List[str]],
#     metric_names: Optional[List[str]] = None,
#     search_start: datetime = None,
#     start_n: int = 0,
# ) -> pd.DataFrame:
#     """ Create a dataframe with all parsed EVAL events in the log. """
#     evaluations = []
#     for i, line in enumerate(evaluation_lines, start=start_n):
#         time, pid, duration, process_duration, fitness, id_, pipe_str, log_time = line
#         # Fitness logged as '(metric1, metric2, ..., metriclast)'
#         metrics_values = [float(value) for value in fitness[1:-1].split(",")]
#         evaluations.append(
#             [i, time, pid, float(duration), *metrics_values, pipe_str, id_]
#         )
#
#     if metric_names is None:
#         metric_names = [f"metric_{m_i}" for m_i in range(len(metrics_values))]
#     column_names = ["n", "start", "pid", "duration", *metric_names, "pipeline", "id"]
#     df = pd.DataFrame(evaluations, columns=column_names)
#     for metric in metric_names:
#         df[f"{metric}_cummax"] = df[metric].cummax()
#
#     df.start = pd.to_datetime(df.start)
#     df.duration = pd.to_timedelta(df.duration, unit="s")
#     search_start = search_start if search_start is not None else df.start.min()
#     if len(df.start) > 0:
#         df["relative_end"] = (
#             (df.start + df.duration) - search_start
#         ).dt.total_seconds()
#     else:
#         df["relative_end"] = pd.Series()
#     return df
#
#
# def _ASHA_data_to_dataframe(
#     asha_lines: List[List[str]], metric_names: Optional[List[str]] = None
# ) -> pd.DataFrame:
#     asha_data = []
#     for i, line in enumerate(asha_lines):
#         rung, duration, fitness, id_, pipeline_str, log_time = line
#         metrics_values = [float(value) for value in fitness[1:-1].split(",")]
#         asha_data.append([i, rung, float(duration), *metrics_values, pipeline_str, id_])  # noqa: E501
#
#     if metric_names is None:
#         metric_names = [f"metric_{m_i}" for m_i in range(len(metrics_values))]
#     column_names = ["n", "rung", "duration", *metric_names, "pipeline", "id"]
#     df = pd.DataFrame(asha_data, columns=column_names)
#     df.duration = pd.to_timedelta(df.duration, unit="s")
#
#     return df
