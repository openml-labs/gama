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

        self.evaluations = evaluations_to_dataframe(
            os.path.join(log_directory, "evaluations.log"), self.metrics
        )

        # This can take a while for long logs (e.g. ~1sec for 10k individuals)
        self.individuals: Dict[str, Individual] = {
            id_: Individual.from_string(pipeline, pset)
            for id_, pipeline in zip(self.evaluations.id, self.evaluations.pipeline)
        }

        # [ ] Allow for updates
        # [ ] Dashboard -- how point to directory??

        self.search_method = self.hyperparameters["search_method"]
        self.method_data = self.evaluations

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
    # A little hackery to get nested configurations (e.g. of search) to work
    # only supports one nested level - will do proper parsing later
    for token in ["()", "(", ")", ",,"]:
        all_arguments = all_arguments.replace(token, ",")
    print(all_arguments)
    return dict(hp.split("=") for hp in all_arguments.split(","))  # type: ignore


# def _find_new_lines(logfile: str, start_from: int = 0) -> List[str]:
#     with open(logfile, "r") as fh:
#         log_lines = [line.rstrip() for line in fh.readlines()]
#     new_lines = log_lines[start_from:]
#     return new_lines


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
