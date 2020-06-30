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
        self._log_directory = os.path.expanduser(log_directory)
        self.name = os.path.split(log_directory)[-1]
        self.phases: List[Tuple[str, str, datetime, float]] = []
        self._last_tell = 0
        self.evaluations: pd.DataFrame = pd.DataFrame()
        self.individuals: Dict[str, Individual] = dict()

        # Parse initialization/progress information from gama.log
        with open(os.path.join(log_directory, "gama.log")) as fh:
            gama_lines = fh.readlines()

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
        self.update(force=True)  # updates self.evaluations and self.method_data
        # [ ] Dashboard -- how point to directory?? => #97

        self.search_method = self.hyperparameters["search"]

    def update(self, force: bool = False) -> bool:
        if not force and not self.incomplete:
            return False

        with open(os.path.join(self._log_directory, "evaluations.log"), "r") as fh:
            header = fh.readline()[:-1]
            self._last_tell = max(self._last_tell, fh.tell())
            fh.seek(self._last_tell)
            try:
                df = pd.read_csv(fh, sep=";", header=None, index_col=False)
            except pd.errors.EmptyDataError:
                return False
            self._last_tell = fh.tell()

            df.columns = header.split(";")
            df["n"] = df.index
            df = df.rename(columns=dict(t_start="start", t_wallclock="duration"))

            def tuple_to_metrics(tuple_str):
                return pd.Series([float(value) for value in tuple_str[1:-1].split(",")])

            df[self.metrics] = df.score.apply(tuple_to_metrics)
            df.start = pd.to_datetime(df.start)  # needed?
            df.duration = pd.to_timedelta(df.duration, unit="s")

            new_individuals = {
                id_: Individual.from_string(pipeline, pset)
                for id_, pipeline in zip(df.id, df.pipeline)
            }

            # Merge with previous records
            self.individuals.update(new_individuals)
            if self.evaluations.empty:
                self.evaluations = df
            else:
                df["n"] += self.evaluations.n.max() + 1
                self.evaluations = pd.concat([self.evaluations, df])
            df = self.evaluations

            search_start = df.start.min()
            for metric in self.metrics:
                df[f"{metric}_cummax"] = df[metric].cummax()
            if len(df.start) > 0:
                df["relative_end"] = (
                    (df.start + df.duration) - search_start
                ).dt.total_seconds()
            else:
                df["relative_end"] = pd.Series()
        return True

    @property
    def successful_evaluations(self):
        """ Return only evaluations that completed successfully """
        with pd.option_context("mode.use_inf_as_na", True):
            return self.evaluations[~self.evaluations[self.metrics].isna().any(axis=1)]


# new
def init_to_hps(init_line: str) -> Dict[str, str]:
    all_arguments = init_line.split("(", maxsplit=1)[-1].rsplit(")", maxsplit=1)[0]
    # A little hackery to get nested configurations (e.g. of search) to work
    # only supports one nested level - will do proper parsing later
    for token in ["()", "(", ")", ",,"]:
        all_arguments = all_arguments.replace(token, ",")
    return dict(hp.split("=") for hp in all_arguments.split(","))  # type: ignore
