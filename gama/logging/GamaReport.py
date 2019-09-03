from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Callable

import pandas as pd

from gama.configuration.classification import clf_config
from gama.configuration.parser import pset_from_config, merge_configurations
from gama.configuration.regression import reg_config
from gama.genetic_programming.components import Individual
from gama.logging.machine_logging import PLE_START, PLE_DELIM, PLE_END, TOKENS, METHOD_TOKENS
from gama.logging import TIME_FORMAT


pset, _ = pset_from_config(merge_configurations(clf_config, reg_config))


class GamaReport:
    """ Contains information over an information trace as parsed from a GAMA analysis log. """

    def __init__(
            self,
            logfile: Optional[str] = None,
            log_lines: Optional[List[str]] = None,
            name: Optional[str] = None
    ):
        """ Parses the logfile or log lines provided. Must provide exactly one of 'logfile' or 'loglines'.

        Parameters
        ----------
        logfile: str, optional (default=None)
            Path to the log file. If not specified, loglines must be provided.
        log_lines: List[str], optional (default=None)
            A list with each element one line from the log file. If not specified, logfile must be provided.
        name: str, optional (default=None)
            Name of the report. If set to None, defaults to `logfile` if it is not None else 'nameless'.
        """
        if logfile is None and log_lines is None:
            raise ValueError("Either 'logfile' or 'loglines' must be provided. Both are None.")
        if logfile is not None and log_lines is not None:
            raise ValueError("Exactly one of 'logfile' and 'loglines' may be provided at once.")

        if logfile is not None:
            with open(logfile, 'r') as fh:
                log_lines = [line.rstrip() for line in fh.readlines()]

        self._individuals = None
        self.name = name if name is not None else (logfile if logfile is not None else 'nameless')

        # Find the Parseable Log Events and discard their start/end tokens.
        ple_lines = [line.split(PLE_DELIM)[1:-1] for line in log_lines
                     if line.startswith(PLE_START) and line.endswith(f'{PLE_END}')]

        events_by_type = defaultdict(list)
        for token, *event in ple_lines:
            events_by_type[token].append(event)

        self.metrics = _find_metric_configuration(events_by_type[TOKENS.INIT])
        self.phases: List[Tuple[str, str, datetime, float]] = _find_phase_information(events_by_type)
        self.evaluations: pd.DataFrame = _evaluations_to_dataframe(events_by_type[TOKENS.EVALUATION_RESULT],
                                                                   metric_names=self.metrics,
                                                                   search_start=self.phases[1][2])

        # This can take a while for long logs (e.g. ~1sec for 10k individuals)
        self.individuals: Dict[str, Individual] = {
            id_: Individual.from_string(pipeline, pset)
            for id_, pipeline in zip(self.evaluations.id, self.evaluations.pipeline)
        }

        parse_method_data: Dict[str, Callable[..., pd.DataFrame]] = defaultdict(
            lambda: lambda *args: None,
            AsynchronousSuccessiveHalving=_ASHA_data_to_dataframe
        )
        method_token = METHOD_TOKENS.get(self.search_method)
        self.method_data = parse_method_data[self.search_method](events_by_type[method_token], self.metrics)

    @property
    def search_method(self):
        return self.phases[1][1]


def _find_metric_configuration(init_lines: List[List[str]]) -> List[str]:
    scoring, regularize_length, *_ = init_lines[0][0].split(',')
    _, metric = scoring.split('=')
    _, regularize = regularize_length.split('=')
    if bool(regularize):
        return [metric, 'length']
    else:
        return [metric]


def _find_phase_information(events_by_type: Dict[str, List[str]]) -> List[Tuple[str, str, datetime, float]]:
    """ For each phase (e.g. search), find the type used (e.g. ASHA) and its duration. """
    phases = ['preprocessing', 'search', 'postprocess']
    phase_info = []
    # Events as phase;algorithm;logtime
    for phase in phases:
        start_phase = [event for event in events_by_type[TOKENS.PHASE_START] if phase in event][0]
        end_phase = [event for event in events_by_type[TOKENS.PHASE_END] if phase in event][0]
        _, _, start_time = start_phase
        _, algorithm, end_time = end_phase
        duration = (datetime.strptime(end_time, TIME_FORMAT) - datetime.strptime(start_time, TIME_FORMAT))
        phase_info.append([phase, algorithm, datetime.strptime(start_time, TIME_FORMAT), duration.total_seconds()])
    return phase_info


def _evaluations_to_dataframe(evaluation_lines: List[List[str]],
                              metric_names: Optional[List[str]] = None,
                              search_start: datetime = None) -> pd.DataFrame:
    """ Create a dataframe with all pipeline evaluations as parsed from EVAL events in the log. """
    evaluations = []
    for i, line in enumerate(evaluation_lines):
        time, duration, process_duration, fitness, id_, pipeline_str, log_time = line
        # Fitness logged as '(metric1, metric2, ..., metriclast)'
        metrics_values = [float(value) for value in fitness[1:-1].split(',')]
        evaluations.append([i, time, float(duration), *metrics_values, pipeline_str, id_])

    if metric_names is None:
        metric_names = [f'metric_{m_i}' for m_i in range(len(metrics_values))]
    column_names = ['n', 'start', 'duration', *metric_names, 'pipeline', 'id']
    df = pd.DataFrame(evaluations, columns=column_names)
    for metric in metric_names:
        df[f'{metric}_cummax'] = df[metric].cummax()

    df.start = pd.to_datetime(df.start)
    df.duration = pd.to_timedelta(df.duration, unit='s')
    search_start = search_start if search_start is not None else df.start.min()
    df['relative_end'] = ((df.start + df.duration) - search_start).dt.total_seconds()
    return df


def _ASHA_data_to_dataframe(asha_lines: List[List[str]],
                            metric_names: Optional[List[str]] = None) -> pd.DataFrame:
    asha_data = []
    for i, line in enumerate(asha_lines):
        rung, duration, fitness, id_, pipeline_str, log_time = line
        metrics_values = [float(value) for value in fitness[1:-1].split(',')]
        asha_data.append([i, rung, float(duration), *metrics_values, pipeline_str, id_])

    if metric_names is None:
        metric_names = [f'metric_{m_i}' for m_i in range(len(metrics_values))]
    column_names = ['n', 'rung', 'duration', *metric_names, 'pipeline', 'id']
    df = pd.DataFrame(asha_data, columns=column_names)
    df.duration = pd.to_timedelta(df.duration, unit='s')

    return df
