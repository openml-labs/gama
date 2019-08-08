from datetime import datetime
from typing import List, Optional, Tuple, Dict

import pandas as pd

from gama.logging.machine_logging import PLE_START, PLE_DELIM, PLE_END, TOKENS
from gama.logging import TIME_FORMAT


class GamaReport:
    """ Contains information over an information trace as parsed from a GAMA analysis log. """

    def __init__(self, logfile: Optional[str] = None, log_lines: Optional[List[str]] = None):
        """ Parses the logfile or log lines provided. Must provide exactly one of 'logfile' or 'loglines'.

        :param logfile: Optional[str] (default=None)
            Path to the log file. If not specified, loglines must be provided.
        :param log_lines: Optional[List[str]] (default=None)
            A list with each element one line from the log file. If not specified, logfile must be provided.
        """
        if logfile is None and log_lines is None:
            raise ValueError("Either 'logfile' or 'loglines' must be provided. Both are None.")
        if logfile is not None and log_lines is not None:
            raise ValueError("Exactly one of 'logfile' and 'loglines' may be provided at once.")

        if logfile is not None:
            with open(logfile, 'r') as fh:
                log_lines = [line.rstrip() for line in fh.readlines()]

        self.metrics = _find_metric_configuration(log_lines)

        # Find the Parseable Log Events and discard their start/end tokens.
        ple_lines = [line.split(PLE_DELIM)[1:-1] for line in log_lines
                     if line.startswith(PLE_START) and line.endswith(f'{PLE_END}')]
        events_by_type = {token: [event for line_token, *event in ple_lines if token == line_token]
                          for token in TOKENS.values()}
        self.evaluations: pd.DataFrame = _evaluations_to_dataframe(events_by_type[TOKENS.EVALUATION_RESULT],
                                                                   metric_names=self.metrics)
        self.phases: List[Tuple[str, str, float]] = _find_phase_information(events_by_type)


def _find_metric_configuration(log_lines: List[str]) -> List[str]:
    # Can line logging init call searching for e.g. 'GamaClassifier(' but right now the location is static anyway.
    init_line = log_lines[1]
    # E.g.: GamaRegressor(scoring=neg_mean_squared_error,regularize_length=True, ...)
    _, arguments = init_line.split('(')
    scoring, regularize_length, *_ = arguments.split(',')
    _, metric = scoring.split('=')
    _, regularize = regularize_length.split('=')
    if bool(regularize):
        return [metric, 'length']
    else:
        return [metric]


def _find_phase_information(events_by_type: Dict[str, List[str]]) -> List[Tuple[str, str, float]]:
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
        phase_info.append([phase, algorithm, duration.total_seconds()])
    return phase_info


def _evaluations_to_dataframe(evaluation_lines: List[List[str]],
                              metric_names: Optional[List[str]] = None) -> pd.DataFrame:
    """ Create a dataframe with all pipeline evaluations as parsed from EVAL events in the log. """
    evaluations = []
    for i, line in enumerate(evaluation_lines):
        time, duration, process_duration, fitness, id_, pipeline_str, log_time = line
        # Fitness logged as '(metric1, metric2, ..., metriclast)'
        metrics_values = [float(value) for value in fitness[1:-1].split(',')]
        evaluations.append([i, time, duration, *metrics_values])

    if metric_names is None:
        metric_names = [f'metric_{m_i}' for m_i in range(len(metrics_values))]
    column_names = ['n', 'start', 'duration', *metric_names]
    df = pd.DataFrame(evaluations, columns=column_names)
    for metric in metric_names:
        df[f'{metric}_cummax'] = df[metric].cummax()

    return df
