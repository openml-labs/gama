from typing import List, Optional, Tuple

import pandas as pd

from gama.logging.machine_logging import PLE_START, PLE_DELIM, PLE_END, TOKENS


class GamaReport:
    """ Contains information over an information trace as parsed from a GAMA analysis log. """

    def __init__(self, logfile: str):
        with open(logfile, 'r') as fh:
            loglines = fh.readlines()

        # Find the Parseable Log Events and discard their start/end tokens.
        ple_lines = [line.split(PLE_DELIM)[1:-1] for line in loglines
                     if line.startswith(PLE_START) and line.endswith(f'{PLE_END}\n')]
        events_by_type = {token: [event for event in ple_lines if token in event]
                          for token in TOKENS.values()}
        self.evaluations = _evaluations_to_dataframe(events_by_type[TOKENS.EVALUATION_RESULT])
        self.phases = _find_phase_information(loglines)


def _find_phase_information(log_lines: List[str]) -> List[Tuple[str, str, float]]:
    """ For each phase (e.g. search), find the type used (e.g. ASHA) and its duration. """
    phase_information = []
    phases = ['preprocessing', 'search', 'postprocessing']
    durations = []
    for phase in phases:
        # duration_line is formatted as "{phase} took {float}s."
        duration_line = [line for line in log_lines if line.startswith(f'{phase} took ')][0]
        duration = float(duration_line[:-2].split(' ')[-1])
        durations.append(duration)
    return phase_information


def _evaluations_to_dataframe(evaluation_lines: List[List[str]],
                              metric_names: Optional[List[str]] = None) -> pd.DataFrame:
    """ Create a dataframe with all pipeline evaluations as parsed from EVAL events in the log. """
    evaluations = []
    for i, line in enumerate(evaluation_lines):
        token, time, duration, process_duration, fitness, id_, pipeline_str, log_time = line
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

###
# One log parser:
# parse evaluations to dataframe with default preprocessing:
# - time per phase, info per phase (e.g. search strategy)
#
# on demand:
# - number of concurrent evaluations over time
# - something about learners like the grid of preprocessing x learner => performance
# - error report
# - hierachy/lineage graph
#
# Aggregate reports
#
##
