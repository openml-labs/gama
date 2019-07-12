""" Module to facility logging in machine parsable format. """
from datetime import datetime
from typing import Iterable, List

from gama.logging import TIME_FORMAT, MACHINE_LOG_LEVEL

PLE_START = 'PLE'
PLE_END = 'END!'
PLE_DELIM = ';'


class TOKENS:
    EVALUATION_RESULT = 'EVAL'
    EVALUATION_ERROR = 'EVAL_ERR'
    SEARCH_END = 'S_END'
    PREPROCESSING_END = 'PRE_END'
    POSTPROCESSING_END = 'POST_END'
    EA_RESTART = 'EA_RST'
    EA_REMOVE_IND = 'RMV_IND'
    EVALUATION_TIMEOUT = 'EVAL_TO'
    MUTATION = 'IND_MUT'
    CROSSOVER = "IND_CX"


def default_time_format(datetime_: datetime):
    return datetime_.strftime(TIME_FORMAT)#[:-3]


def log_event(log_, token: str, *args):
    """ Writes the described event to the machine log level formatted for later parsing. """
    args = [default_time_format(arg) if isinstance(arg, datetime) else arg for arg in args]
    attrs = f'{PLE_DELIM}'.join([str(arg) for arg in args])
    message = f'{PLE_DELIM}'.join([PLE_START, token, attrs, default_time_format(datetime.now()), PLE_END])
    log_.log(level=MACHINE_LOG_LEVEL, msg=message)


def parse_optimization(lines: List[str]) -> 'OptimizationReport':
    """ Parse all parsable log events for one gama instance. """
    # Only keep 'parsable log event' lines, discard their delimiters.
    log = [line.split(PLE_DELIM)[1:-1] for line in lines
           if line.startswith(PLE_START) and line.endswith(f"{PLE_END}\n")]

    def log_for_token(token_to_match):
        token_lines = []
        try:
            for line in log:
                token, *args, time = line
                if token == token_to_match:
                    token_lines.append((*args, time))
        except ValueError:
            raise Exception(str(line))
        return token_lines

    evaluation_lines = log_for_token(TOKENS.EVALUATION_RESULT)
    # Fitness represented in Tuples  (!Not necessarily - based on metrics)
    scores = [float(info[3].split(',')[0][1:]) for info in evaluation_lines]
    return OptimizationReport(scores)


def parse_log(logfile: str) -> Iterable['OptimizationReport']:
    """ Parse all optimization traces for one gama log. """
    with open(logfile, 'r') as fh:
        log = fh.readlines()

    start_indices = [i for i, line in enumerate(log) if line.startswith('Using GAMA version')] + [-1]
    # One log can store multiple optimization traces. Most easily separated by messages logged on initialization.
    for start_this, start_next in zip(start_indices, start_indices[1:]):
        yield parse_optimization(log[start_this:start_next])


class OptimizationReport:
    def __init__(self, scores):
        self.evaluations = scores

    @property
    def max_over_iterations(self):
        best_so_far = float('-inf')
        for score in self.evaluations:
            if score > best_so_far:
                best_so_far = score
            yield best_so_far


class Event:

    def __init__(self, start, end, token):
        self.start_time = start
        self.end_time = end
        self.token = token

    @property
    def duration(self):
        return (self.end_time - self.start_time).total_seconds()

    @classmethod
    def from_string(cls, string):
        token, *args = string.split('PLE;')[1].split(';END!')[0].split(';')
        if 'EVAL' in token:
            start, end, pl, *args = args

        datetime.strptime(start, TIME_FORMAT)
        return cls(token, start, end)
