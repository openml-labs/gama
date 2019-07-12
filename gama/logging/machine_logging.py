""" Module to facility logging in machine parsable format. """
from datetime import datetime

from gama.logging import TIME_FORMAT, MACHINE_LOG_LEVEL


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


def default_time_format(datetime_):
    return datetime_.strftime(TIME_FORMAT)#[:-3]


def log_parseable_event(log_, token, *args):
    args = [default_time_format(arg) if isinstance(arg, datetime) else arg for arg in args]
    start = "PLE;{};".format(token)
    attrs = ';'.join([str(arg) for arg in args])
    end = ';' + default_time_format(datetime.now()) + ';END!'
    log_.log(level=MACHINE_LOG_LEVEL, msg=start+attrs+end)


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
