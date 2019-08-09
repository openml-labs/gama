""" Module to facility logging in machine parsable format. """
from datetime import datetime
from typing import List

from gama.logging import TIME_FORMAT, MACHINE_LOG_LEVEL

PLE_START = 'PLE'
PLE_END = 'END!'
PLE_DELIM = ';'


class TOKENS:
    EVALUATION_RESULT = 'EVAL'
    EVALUATION_ERROR = 'EVAL_ERR'
    PHASE_START = 'PS'
    PHASE_END = 'PE'
    EA_RESTART = 'EA_RST'
    EA_REMOVE_IND = 'RMV_IND'
    EVALUATION_TIMEOUT = 'EVAL_TO'
    MUTATION = 'IND_MUT'
    CROSSOVER = "IND_CX"

    @classmethod
    def values(cls) -> List[str]:
        """ Return a list of each possible TOKENS value. """
        # skips default dunder attributes like __module__
        return [getattr(cls, attribute) for attribute in vars(cls)
                if not (attribute.startswith('__') and attribute.endswith('__'))]


def default_time_format(datetime_: datetime):
    return datetime_.strftime(TIME_FORMAT)#[:-3]


def log_event(log_, token: str, *args):
    """ Writes the described event to the machine log level formatted for later parsing. """
    args = [default_time_format(arg) if isinstance(arg, datetime) else arg for arg in args]
    attrs = f'{PLE_DELIM}'.join([str(arg) for arg in args])
    message = f'{PLE_DELIM}'.join([PLE_START, token, attrs, default_time_format(datetime.now()), PLE_END])
    log_.log(level=MACHINE_LOG_LEVEL, msg=message)


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
