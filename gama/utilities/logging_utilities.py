import itertools
import logging
import multiprocessing as mp
import queue
from datetime import datetime


class MultiprocessingLogger(object):
    """ Stores log messages to be written to a log later. This is helpful when communicating log messages from
    auxiliary processes to the main process.
    """

    def __init__(self):
        manager = mp.Manager()
        self._queue = manager.Queue()

    def info(self, msg):
        self._queue.put((logging.INFO, msg))

    def debug(self, msg):
        self._queue.put((logging.DEBUG, msg))

    def warning(self, msg):
        self._queue.put((logging.WARNING, msg))

    def error(self, msg):
        self._queue.put((logging.WARNING, msg))

    def flush_to_log(self, log):
        # According to the official documentation, Queue.Empty is not reliable, so we just poll the queue until empty.
        for i in itertools.count():
            try:
                level, message = self._queue.get(block=False)
                log.log(level, message)
            except queue.Empty:
                break

        if i > 0:
            log.debug("Flushed {} log messages.".format(i))


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
    log_.debug(start+attrs+end)


TIME_FORMAT = '%Y-%m-%d %H:%M:%S,%f'


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
