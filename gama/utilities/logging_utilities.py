import itertools
import logging
import multiprocessing as mp
import queue
import sys
from datetime import datetime

MACHINE_LOG_LEVEL = 5
TIME_FORMAT = '%Y-%m-%d %H:%M:%S,%f'


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


gama_log = logging.getLogger('gama')
gama_log.setLevel(MACHINE_LOG_LEVEL)  # We also produce log messages below DEBUG level (machine parseable).


def register_stream_log(verbosity):
    previously_registered_handler = [handler for handler in gama_log.handlers if hasattr(handler, 'tag')]
    if len(previously_registered_handler) > 0:
        gama_log.debug("Removing handlers registered by previous GAMA instances.")
        gama_log.handlers = [handler for handler in gama_log.handlers if not hasattr(handler, 'tag')]

    stdout_streamhandler = logging.StreamHandler(sys.stdout)
    stdout_streamhandler.tag = 'machine_set'
    stdout_streamhandler.setLevel(verbosity)
    gama_log.addHandler(stdout_streamhandler)


def register_file_log(filename):
    if not any([handler for handler in gama_log.handlers
                if isinstance(handler, logging.FileHandler) and filename in handler.baseFilename]):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(5)
        gama_log.addHandler(file_handler)


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

    def log(self, level, msg):
        self._queue.put((level, msg))

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
