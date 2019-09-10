import itertools
import logging
import multiprocessing as mp
import queue
import sys

gama_log = logging.getLogger('gama')


def register_stream_log(verbosity):
    previously_registered_handler = [handler for handler in gama_log.handlers if hasattr(handler, 'tag')]
    if len(previously_registered_handler) > 0:
        gama_log.debug("Removing StreamHandlers registered by previous GAMA instance(s).")
        gama_log.handlers = [handler for handler in gama_log.handlers
                             if not (hasattr(handler, 'tag') and isinstance(handler, logging.StreamHandler))]

    stdout_streamhandler = logging.StreamHandler(sys.stdout)
    stdout_streamhandler.tag = 'machine_set'
    stdout_streamhandler.setLevel(verbosity)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_streamhandler.setFormatter(formatter)
    
    gama_log.addHandler(stdout_streamhandler)


def register_file_log(filename):
    if any([isinstance(handler, logging.FileHandler) and hasattr(handler, 'tag') for handler in gama_log.handlers]):
        gama_log.debug("Removing FileHandlers registered by previous GAMA instance(s).")
        gama_log.handlers = [handler for handler in gama_log.handlers
                             if not (hasattr(handler, 'tag') and isinstance(handler, logging.FileHandler))]

    file_handler = logging.FileHandler(filename)
    file_handler.tag = 'machine_set'
    file_handler.setLevel(5)
    gama_log.addHandler(file_handler)


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
