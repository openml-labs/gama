import logging
from gama.logging import MACHINE_LOG_LEVEL


class MachineLogFileHandler(logging.FileHandler):
    def __init__(self, filename, log_levels=None, mode='a', encoding=None, delay=False):
        """ As logging.FileHandler except only logs messages from `log_levels` to file.
        Used to log debug and error but not info.
        """
        super().__init__(filename, mode, encoding, delay)
        if log_levels is None:
            self.allowed_levels = [MACHINE_LOG_LEVEL, logging.WARNING, logging.ERROR, logging.CRITICAL]
        else:
            self.allowed_levels = log_levels
        super().setLevel(MACHINE_LOG_LEVEL)

    def emit(self, record):
        if record.levelno not in self.allowed_levels:
            return
        super().emit(record)
