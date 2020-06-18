import logging
import sys

from gama.logging.MachineLogFileHandler import MachineLogFileHandler

gama_log = logging.getLogger("gama")


def register_stream_log(verbosity):
    previously_registered_handler = [
        handler for handler in gama_log.handlers if hasattr(handler, "tag")
    ]
    if len(previously_registered_handler) > 0:
        gama_log.debug(
            "Removing StreamHandlers registered by previous GAMA instance(s)."
        )
        gama_log.handlers = [
            handler
            for handler in gama_log.handlers
            if not (
                hasattr(handler, "tag") and isinstance(handler, logging.StreamHandler)
            )
        ]

    stdout_streamhandler = logging.StreamHandler(sys.stdout)
    stdout_streamhandler.tag = "machine_set"
    stdout_streamhandler.setLevel(verbosity)
    gama_log.addHandler(stdout_streamhandler)


def register_file_log(filename):
    if any(
        [isinstance(handler, MachineLogFileHandler) for handler in gama_log.handlers]
    ):
        gama_log.debug("Removing FileHandlers registered by previous GAMA instance(s).")
        gama_log.handlers = [
            handler
            for handler in gama_log.handlers
            if not isinstance(handler, MachineLogFileHandler)
        ]
    gama_log.addHandler(MachineLogFileHandler(filename))
