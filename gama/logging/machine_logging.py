""" Module to facility logging in machine parsable format. """
from datetime import datetime
from typing import List
import logging
from gama.logging import TIME_FORMAT, MACHINE_LOG_LEVEL

PLE_START = "PLE"
PLE_END = "END!"
PLE_DELIM = ";"


class TOKENS:
    EVALUATION_RESULT = "EVAL"
    EVALUATION_ERROR = "EVAL_ERR"
    PHASE_START = "PS"
    PHASE_END = "PE"
    EA_RESTART = "EA_RST"
    EA_REMOVE_IND = "RMV_IND"
    EVALUATION_TIMEOUT = "EVAL_TO"
    MUTATION = "IND_MUT"
    CROSSOVER = "IND_CX"
    INIT = "INIT"

    @classmethod
    def values(cls) -> List[str]:
        """ Return a list of each possible TOKENS value. """
        # skips default dunder attributes like __module__
        return [
            getattr(cls, attribute)
            for attribute in vars(cls)
            if not (attribute.startswith("__") and attribute.endswith("__"))
        ]


METHOD_TOKENS = dict(AsynchronousSuccessiveHalving="ASHA")


def datetime_to_str(datetime_: datetime):
    return datetime_.strftime(TIME_FORMAT)  # [:-3]


def log_event(log_: logging.Logger, token: str, *args: object):
    """ Write an event to the machine log level formatted for later parsing. """
    formatted_args = [
        datetime_to_str(arg) if isinstance(arg, datetime) else str(arg) for arg in args
    ]
    attrs = f"{PLE_DELIM}".join(formatted_args)
    message = f"{PLE_DELIM}".join(
        [PLE_START, token, attrs, datetime_to_str(datetime.now()), PLE_END]
    )
    log_.log(level=MACHINE_LOG_LEVEL, msg=message)
