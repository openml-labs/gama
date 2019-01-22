from typing import NamedTuple, Tuple

Fitness = NamedTuple("Fitness",
                     [("values", Tuple),
                      ("start_time", int),
                      ("wallclock_time", int),
                      ("process_time", int)])
