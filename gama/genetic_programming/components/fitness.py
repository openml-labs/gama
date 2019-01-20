from typing import NamedTuple, Tuple

Fitness = NamedTuple("Fitness",
                     [("values", Tuple),
                      ("start_time", int),
                      ("time", int)])
