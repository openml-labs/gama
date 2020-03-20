from datetime import datetime
from typing import NamedTuple, Tuple


class Fitness(NamedTuple):
    values: Tuple
    start_time: datetime
    wallclock_time: int
    process_time: int
