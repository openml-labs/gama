from datetime import datetime
from functools import partial
import operator
from typing import Any, Optional, Dict, Callable, Iterable

from gama.logging import TIME_FORMAT
from gama.utilities.evaluation_library import Evaluation


def _nested_getattr(o: object, attr: str) -> Any:
    """Resolved nested properties, e.g., `individual.fitness.score`."""
    for a in attr.split("."):
        o = getattr(o, a)
    return o


class EvaluationLogger:
    def __init__(
        self,
        file_path: str,
        separator: str = ";",
        fields: Optional[Dict[str, Callable[[Evaluation], str]]] = None,
        extra_fields: Optional[Dict[str, Callable[[Evaluation], str]]] = None,
    ):
        """Formats evaluations for output to a csv file.

        Parameters
        ----------
        file_path: str
            The log file to write to.
        separator: str (default=';')
            The delimiter for the csv file.
            Note that the default `fields` results in ',' is cell values.
        fields: Dict[str, Callable[[Evaluation], str]], optional (default=None)
            Mapping of column names to a function which extracts the corresponding
            value from an evaluation.
            If None, a default set of columns is used.
        extra_fields: Dict[str, Callable[[Evaluation], str]], optional (default=None)
            Additional fields to log. Useful if you want to keep the default `fields`,
            but need additional information.
        """
        self._file_path = file_path
        self._sep = separator

        if fields is None:
            self.fields: Dict[str, Callable[[Evaluation], str]] = dict(
                id=partial(_nested_getattr, attr="individual._id"),
                pid=operator.attrgetter("pid"),
                t_start=partial(_nested_getattr, attr="individual.fitness.start_time"),
                t_wallclock=partial(
                    _nested_getattr, attr="individual.fitness.wallclock_time"
                ),
                t_process=partial(
                    _nested_getattr, attr="individual.fitness.process_time"
                ),
                score=partial(_nested_getattr, attr="individual.fitness.values"),
                pipeline=lambda e: e.individual.pipeline_str(),
                error=operator.attrgetter("error"),
            )
        else:
            self.fields = fields

        if extra_fields is not None:
            self.fields.update(extra_fields)

        self.log_line(list(self.fields))

    def log_line(self, values: Iterable[str]) -> None:
        """Appends `values` as a row of separated values to the file."""
        with open(self._file_path, "a") as evaluations:
            evaluations.write(self._sep.join(values) + "\n")

    def log_evaluation(self, evaluation) -> None:
        values = [getter(evaluation) for getter in self.fields.values()]

        def format_value(v):
            if isinstance(v, datetime):
                return v.strftime(TIME_FORMAT)
            return str(v)

        self.log_line(map(format_value, values))
