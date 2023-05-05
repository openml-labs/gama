import pytest

from gama.genetic_programming.components import Terminal
from gama.genetic_programming.components.terminal import find_terminal


@pytest.mark.parametrize(
    "string, output, value",
    [
        ("alpha=0.1", "alpha", 1e-1),
        ("alpha=0.01", "alpha", 1e-2),
        ("DecisionTreeClassifier.criterion='gini'", "criterion", "gini"),
    ],
)
def test_find_terminal(pset, string, output, value):
    terminal = find_terminal(pset, string)
    assert isinstance(terminal, Terminal)
    assert terminal.value == value
    assert terminal.output == output


@pytest.mark.parametrize(
    "string",
    ["alpha=0.5", "DecisionTreeClassifier.criterion='nonsense'"],
)
def test_find_terminal_outside_pset(pset, string):
    with pytest.raises(KeyError):
        find_terminal(pset, string)
