from gama.utilities.generic.paretofront import ParetoFront


def test_pareto_initialization_empty():
    """ Test initialization of empty front. """
    pf = ParetoFront()

    assert list(pf) == []
    assert str(pf) == "[]"
    assert repr(pf) == "ParetoFront([])"


def test_pareto_initialization_pareto_front():
    """ Initialization with only Pareto front elements. """
    list_ = [(1, 2, 3), (3, 2, 1), (0, 5, 0)]
    pf = ParetoFront(list_)

    assert list(pf) == [(1, 2, 3), (3, 2, 1), (0, 5, 0)]
    assert str(pf) == "[(1, 2, 3), (3, 2, 1), (0, 5, 0)]"
    assert repr(pf) == "ParetoFront([(1, 2, 3), (3, 2, 1), (0, 5, 0)])"


def test_pareto_initialization_with_inferiors():
    """" Initialization containing elements that should not be in the Pareto front. """
    list_ = [(1, 2), (4, 3), (4, 5), (5, 4)]
    pf = ParetoFront(list_)

    assert list(pf) == [(4, 5), (5, 4)]
    assert str(pf) == "[(4, 5), (5, 4)]"
    assert repr(pf) == "ParetoFront([(4, 5), (5, 4)])"


def test_pareto_initialization_with_duplicates():
    """ Initialization with duplicate elements. """
    list_ = [(1, 2), (3, 1), (1, 2)]
    pf = ParetoFront(list_)

    assert list(pf) == [(1, 2), (3, 1)]
    assert str(pf) == "[(1, 2), (3, 1)]"
    assert repr(pf) == "ParetoFront([(1, 2), (3, 1)])"


def test_pareto_update_unique():
    """ Creating Pareto front by updating one by one. """
    list_ = [(1, 2, 3), (3, 2, 1), (0, 5, 0)]
    pf = ParetoFront()

    for i in range(len(list_)):
        pf.update(list_[i])
        assert list(pf) == list_[: i + 1]


def test_pareto_front_clear():
    """ Calling `clear` empties the Pareto front. """
    pf = ParetoFront([(1, 2), (2, 1)])
    assert list(pf) == [(1, 2), (2, 1)]

    pf.clear()
    assert list(pf) == []


def test_pareto_front_custom_function():
    """ Test construction of Pareto front with custom object and value function. """

    class A:
        def __init__(self, v1, v2):
            self.v1 = v1
            self.v2 = v2

    item1, item2, item3 = A(1, 2), A(2, 1), A(3, 1)
    pf = ParetoFront(get_values_fn=lambda x: (x.v1, x.v2))

    pf.update(item1)
    assert list(pf) == [item1]

    pf.update(item2)
    assert list(pf) == [item1, item2]

    pf.update(item3)
    assert list(pf) == [item1, item3]
