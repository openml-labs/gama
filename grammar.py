from collections import defaultdict

import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler

#plus_probabilities = [1, 1] + (0.5**i for i in itertools.count())
config = {
    VotingClassifier:
        ("D P",
         {
            'estimators': 'P+',
            'voting': ['soft', 'hard']
         }),
    DecisionTreeClassifier:
        ("D P",
         {
            'max_depth': range(3,7)
         }),
    BernoulliNB:
        ("D[N,NN] P",
         {
            'alpha': [0.5, 1.0]
         }),
    Imputer:
        ("D D[N]",
         {}),
    MinMaxScaler:
        ("D[N] D[N,NN]",
         {})
}


class Terminal(object):
    def __init__(self, return_type, value):
        self._return_type = return_type
        self._value = value

    def __str__(self):
        if isinstance(self._value, list):
            return self._return_type + '=[' + ','.join([str(x) for x in self._value]) + ']'
        else:
            return self._return_type + '=' + str(self._value)


class Primitive(object):

    def __init__(self, name, input_, hyperparameters, output):
        self.name = name
        self.input = input_
        self.output = output
        self.hyperparameters = hyperparameters

    def __str__(self):
        return self.name

def create_pset(config_):
    primitives = defaultdict(list)
    terminals = defaultdict(list)
    for key, (grammar, hyperparameter_space) in config_.items():
        terminal_names = []
        for hp_key, values in hyperparameter_space.items():
            terminal_name = key.__name__ + '.' + hp_key
            terminal_names.append(terminal_name)
            if isinstance(values, str):
                # Specifies a grammar expression, which is handled at generation-time.
                terminals[terminal_name] = values
                continue
            for value in values:
                terminals[terminal_name].append(Terminal(terminal_name, value))
        input_, output = grammar.split(' ')
        primitive = Primitive(key.__name__, input_, terminal_names, output)
        primitives[output].append(primitive)
    return primitives, terminals


class list_with_str(list):

    def __str__(self):
        primitive, input_process, p_terminals = self
        if input_process is not None:
            preprocess_str = [individual_to_string(input_process)]
        else:
            preprocess_str = []
        return str(primitive) + '(' + ','.join(preprocess_str + [str(el) for el in p_terminals]) + ')'

def generate_individual(primitives, terminals, input_='D', output='P'):
    primitive = np.random.choice(primitives[output])
    p_terminals = []

    for hyperparameter_name in primitive.hyperparameters:
        if isinstance(terminals[hyperparameter_name], str):
            # Grammar expression
            n_repeats = 3 if terminals[hyperparameter_name].endswith('+') else 1
            children = [list_with_str(generate_individual(primitives, terminals, input_, output=terminals[hyperparameter_name][0])) for _ in range(n_repeats)]
            p_terminals.append(Terminal(hyperparameter_name, children))
        else:
            p_terminals.append(np.random.choice(terminals[hyperparameter_name]))

    if primitive.input != input_:
        input_process = generate_individual(primitives, terminals, input_, output=primitive.input)
    else:
        input_process = None

    return [primitive, input_process, p_terminals]


def individual_to_string(ind):
    primitive, input_process, p_terminals = ind
    if input_process is not None:
        preprocess_str = [individual_to_string(input_process)]
    else:
        preprocess_str = []
    return str(primitive) + '(' + ','.join(preprocess_str+[str(el) for el in p_terminals]) + ')'


prims, terms = create_pset(config)
individual_to_string(generate_individual(prims, terms))
# generate_individual(create_pset(config))

