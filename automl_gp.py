""" Contains classes and function(s) which help define automated machine 
learning as a genetic programming problem.
(Yes, I need to find a better file name.)
"""
from collections import defaultdict

import numpy as np
from deap import gp, creator
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import stopit

from stacking_transformer import make_stacking_transformer
from modified_deap import gen_grow_safe


class Data(np.ndarray):
    """ Dummy class that represents a dataset."""
    pass 


class Predictions(np.ndarray):
    """ Dummy class that represents prediction data. """
    pass


def pset_from_config(configuration):
    """ Create a pset for the given configuration dictionary.
    
    Given a configuration dictionary specifying operators (e.g. sklearn 
    estimators), their hyperparameters and values for each hyperparameter,
    create a gp.PrimitiveSetTyped that contains:
        - For each operator a primitive
        - For each possible hyperparameter-value combination a unique terminal
        
    Side effect: Imports the classes of each primitive.
        
    Returns the given Pset.
    """
    pset = gp.PrimitiveSetTyped("pipeline", in_types=[Data], ret_type=Predictions)
    parameter_checks = {}
    pset.renameArguments(ARG0="data")
    
    shared_hyperparameter_types = {}
    # We have to make sure the str-keys are evaluated first: they describe shared hyperparameters
    # We can not rely on order-preserving dictionaries as this is not in the Python 3.5 specification.
    sorted_keys = reversed(sorted(configuration.keys(), key=lambda x: str(type(x))))
    for key in sorted_keys:
        values = configuration[key]
        if isinstance(key, str):
            # Specification of shared hyperparameters
            hyperparameter_type = type(str(key), (object,), {})
            shared_hyperparameter_types[key] = hyperparameter_type
            for value in values:
                # Escape string values with quotes
                value_str = "'{}'".format(value) if isinstance(value, str) else str(value)
                hyperparameter_str = "{}={}".format(key, value_str)
                pset.addTerminal(value, hyperparameter_type, hyperparameter_str)
        elif isinstance(key, object):
            #Specification of operator (learner, preprocessor)
            hyperparameter_types = []
            for name, param_values in values.items():
                # We construct a new type for each hyperparameter, so we can specify
                # it as terminal type, making sure it matches with expected
                # input of the operators. Moreover it automatically makes sure that
                # crossover only happens between same hyperparameters.
                if param_values == []:
                    # An empty list indicates a shared hyperparameter
                    hyperparameter_types.append(shared_hyperparameter_types[name])
                elif name == "param_check":
                    # This allows users to define illegal hyperparameter combinations, but is not a terminal.
                    parameter_checks[key.__name__] = param_values[0]
                else:                
                    hyperparameter_type = type("{}{}".format(key.__name__, name), (object,), {})
                    hyperparameter_types.append(hyperparameter_type)
                    for value in param_values:
                        # Escape string values with quotes otherwise they are variables
                        value_str = ("'{}'".format(value) if isinstance(value, str)
                                     else "{}".format(value.__name__) if callable(value)
                                     else str(value))
                        hyperparameter_str = "{}.{}={}".format(key.__name__, name, value_str)
                        pset.addTerminal(value, hyperparameter_type, hyperparameter_str)

            # After registering the hyperparameter types, we can register the operator itself.
            if issubclass(key, sklearn.base.TransformerMixin):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Data)
            elif issubclass(key, sklearn.base.ClassifierMixin):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Predictions)

                if True:
                    # Does not work with multi-processing.
                    stacking_class = make_stacking_transformer(key)
                    primname = key.__name__ + stacking_class.__name__
                    pset.addPrimitive(stacking_class, [Data, *hyperparameter_types], Data, name=primname)
                    if key.__name__ in parameter_checks:
                        parameter_checks[primname] = parameter_checks[key.__name__]
            elif issubclass(key, sklearn.base.RegressorMixin):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Predictions)

                if True:
                    # Does not work with multi-processing.
                    stacking_class = make_stacking_transformer(key)
                    primname = key.__name__ + stacking_class.__name__
                    pset.addPrimitive(stacking_class, [Data, *hyperparameter_types], Data, name=primname)
                    if key.__name__ in parameter_checks:
                        parameter_checks[primname] = parameter_checks[key.__name__]
            else:
                raise TypeError("Expected {} to be either subclass of "
                                "TransformerMixin, RegressorMixin or ClassifierMixin.".format(key))
        else:
            raise TypeError('Encountered unknown type as key in dictionary.'
                            'Keys in the configuration should be str or class.')
    
    return pset, parameter_checks


def compile_individual(ind, pset, parameter_checks=None):
    """ Compile the individual to a sklearn pipeline."""
    components = []
    name_counter = defaultdict(int)
    while len(ind) > 0:
        prim, remainder = ind[0], ind[1:]
        if isinstance(prim, gp.Terminal):
            if len(remainder)>0:
                raise Exception
            break

        try:
            component, n_kwargs = expression_to_component(prim, reversed(remainder), pset, parameter_checks)
        except ValueError:
            return None

        # Each component in the pipeline must have a unique name.
        name = prim.name + str(name_counter[prim.name])
        name_counter[prim.name] += 1

        components.append((name, component))
        ind = ind[1:-n_kwargs]

    return Pipeline(list(reversed(components)))


def expression_to_component(primitive, terminals, pset, parameter_checks=None):
    """ Creates Python-object for the primitive-terminals combination.

    It is allowed to have trailing terminals in the list, they will be ignored.

    Returns an instantiated python object and the number of terminals used.
    """
    # See if all terminals have a value provided (except Data Terminal)
    required = reversed([terminal for terminal in primitive.args if not terminal.__name__ == 'Data'])
    required_provided = list(zip(required, terminals))
    if not all(r == p.ret for (r, p) in required_provided):
        print([(r, p.ret) for (r,p) in required_provided])
        raise ValueError('Missing {}-terminal for {}-primitive.')

    def extract_arg_name(terminal_name):
        equal_idx = terminal_name.rfind('=')
        start_parameter_name = terminal_name.rfind('.', 0, equal_idx) + 1
        return terminal_name[start_parameter_name:equal_idx]

    kwargs = {
        extract_arg_name(p.name): pset.context[p.name]
        for r, p in required_provided
    }

    primitive_class = pset.context[primitive.name]

    if (parameter_checks is not None
            and primitive.name in parameter_checks
            and not parameter_checks[primitive.name](kwargs)):
        raise ValueError('Not a valid configuration according to the parameter check.')

    return primitive_class(**kwargs), len(kwargs)


def compile_individual_tree(ind, pset, parameter_checks=None):
    """ Compile the individual to a sklearn pipeline."""
    components = []
    name_counter = defaultdict(int)
    while (len(ind) > 0):
        # log_message('compiling ' + str(ind), level = 4)
        prim, remainder = ind[0], ind[1:]
        if isinstance(prim, gp.Terminal):
            if len(remainder) > 0:
                raise Exception
            break
        # See if all terminals have a value provided (except Data Terminal)
        required_provided = list(zip(reversed(prim.args[1:]), reversed(remainder)))
        if all(r == p.ret for (r, p) in required_provided):
            # log_message('compiling ' + str([p.name for r, p in required_provided]), level = 5)
            # If so, instantiate the pipeline component with given arguments.
            def extract_arg_name(terminal_name):
                equal_idx = terminal_name.rfind('=')
                start_parameter_name = terminal_name.rfind('.', 0, equal_idx) + 1
                return terminal_name[start_parameter_name:equal_idx]

            args = {
                extract_arg_name(p.name): pset.context[p.name]
                for r, p in required_provided
            }
            class_ = pset.context[prim.name]
            # All pipeline components must have a unique name
            name = prim.name + str(name_counter[prim.name])
            name_counter[prim.name] += 1
            if (parameter_checks is not None
                    and prim.name in parameter_checks
                    and not parameter_checks[prim.name](args)):
                return None

            components.append((name, class_(**args)))
            ind = ind[1:-len(args)]
        else:
            raise TypeError("Type is wrong or missing.")

    return Pipeline(list(reversed(components)))


def evaluate_pipeline(pl, X, y, timeout, scoring='accuracy', cv=5):
    """ Evaluates a pipeline used k-Fold CV. """
    
    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            score = np.mean(cross_val_score(pl, X, y, cv=cv, scoring=scoring))
        except stopit.TimeoutException:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(type(e), str(e))
            score = -float("inf")
    
    if c_mgr.state == c_mgr.INTERRUPTED:
        print('Interrupt!')
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        raise stopit.TimeoutException()

    if not c_mgr:
        print('Evaluation timeout')
        # For now we treat a eval timeout the same way as e.g. NaN exceptions.
        fitness_values = (-float("inf"), timeout)
    else:
        fitness_values = (score, c_mgr.seconds)
            
    return fitness_values


def generate_valid(pset, min_, max_, toolbox):
    """ Generates a valid pipeline. """
    for _ in range(50):
        ind = gen_grow_safe(pset, min_, max_)
        pl = toolbox.compile(ind)
        if pl is not None:
            return ind
    raise Exception


def mut_replace_terminal(ind, pset):
    """ Mutation function which replaces a terminal."""
    
    eligible = [i for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    #els = [el for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    if eligible == []:
        #print('No way to mutate '+str(ind)+' was found.')
        return ind,
    
    to_change = np.random.choice(eligible)    
    alternatives = [t for t in pset.terminals[ind[to_change].ret] if t != ind[to_change]]
    ind[to_change] = np.random.choice(alternatives)
    return ind, 


def find_unmatched_terminal(individual):
    """ Finds the location of the first terminal that can not be matched with a primitive.
    
    Raises a `ValueError` if no terminals are found.
    """
    unmatched_args = []    
    for i, el in enumerate(individual):
        if len(unmatched_args) > 0 and el.ret == unmatched_args[0]:
            unmatched_args.pop(0)
        elif issubclass(type(el), gp.Terminal):
            return i
        if issubclass(type(el), gp.Primitive):
            # Replace with list-inserts if performance is bad.
            unmatched_args = el.args + unmatched_args
    
    return False


def mut_replace_primitive(ind, pset):
    """ Mutation function which replaces a primitive (and corresponding terminals). """
    # DEAP.gp's mutNodeReplacement does not work since it will only replace primitives
    # if they have the same input arguments (which is not true in this context)
    
    eligible = [i for i,el in enumerate(ind) if (issubclass(type(el), gp.Primitive) and len(pset.primitives[el.ret])>1)]
    if eligible == []:
        return ind,
    
    to_change = np.random.choice(eligible)   
    number_of_removed_terminals = len(ind[to_change].args) - 1
    
    # Determine new primitive and terminals that need to be added.
    alternatives = [prim for prim in pset.primitives[ind[to_change].ret] if prim.name != ind[to_change].name]
    new_primitive = np.random.choice(alternatives)
    new_terminals = [np.random.choice(pset.terminals[ret_type]) for ret_type in new_primitive.args[1:]]
    
    # Determine which terminals should also be removed.
    # We want to find the first unmatched terminal, but can ignore the data 
    # input terminal, as that is a subtree we do not wish to replace.
    terminal_index = find_unmatched_terminal(ind[to_change+1:])
    if (terminal_index is False):
        if (number_of_removed_terminals) == 0:
            # No terminals need to be removed and everything after the primitive is a perfect (data) subtree.
            new_expr = ind[:] + new_terminals
            new_expr[to_change] = new_primitive
            return creator.Individual(new_expr),
        else:
            raise Exception("Found no unmatched terminals after removing a primitive which had terminals: {}".format(str(ind)))
    else:        
        # Adjust for the fact the searched individual had part removed.
        # (Since the unmatched terminal was created through removing primitives
        # before it, this means the adjustment is always necessary)
        terminal_index += (to_change+1)
        # In the case the unmatched terminal was the Data terminal, we actually
        # would like to start adding terminals only after this position.
        # This way there is no need to make a distinction later on whether a
        # primitive's data-terminal is a leaf or a subtree.    
        if (ind[terminal_index].value in pset.arguments):
            terminal_index += 1
    
        # 3. Construct the new individual
        # Replacing terminals can not be done in-place, as the number of terminals can vary.
        new_expr = ind[:terminal_index] + new_terminals + ind[terminal_index+number_of_removed_terminals:]
        # Replacing the primitive can be done in-place.
        new_expr[to_change] = new_primitive
        #expr = ind[:to_change] + [new_primitive] + ind[to_change+1:terminal_index] + new_terminals + ind[terminal_index+number_of_removed_terminals:]
        ind = creator.Individual(new_expr)
        
        return ind, 

def random_valid_mutation(ind, pset):
    """ Picks a mutation uniform at random from options which are possible. 
    
    The choices are `mut_random_primitive`, `mut_random_terminal`, 
    `mutShrink` and `mutInsert`.
    In particular a pipeline can not shrink a primitive if it only has one.
    """
    available_mutations = [mut_replace_terminal, mut_replace_primitive, gp.mutInsert]
    if len([el for el in ind if issubclass(type(el), gp.Primitive)]) > 1:
        available_mutations.append(gp.mutShrink)
        
    mut_fn = np.random.choice(available_mutations)
    if gp.mutShrink == mut_fn:
        # only mutShrink function does not need pset.
        return mut_fn(ind)
    else:
        return mut_fn(ind, pset)


def pipeline_length(individual):
    """ Gives a measure for the length of the pipeline. Currently, this is the number of primitives. """
    return len([el for el in individual if isinstance(el, gp.Primitive)])
