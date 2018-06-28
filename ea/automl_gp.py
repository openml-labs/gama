""" Contains classes and function(s) which help define automated machine 
learning as a genetic programming problem.
(Yes, I need to find a better file name.)
"""
from collections import defaultdict
import os
import pickle
import logging
import uuid
import time

import numpy as np
from deap import gp, tools
import sklearn
from sklearn.pipeline import Pipeline
import stopit

from ..utilities.stacking_transformer import make_stacking_transformer
from ..ea.modified_deap import gen_grow_safe

from .mutation import mut_replace_terminal, mut_replace_primitive
from .evaluation import cross_val_predict_score

log = logging.getLogger(__name__)


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
            transformer_tags = ["DATA_PREPROCESSING", "FEATURE_SELECTION", "DATA_TRANSFORMATION"]
            if (issubclass(key, sklearn.base.TransformerMixin) or
                  (hasattr(key, 'metadata') and key.metadata.query()["primitive_family"] in transformer_tags)):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Data)
            elif (issubclass(key, sklearn.base.ClassifierMixin) or
                  (hasattr(key, 'metadata') and key.metadata.query()["primitive_family"] == "CLASSIFICATION")):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Predictions)

                if False:
                    # Does not work with multi-processing.
                    stacking_class = make_stacking_transformer(key)
                    primname = key.__name__ + stacking_class.__name__
                    pset.addPrimitive(stacking_class, [Data, *hyperparameter_types], Data, name=primname)
                    if key.__name__ in parameter_checks:
                        parameter_checks[primname] = parameter_checks[key.__name__]
            elif (issubclass(key, sklearn.base.RegressorMixin) or
                  (hasattr(key, 'metadata') and key.metadata.query()["primitive_family"] == "REGRESSION")):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Predictions)

                if False:
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
            if len(remainder) > 0:
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
        if n_kwargs == 0:
            ind = ind[1:]
        else:
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
    while len(ind) > 0:
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
            if len(args) > 0:
                ind = ind[1:-len(args)]
            else:
                ind = ind[1:]
        else:
            raise TypeError("Type is wrong or missing.")
    return Pipeline(list(reversed(components)))


def evaluate_pipeline(pl, X, y, timeout, scoring='accuracy', cv=5, cache_dir=None, logger=None):
    """ Evaluates a pipeline used k-Fold CV. """
    if not logger:
        logger = log

    if pl is None:
        return (-float("inf"), timeout, float("inf"))

    start = time.process_time()
    with stopit.ThreadingTimeout(timeout) as c_mgr:
        try:
            prediction, score = cross_val_predict_score(pl, X, y, cv=cv, scoring=scoring)
        except stopit.TimeoutException:
            score = -float("inf")
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.info('{} encountered while evaluating pipeline.'.format(type(e)))#, exc_info=True)
            logger.debug('Error evaluating pipeline {}. {}: {}'.format(pl, type(e), e))#, exc_info=True)
            score = -float("inf")

    if cache_dir and score != -float("inf"):
        pl_filename = str(uuid.uuid4())

        try:
            with open(os.path.join(cache_dir, pl_filename + '.pkl'), 'wb') as fh:
                pickle.dump((pl, prediction, score), fh)
        except FileNotFoundError:
            log.warning("File not found while saving predictions. This can happen in the multi-process case if the "
                        "cache gets deleted within `max_eval_time` of the end of the search process.", exc_info=True)

    evaluation_time = time.process_time() - start
    pipeline_length = len(pl.steps)
    
    if c_mgr.state == c_mgr.INTERRUPTED:
        # A TimeoutException was raised, but not by the context manager.
        # This indicates that the outer context manager (the ea) timed out.
        logger.info("Outer-timeout during evaluation of {}".format(pl))
        raise stopit.utils.TimeoutException()

    if not c_mgr:
        # For now we treat a eval timeout the same way as e.g. NaN exceptions.
        fitness_values = (-float("inf"), timeout, pipeline_length)
        logger.info('Timeout encountered while evaluating pipeline.')#, exc_info=True)
        logger.debug("Timeout after {}s: {}".format(timeout, pl))
    else:
        fitness_values = (score, evaluation_time, pipeline_length)

    return fitness_values


def generate_valid(pset, min_, max_, toolbox):
    """ Generates a valid pipeline. """
    for _ in range(50):
        ind = gen_grow_safe(pset, min_, max_)
        pl = toolbox.compile(ind)
        if pl is not None:
            return ind
    raise Exception


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


def individual_length(individual):
    """ Gives a measure for the length of the pipeline. Currently, this is the number of primitives. """
    return len([el for el in individual if isinstance(el, gp.Primitive)])


def eliminate_NSGA(pop, n):
    return tools.selNSGA2(pop, k=len(pop))[-n:]


def eliminate_worst(pop, n):
    return list(sorted(pop, key=lambda x: x.fitness.wvalues[0]))[-n:]


def offspring_mate_and_mutate(pop, n, cxpb, mutpb, toolbox):
    """ Creates n new individuals based on the population. Can apply both crossover and mutation. """
    offspring = []
    for _ in range(n):
        ind1, ind2 = np.random.choice(range(len(pop)), size=2, replace=False)
        ind1, ind2 = toolbox.clone(pop[ind1]), toolbox.clone(pop[ind2])
        if np.random.random() < cxpb:
            ind1, ind2 = toolbox.mate(ind1, ind2)
        if np.random.random() < mutpb:
            ind1, = toolbox.mutate(ind1)
        offspring.append(ind1)
    return offspring
