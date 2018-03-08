""" Contains classes and function(s) which help define automated machine 
learning as a genetic programming problem.
(Yes, I need to find a better file name.)
"""
from collections import defaultdict

import numpy as np
from deap import gp, creator
import sklearn
from sklearn.pipeline import Pipeline

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
    pset = gp.PrimitiveSetTyped("pipeline",in_types=[Data], ret_type=Predictions)
    parameter_checks = {}
    pset.renameArguments(ARG0="data")
    
    shared_hyperparameter_types = {}
    for key, values in configuration.items():
        if isinstance(key, str):
            # Specification of shared hyperparameters
            hyperparameter_type = type(f"{key}",(object,), {})
            shared_hyperparameter_types[key] = hyperparameter_type
            for value in values:
                # Escape string values with quotes otherwise they are variables
                value_str = f"'{value}'" if isinstance(value, str) else f"{value}"
                hyperparameter_str = f"{key}={value_str}"            
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
                    hyperparameter_types.append(shared_hyperparameter_types[name])
                elif name == "param_check":
                    parameter_checks[key.__name__] = param_values[0]
                else:                
                    hyperparameter_type = type(f"{key.__name__}{name}",(object,), {})
                    hyperparameter_types.append(hyperparameter_type)
                    for value in param_values:
                        # Escape string values with quotes otherwise they are variables
                        value_str = (f"'{value}'" if isinstance(value, str) 
                                     else f"{value.__name__}" if callable(value)
                                     else f"{value}")
                        hyperparameter_str = f"{key.__name__}.{name}={value_str}"            
                        pset.addTerminal(value, hyperparameter_type, hyperparameter_str)
            
            if issubclass(key, sklearn.base.TransformerMixin):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Data)
            elif issubclass(key, sklearn.base.ClassifierMixin):
                pset.addPrimitive(key, [Data, *hyperparameter_types], Predictions)
                
                stacking_class = make_stacking_transformer(key)
                primname = key.__name__ + stacking_class.__name__
                pset.addPrimitive(stacking_class, [Data, *hyperparameter_types], Data, name = primname)
                if key.__name__ in parameter_checks:
                    parameter_checks[primname] = parameter_checks[key.__name__]
            else:
                raise TypeError(f"Expected {key} to be either subclass of "
                                "TransformerMixin or ClassifierMixin.")
        else:
            raise TypeError('Encountered unknown type as key in dictionary.'
                            'Keys in the configuration should be str or class.')
                            
        
    
    return pset, parameter_checks

def compile_individual(ind, pset, parameter_checks = None):
    """ Compile the individual to a sklearn pipeline."""
    components = []
    name_counter = defaultdict(int)
    while(len(ind) > 0):
        #log_message('compiling ' + str(ind), level = 4)
        prim, remainder = ind[0], ind[1:]
        if isinstance(prim, gp.Terminal):
            if len(remainder)>0:
                print([el.name for el in remainder])
                raise Exception
            break
        # See if all terminals have a value provided (except Data Terminal)
        required_provided = list(zip(reversed(prim.args[1:]), reversed(remainder)))
        if all(r==p.ret for (r,p) in required_provided):            
            #log_message('compiling ' + str([p.name for r, p in required_provided]), level = 5)
            # If so, instantiate the pipeline component with given arguments.
            def extract_arg_name(terminal_name):
                equal_idx = terminal_name.rfind('=')
                start_parameter_name = terminal_name.rfind('.',0,equal_idx)+1
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

def generate_valid(pset, min_, max_, toolbox):
    """ Generates a valid pipeline. """
    for _ in range(50):
        ind = gen_grow_safe(pset, min_, max_)
        pl = toolbox.compile(ind)
        if pl is not None:
            return ind
    raise Exception('Failed')

def mut_replace_terminal(ind, pset):
    """ Mutation function which replaces a terminal."""
    
    eligible = [i for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    #els = [el for i,el in enumerate(ind) if (issubclass(type(el), gp.Terminal) and len(pset.terminals[el.ret])>1)]
    if eligible == []:
        #print('No way to mutate '+str(ind)+' was found.')
        return ind,
    
    to_change = np.random.choice(eligible)    
    ind[to_change] = np.random.choice(pset.terminals[ind[to_change].ret])
    return ind, 

def find_next_unmatched_terminal(individual, start, pset, ignore_pset_arguments=True):
    """ Finds the location of the terminal that provides input of `ret_type` to a given primitive. """
    
    # We need to keep track of terminals we expect.
    argument_ret_types = [pset.mapping[tname].ret for tname in pset.arguments]
    unmatched_args = []
    
    for i, el in enumerate(individual[start:], start=start):
        if issubclass(type(el), gp.Primitive):
            new_expected_args = [arg_type for arg_type in el.args if arg_type not in argument_ret_types]
            # Replace with list-inserts if performance is bad.
            unmatched_args = new_expected_args + unmatched_args
        elif el.ret in argument_ret_types:
            # We ignore 
            continue
        elif len(unmatched_args) > 0 and el.ret == unmatched_args[0]:
            unmatched_args.pop(0)
        elif len(unmatched_args) == 0:
            return i
    
    return i+1
    #raise ValueError(f"No unmatched terminals found. Suggested: {i}")    

def mut_replace_primitive(ind, pset):
    """ Mutation function which replaces a primitive (and corresponding terminals). """
    # DEAP.gp's mutNodeReplacement does not work since it will only replace primitives
    # if they have the same input arguments (which is not true in this context)
    
    eligible = [i for i,el in enumerate(ind) if (issubclass(type(el), gp.Primitive) and len(pset.primitives[el.ret])>1)]
    if eligible == []:
        return ind,
    
    to_change = np.random.choice(eligible)   
    
    # Replacing a primtive requires three steps:
    # 1. Determine which terminals should also be removed.
    # We want to find the first unmatched terminal, but can ignore the data 
    # input terminal, as that is a subtree we do not wish to replace.
    terminal_index = find_next_unmatched_terminal(ind, to_change + 1, pset)  
    number_of_removed_terminals = len(ind[to_change].args) - 1
            
    # 2. Determine new primitive and terminals need to be added.
    new_primitive = np.random.choice(pset.primitives[ind[to_change].ret])
    new_terminals = [np.random.choice(pset.terminals[ret_type]) for ret_type in new_primitive.args[1:]]
    
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