""" Contains classes and function(s) which help define automated machine 
learning as a genetic programming problem.
(Yes, I need to find a better file name.)
"""
from collections import defaultdict

import numpy as np
from deap import gp
import sklearn
from sklearn.pipeline import Pipeline

from stacking_transformer import make_stacking_transformer

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
    pset.renameArguments(ARG0="data")
    
    for path, hyperparameters in configuration.items():
        if '.' in path:
            module_path, class_ = path.rsplit('.', maxsplit=1)
            exec(f"from {module_path} import {class_}")
        else:
            class_ = path
            exec(f"import {class_}")
        
        hyperparameter_types = []
        for name, values in hyperparameters.items():
            # We construct a new type for each hyperparameter, so we can specify
            # it as terminal type, making sure it matches with expected
            # input of the operators. Moreover it automatically makes sure that
            # crossover only happens between same hyperparameters.
            hyperparameter_type = type(f"{class_}{name}",(object,), {})
            hyperparameter_types.append(hyperparameter_type)
            for value in values:
                # Escape string values with quotes otherwise they are variables
                value_str = f"'{value}'" if isinstance(value, str) else f"{value}"
                hyperparameter_str = f"{class_}.{name}={value_str}"            
                pset.addTerminal(value, hyperparameter_type, hyperparameter_str)
                
        class_type = eval(class_)
        if issubclass(class_type, sklearn.base.TransformerMixin):
            pset.addPrimitive(class_type, [Data, *hyperparameter_types], Data)
        elif issubclass(class_type, sklearn.base.ClassifierMixin):
            pset.addPrimitive(class_type, [Data, *hyperparameter_types], Predictions)
            stacking_class = make_stacking_transformer(class_type)
            pset.addPrimitive(stacking_class, [Data, *hyperparameter_types], Data, name = class_ + stacking_class.__name__)
        else:
            raise TypeError(f"Expected {class_} to be either subclass of "
                            "TransformerMixin or ClassifierMixin.")
    
    return pset

def compile_individual(ind, pset):
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
                return terminal_name[terminal_name.rfind('.',0,equal_idx)+1:equal_idx]
            args = {
                    extract_arg_name(p.name): pset.context[p.name]
                    for r, p in required_provided
                    }
            class_ = pset.context[prim.name]
            # All pipeline components must have a unique name
            name = prim.name + str(name_counter[prim.name])
            name_counter[prim.name] += 1
            components.append((name, class_(**args)))
            ind = ind[1:-len(args)]
        else:
            raise TypeError("Type is wrong or missing.")
            
    return Pipeline(list(reversed(components)))
