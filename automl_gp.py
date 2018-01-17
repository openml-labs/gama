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

def pset_from_config_new(configuration):
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
