import unittest

from deap import gp

from automl_gp import mut_replace_primitive, mut_replace_terminal
from gama import Gama

def automl_gp_test_suite():
    test_cases = [AutomlGpTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class AutomlGpTestCase(unittest.TestCase):
    
    def setUp(self):
        self.gama = Gama(random_state=0)
        self.len_2_individual = """RandomForestClassifier(
        FeatureAgglomeration(
                data,
                FeatureAgglomeration.linkage='complete',
                FeatureAgglomeration.affinity='l2'), 
        RandomForestClassifier.n_estimators=100, 
        RandomForestClassifier.criterion='gini', 
        RandomForestClassifier.max_features=0.6000000000000001, 
        RandomForestClassifier.min_samples_split=6, 
        RandomForestClassifier.min_samples_leaf=7, 
        RandomForestClassifier.bootstrap=True)"""
        
    # pset_from_config
    # compile_individual
    # generate_valid
    # mut_replace_terminal
    # find_next_unmatched_terminal
    # mut_replace_primitive
    # random_valid_mutation
    
    def test_mut_replace_terminal(self):
        """ Tests if mut_replace_terminal replaces exactly one terminal. 
        
        The fact that the new expression is also valid already follows from the
        fact an individual is made from it.
        """
        ind = gp.PrimitiveTree.from_string(self.len_2_individual, self.gama._pset)
        ind_clone = self.gama._toolbox.clone(ind)
        new_ind, = mut_replace_terminal(ind_clone, self.gama._pset)
        
        replaced_elements = [el1 for el1, el2 in zip(ind, new_ind) if el1.name != el2.name]            
        
        self.assertEqual(len(replaced_elements), 1,
                         "Exactly one component should be replaced. Found {}".format(replaced_elements))        
        self.assertTrue(isinstance(replaced_elements[0], gp.Terminal), 
                        "Replaced component should be a terminal, is {}".format(type(replaced_elements[0])))    
    
    def test_mut_replace_primitive(self):
        """ Tests if mut_replace_primitive replaces exactly one primitive. 
        
        The fact that the new expression is also valid already follows from the
        fact an individual is made from it.
        """
        ind = gp.PrimitiveTree.from_string(self.len_2_individual, self.gama._pset)
        ind_clone = self.gama._toolbox.clone(ind)
        new_ind, = mut_replace_primitive(ind_clone, self.gama._pset)
        
        replaced_primitives = [el1 for el1, el2 in zip(ind, new_ind) if (el1.name != el2.name and isinstance(el1, gp.Primitive))]            
        
        self.assertEqual(len(replaced_primitives), 1,
                         "Exactly one primitive should be replaced. Found {}".format(replaced_primitives))