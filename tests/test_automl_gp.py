import unittest

from deap import gp

from automl_gp import mut_replace_primitive, mut_replace_terminal, find_unmatched_terminal, compile_individual
from gama import Gama

def automl_gp_test_suite():
    test_cases = [AutomlGpTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class AutomlGpTestCase(unittest.TestCase):
    """ Unit Tests for some functions of automl_gp.py.
    
    Functions excluded (for now?):
    [ ] pset_from_config
    [ ] compile_individual 
        If this breaks it's probably not subtle, hard to test edge cases, leave it to system test?
    [ ] generate_valid
        If this breaks it's probably not subtle, hard to test edge cases, leave it to system test?
    [ ] random_valid_mutation
        Not sure how to test this except just calling it a couple times and registering output.
    """
    
    def setUp(self):
        self.gama = Gama(random_state=0)
        
        self.ind_strings = [
                "GaussianNB(data)",
                """RandomForestClassifier(
                FeatureAgglomeration(
                        data,
                        FeatureAgglomeration.linkage='complete',
                        FeatureAgglomeration.affinity='l2'), 
                RandomForestClassifier.n_estimators=100, 
                RandomForestClassifier.criterion='gini', 
                RandomForestClassifier.max_features=0.6000000000000001, 
                RandomForestClassifier.min_samples_split=6, 
                RandomForestClassifier.min_samples_leaf=7, 
                RandomForestClassifier.bootstrap=True)""",
                """LinearSVC(data,
                LinearSVC.penalty='l2',
                LinearSVC.loss='squared_hinge',
                LinearSVC.dual=True,
                LinearSVC.tol=1e-05,
                LinearSVC.C=0.001)"""
                ]
        
        self.individuals = { 
                ind_str: gp.PrimitiveTree.from_string(ind_str, self.gama._pset)
                for ind_str in self.ind_strings
                }
        
    
    
    def test_find_next_unmatched_terminal(self):
        
        # (data) -> immediately missing
        pruned_ind = self.individuals[self.ind_strings[0]][1:]
        self.assertEqual(find_unmatched_terminal(pruned_ind), 0)
        
        # FA(data, FA_t1, FA_t2), RFC_t1,...,RFC_tN -> 4
        pruned_ind = self.individuals[self.ind_strings[1]][1:]
        self.assertEqual(find_unmatched_terminal(pruned_ind), 4)
        
        # RFC(__(data, FA_t1, FA_t2), RFC_t1,...,RFC_tN -> 1
        pruned_ind = self.individuals[self.ind_strings[1]][:1] + self.individuals[self.ind_strings[1]][2:]
        self.assertEqual(find_unmatched_terminal(pruned_ind), 2)
        
    
    def test_mut_replace_terminal(self):
        """ Tests if mut_replace_terminal replaces exactly one terminal. """
        ind = self.individuals[self.ind_strings[1]]
        ind_clone = self.gama._toolbox.clone(ind)
        new_ind, = mut_replace_terminal(ind_clone, self.gama._pset)
        
        replaced_elements = [el1 for el1, el2 in zip(ind, new_ind) if el1.name != el2.name]            
        
        self.assertEqual(len(replaced_elements), 1,
                         "Exactly one component should be replaced. Found {}".format(replaced_elements))        
        self.assertTrue(isinstance(replaced_elements[0], gp.Terminal), 
                        "Replaced component should be a terminal, is {}".format(type(replaced_elements[0]))) 
        # Should be able to compile the individual.
        compile_individual(new_ind, self.gama._pset)
        
    def test_mut_replace_primitive_len_1_no_terminal(self):
        """ Tests if mut_replace_primitive replaces exactly one primitive. """
        ind = self.individuals[self.ind_strings[1]]
        ind_clone = self.gama._toolbox.clone(ind)
        new_ind, = mut_replace_primitive(ind_clone, self.gama._pset)
        
        replaced_primitives = [el1 for el1, el2 in zip(ind, new_ind) if (el1.name != el2.name and isinstance(el1, gp.Primitive))]            
        
        self.assertEqual(len(replaced_primitives), 1,
                         "Exactly one primitive should be replaced. Found {}".format(replaced_primitives))
        # Should be able to compile the individual.
        compile_individual(new_ind, self.gama._pset)
        
        ind = self.individuals[self.ind_strings[2]]
        ind_clone = self.gama._toolbox.clone(ind)
        new_ind, = mut_replace_primitive(ind_clone, self.gama._pset)
        
        replaced_primitives = [el1 for el1, el2 in zip(ind, new_ind) if (el1.name != el2.name and isinstance(el1, gp.Primitive))]            
        
        self.assertEqual(len(replaced_primitives), 1,
                         "Exactly one primitive should be replaced. Found {}".format(replaced_primitives))
        # Should be able to compile the individual.
        compile_individual(new_ind, self.gama._pset)
    
    def test_mut_replace_primitive_len_2(self):
        """ Tests if mut_replace_primitive replaces exactly one primitive. """
        ind = self.individuals[self.ind_strings[1]]
        ind_clone = self.gama._toolbox.clone(ind)
        new_ind, = mut_replace_primitive(ind_clone, self.gama._pset)
        
        replaced_primitives = [el1 for el1, el2 in zip(ind, new_ind) if (el1.name != el2.name and isinstance(el1, gp.Primitive))]            
        
        self.assertEqual(len(replaced_primitives), 1,
                         "Exactly one primitive should be replaced. Found {}".format(replaced_primitives))
        try:
            # Should be able to compile the individual.
            compile_individual(new_ind, self.gama._pset)
        except Exception as e:
            self.fail(
                    f"""Mutated individual could not be compiled because of error: {str(e)}\n
                    Original: {str(ind)}\n
                    New: {str(new_ind)}
                    """
                    )