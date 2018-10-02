import unittest

from gama.utilities.pretty_string_methods import clean_pipeline_string


def pretty_string_methods_test_suite():
    test_cases = [PrettyStringMethodsTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class PrettyStringMethodsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_clean_pipeline_string_length_1(self):
        """ A string with a pipeline of one component is formatted correctly. """
        full_string = "GaussianNB(data)"
        self.assertEqual(
            "GaussianNB()",
            clean_pipeline_string(full_string)
        )

    def test_clean_pipeline_string_length_2(self):
        """ A string of a pipeline with two components is formatted correctly. """
        full_string = ("RandomForestClassifier("
                       "FeatureAgglomeration(data, "
                       "FeatureAgglomeration.affinity='l2', "
                       "FeatureAgglomeration.linkage='complete'), "
                       "RandomForestClassifier.bootstrap=True, "
                       "RandomForestClassifier.criterion='gini', "
                       "RandomForestClassifier.max_features=0.6, "
                       "RandomForestClassifier.min_samples_leaf=7, "
                       "RandomForestClassifier.min_samples_split=6, "
                       "RandomForestClassifier.n_estimators=100)")
        expected_string = ("RandomForestClassifier("
                           "FeatureAgglomeration("
                           "affinity='l2', "
                           "linkage='complete'), "
                           "bootstrap=True, "
                           "criterion='gini', "
                           "max_features=0.6, "
                           "min_samples_leaf=7, "
                           "min_samples_split=6, "
                           "n_estimators=100)")
        self.assertEqual(
            expected_string,
            clean_pipeline_string(full_string)
        )

    def test_clean_pipeline_string_invalid(self):
        """ A string that is not of a pipeline raises a ValueError. """
        full_string = "This is not a valid string"
        with self.assertRaises(ValueError) as error:
            clean_pipeline_string(full_string)

        self.assertTrue("All pipeline strings should contain the data terminal." in str(error.exception))