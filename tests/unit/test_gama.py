import gama


def test_reproducible_initialization():
    g1 = gama.GamaClassifier(random_state=1, keep_analysis_log=False)
    pop1 = [g1._operator_set.individual() for _ in range(10)]

    g2 = gama.GamaClassifier(random_state=1, keep_analysis_log=False)
    pop2 = [g2._operator_set.individual() for _ in range(10)]
    for ind1, ind2 in zip(pop1, pop2):
        assert ind1.pipeline_str() == ind2.pipeline_str(), "The initial population should be reproducible."
