import pytest

import gama


def test_reproducible_initialization():
    g1 = gama.GamaClassifier(random_state=1, keep_analysis_log=None)
    pop1 = [g1._operator_set.individual() for _ in range(10)]

    g2 = gama.GamaClassifier(random_state=1, keep_analysis_log=None)
    pop2 = [g2._operator_set.individual() for _ in range(10)]
    for ind1, ind2 in zip(pop1, pop2):
        assert ind1.pipeline_str() == ind2.pipeline_str(), "The initial population should be reproducible."


def test_gama_fail_on_invalid_hyperparameter_values():
    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(max_total_time=0)
    assert "max_total_time should be integer greater than zero" in str(e.value)

    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(max_total_time=None)
    assert "max_total_time should be integer greater than zero" in str(e.value)

    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(max_eval_time=0)
    assert "max_eval_time should be None or integer greater than zero" in str(e.value)

    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(n_jobs=-2)
    assert "n_jobs should be -1 or positive integer" in str(e.value)
