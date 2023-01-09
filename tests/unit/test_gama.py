import pytest

import os
import shutil

import gama


def test_output_directory_must_be_empty(tmp_path):
    with open(tmp_path / "remove.txt", "w") as fh:
        fh.write("Created for GAMA unit test.")

    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(output_directory=tmp_path)
    assert "`output_directory`" in str(e.value)


def test_reproducible_initialization():
    g1 = gama.GamaClassifier(random_state=1, store="nothing")
    pop1 = [g1._operator_set.individual() for _ in range(10)]

    g2 = gama.GamaClassifier(random_state=1, store="nothing")
    pop2 = [g2._operator_set.individual() for _ in range(10)]
    assert all(
        [ind1.pipeline_str() == ind2.pipeline_str() for ind1, ind2 in zip(pop1, pop2)]
    ), "The initial population should be reproducible."
    g1.cleanup("all")
    g2.cleanup("all")


def test_gama_fail_on_invalid_hyperparameter_values():
    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(max_total_time=0, store="nothing")
    assert "Expect positive int for max_total_time" in str(e.value)

    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(max_total_time=None, store="nothing")
    assert "Expect positive int for max_total_time" in str(e.value)

    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(max_eval_time=0, store="nothing")
    assert "Expect None or positive int for max_eval_time" in str(e.value)

    with pytest.raises(ValueError) as e:
        gama.GamaClassifier(n_jobs=-2, store="nothing")
    assert "n_jobs should be -1 or positive int but is" in str(e.value)
