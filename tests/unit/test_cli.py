from typing import List

import pytest

import gama
from gama.utilities.cli import main


def test_classifier_invocation(capfd):
    main("tests/data/breast_cancer_train.arff -dry")
    out, err = capfd.readouterr()
    assert "classification" in out


def test_classifier_invocation_csv(capfd):
    main("tests/data/openml_d_23380.csv --target TR -dry")
    out, err = capfd.readouterr()
    assert "classification" in out


def test_regressor_invocation(capfd):
    main("tests/data/boston.arff -dry")
    out, err = capfd.readouterr()
    assert "regression" in out


def test_complex_invocation(capfd):
    main("tests/data/boston.arff --target MEDV -py myfile.py -t 60 -v -n 4 -dry")
    out, err = capfd.readouterr()
    assert "regression" in out
    assert gama.__version__ in out
    assert "n_jobs=4" in out
    assert "max_total_time=3600" in out


def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        main("invalid.file -dry")


def test_invalid_argument():
    with pytest.raises(SystemExit) as e:
        main("tests/data/boston.arff -invalid")
    assert 2 == e.value.code
