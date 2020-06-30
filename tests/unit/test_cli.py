import subprocess
import sys
from typing import List
import gama


def cli_command(file) -> List[str]:
    return [sys.executable, "gama/utilities/cli.py", file, "-dry"]


def test_classifier_invocation():
    command = cli_command("tests/data/breast_cancer_train.arff")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert 0 == process.returncode, process.stderr
    assert "classification" in str(process.stdout)


def test_classifier_invocation_csv():
    command = cli_command("tests/data/openml_d_23380.csv")
    command.extend("--target TR".split(" "))
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert 0 == process.returncode, process.stderr
    assert "classification" in str(process.stdout)


def test_regressor_invocation():
    command = cli_command("tests/data/boston.arff")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert 0 == process.returncode, process.stderr
    assert "regression" in str(process.stdout)


def test_complex_invocation():
    command = cli_command("tests/data/boston.arff")
    command.extend("--target MEDV -py myfile.py -t 60 -v -n 4".split(" "))
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert 0 == process.returncode, process.stderr
    assert "regression" in str(process.stdout)
    assert gama.__version__ in str(process.stdout)
    assert "n_jobs=4" in str(process.stdout)
    assert "max_total_time=3600" in str(process.stdout)


def test_invalid_file():
    command = cli_command("invalid.file")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert 0 != process.returncode, "Invalid file should terminate with non-zero code"
    assert "FileNotFoundError: invalid.file" in str(process.stderr)


def test_invalid_argument():
    command = cli_command("tests/data/boston.arff")
    command.append("-invalid")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert 0 != process.returncode, "Invalid arguments should cause non-zero exit code"
    assert "unrecognized arguments: -invalid" in str(process.stderr)
