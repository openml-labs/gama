#!/usr/bin/env python
import os

from setuptools import setup, find_packages

with open("gama/__version__.py", "r") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

base = [
    "numpy>=1.14.0",
    "scipy>=1.0.0",
    "scikit-learn>=0.24.0 < 0.25.0",
    "pandas>=1.0,<1.3",
    "stopit>=1.1.1",
    "liac-arff>=2.2.2",
    "category-encoders>=2.3.0",
    "black==19.10b0",
    "psutil",
]

vis = [
    "dash==1.3",
    "dash-daq==0.1.0",
    "dash-bootstrap-components",
    "visdcc",
]

doc = ["sphinx", "sphinx_rtd_theme"]

test = [
    "pre-commit==2.1.1",
    "pytest>=4.4.0",
    "pytest-mock",
    "pytest-xdist<2.0.0",
    "codecov",
    "pytest-cov",
]

# Black, Flake8 and Mypy will be installed through calling pre-commit install
dev = test + doc
all_ = test + doc + vis

with open(os.path.join("README.md")) as fid:
    README = fid.read()

setup(
    name="gama",
    version=version,
    description="A package for automated machine learning based on scikit-learn.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Pieter Gijsbers",
    author_email="p.gijsbers@tue.nl",
    url="https://github.com/openml-labs/GAMA",
    project_urls={
        "Bug Tracker": "https://github.com/openml-labs/gama/issues",
        "Documentation": "https://openml-labs.github.io/gama/",
        "Source Code": "https://github.com/openml-labs/gama",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=base,
    extras_require={"vis": vis, "dev": dev, "all": all_,},
    python_requires=">=3.6.0",
    entry_points={
        "console_scripts": [
            "gama=gama.utilities.cli:main",
            "gamadash=gama.dashboard.app:main",
        ]
    },
)
