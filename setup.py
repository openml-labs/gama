#!/usr/bin/env python
import os

from setuptools import setup, find_packages

with open("gama/__version__.py", 'r') as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    'numpy>=1.14.0',
    'scipy>=1.0.0',
    'scikit-learn>=0.20.0',
    'stopit>=1.1.1',
    'liac-arff>=2.2.2',
    'category-encoders>=1.2.8'
]

test_requirements = [
    'pytest>=4.4.0',
    'pytest-mock',
    'pytest-xdist',
    'codecov',
    'pytest-cov'
]

visualization_requirements = [
    'dash==1.3',
    'dash-daq==0.1.0',
    'dash-bootstrap-components',
    'visdcc'
]

documentation_requirements = [
    'sphinx',
    'sphinx_rtd_theme'
] + visualization_requirements

all_ = requirements + visualization_requirements + documentation_requirements

with open(os.path.join("README.md")) as fid:
    README = fid.read()

setup(
    name='gama',
    version=version,
    description='A package for automated machine learning based on scikit-learn.',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Pieter Gijsbers',
    author_email='p.gijsbers@tue.nl',
    url='https://github.com/PGijsbers/GAMA',
    project_urls={
        "Bug Tracker": "https://github.com/PGijsbers/gama/issues",
        "Documentation": "https://pgijsbers.github.io/gama/",
        "Source Code": "https://github.com/PGijsbers/gama",
    },
    packages=find_packages(exclude=['tests', "tests.*"]),
    install_requires=requirements,
    extras_require={
        'test': test_requirements,
        'vis': visualization_requirements,
        'doc': documentation_requirements,
        'all': all_
    },
    python_requires='>=3.6.0',
    entry_points={'console_scripts': ['gama=gama.utilities.cli:main', 'gamadash=gama.dashboard.app:main']}
)
