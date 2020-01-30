#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    'numpy==1.17.3',
    'scipy>=1.0.0',
    'scikit-learn==0.20.2',
    'stopit>=1.1.1',
    'liac-arff>=2.2.2',
    'category-encoders>=1.2.8',
    'interpret'
]

test_requirements = [
    'pytest>=4.4.0',
    'pytest-mock',
    'pytest-xdist',
    'codecov',
    'pytest-cov'
]

visualization_requirements = [
    'dash==1.1.1',
    'dash-daq==0.1.0'
]

documentation_requirements = [
    'sphinx',
    'sphinx_rtd_theme'
] + visualization_requirements

all_ = requirements + visualization_requirements + documentation_requirements

setup(
    name='gama',
    version='19.08.0',
    description='A package for automated machine learning based on scikit-learn. d3m-version',
    long_description='',
    long_description_content_type='text/markdown',
    author='Pieter Gijsbers',
    author_email='p.gijsbers@tue.nl',
    url='https://github.com/PGijsbers/GAMA',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    extras_require={
        'test': test_requirements,
        'vis': visualization_requirements,
        'doc': documentation_requirements,
        'all': all_
    },
    python_requires='>=3.6.0'
)
