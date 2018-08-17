#!/usr/bin/env python

from setuptools import setup

requirements = [
    'numpy>=1.14.0',
    'scipy>=1.0.0',
    'scikit-learn>=0.19.0',
    'deap>=1.2',
    'stopit>=1.1.1'
]

setup(
    name='gama',
    version='0.1dev',
    description='A package for automated machine learning based on scikit-learn.',
    author='Pieter Gijsbers',
    url='https://github.com/PGijsbers/GAMA',
    packages=['gama'],
    install_requires=requirements
)
