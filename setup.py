#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    'numpy>=1.14.0',
    'scipy>=1.0.0',
    'scikit-learn==0.20.0',
    'stopit>=1.1.1',
    'liac-arff>=2.2.2',
    'category-encoders>=1.2.8'
]

setup(
    name='gama',
    version='0.1dev',
    description='A package for automated machine learning based on scikit-learn.',
    author='Pieter Gijsbers',
    url='https://github.com/PGijsbers/GAMA',
    packages=find_packages(),
    install_requires=requirements
)
