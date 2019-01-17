---
title: 'GAMA: Genetic Automated Machine learning Assistant'
tags:
  - AutoML
  - evolutionary algorithm
  - genetic programming
authors:
 - name: Pieter Gijsbers
   orcid: 0000-0001-7346-8075
   affiliation: "1"
 - name: Joaquin Vanschoren
   orcid: 0000-0001-7044-9805
   affiliation: "1"
affiliations:
 - name: Eindhoven University of Technology
   index: "1"
date: 13 October 2018
bibliography: paper.bib
---

# Summary

Successful machine learning applications hinge on a plethora of design decisions, which require extensive experience and relentless empirical evaluation. 
To train a successful model, one has to decide which algorithms to use, how to preprocess the data, and how to tune any hyperparameters that influence the final model.
Automating this process of algorithm selection and hyperparameter optimization in the context of machine learning is often called AutoML (Automated Machine Learning).

The usefulness of AutoML is twofold.
It makes creating good machine learning models more accessible to non-experts, as AutoML systems often support simple to use interfaces.
For experts, it takes away a time consuming process of model selection and hyperparameter tuning.
They can instead focus on related tasks such as interpreting the model or curating the data.

GAMA is an AutoML package for end-users and AutoML researchers.
It uses genetic programming to efficiently generate optimized machine learning pipelines given specific input data and resource constraints.
A machine learning pipeline contains data preprocessing as well as a machine learning algorithm, with fine-tuned hyperparameter settings.
By default, GAMA uses scikit-learn [@scikit-learn] implementations for preprocessing (e.g. Normalizer, PCA, ICA) and learners (e.g. Gaussian Naive Bayes, Support Vector Machine, Random Forest).

GAMA can also combine multiple tuned machine learning pipelines together into an ensemble, which on average should help model performance.
At the moment, GAMA is restricted to classification and regression problems on tabular data.

In addition to its general use AutoML functionality, GAMA aims to serve AutoML researchers as well.
During the optimization process, GAMA keeps an extensive log of progress made.
Using this log, insight can be obtained on the behaviour of the population of pipelines.
It can answer questions such as which mutation operator is most effective, how fitness changes over time, and how much time each algorithm takes.
For example, Figure 1 shows fitness over time (in blue) and pipeline length over time (in red), and is (simplified) output that can be generated from the analysis log.

![A simplified visualization of the optimization process based on log data.
The blue line indicates the moving average of performance over time (left y-axis, higher is better).
The red line represents the moving average of number of steps in the pipeline over time (right y axis, lower is better).
](https://raw.githubusercontent.com/PGijsbers/gama/master/images/fitnessgraph.png)

# Related Work

There are already many AutoML systems, both open-source and closed-source.
Amongst these are auto-sklearn [@autosklearn], TPOT [@TPOT], ML-Plan [@MLPlan] and Auto-WEKA [@AutoWEKA].
They differ in optimization strategy, programming language, target audience or underlying machine learning package.

Most closely related to GAMA is TPOT [@TPOT].
TPOT is also a Python package performing AutoML based on genetic programming using scikit-learn pipelines.
The major difference between the two is the evolutionary algorithm used.
TPOT uses a synchronous evolutionary algorithm, whereas GAMA uses an asynchronous algorithm.
Other differences include having a CLI (TPOT), exporting independent Python code (TPOT), direct ARFF support (GAMA) and visualization of the optimization process (GAMA).
Further comparison is contained in GAMA's documentation.

# Acknowledgements
This software was developed with support from the Data Driven Discovery of Models (D3M) program run by DARPA and the Air Force Research Laboratory.

# References
