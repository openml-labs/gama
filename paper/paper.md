---
title: 'GAMA: Genetic Automated Machine learning Assistant'
tags:
  - AutoML
  - evolutionary algorithm
  - genetic programming
authors:
 - name: Pieter Gijsbers
   orcid: 0000-0001-7346-8075
   affiliation: Eindhoven University of Technology
date: 02 October 2018
bibliography: paper.bib
---

# Summary

Machine learning is a technique to automatically learn models from data.
Those models are used today in many real world applications, from detecting fraudulent transactions to recommending movies to driving cars autonomously.
To create a successful model with machine learning, an expert has to pick the right algorithms to use, as well as do fine-tuning in the form of hyperparameter optimization.

In recent years an active field of research has developed around automating this process of algorithm selection and hyperparameter optimization in the context of machine learning.
This field is often called AutoML (Automated Machine Learning), and an AutoML system is able to automatically create a good machine learning pipeline from data.
Its usefulness is twofold.
It makes creating good machine learning models more accessible to non-experts, as AutoML systems often support simple to use interfaces.
For experts, it takes away a time consuming process of model selection and hyperparameter tuning.
They can instead focus on related tasks such as interpreting the model or curating the data.

GAMA is an AutoML package for end-users and AutoML researchers.
Given data and resource constraints, it will try to construct good machine learning pipelines.
A machine learning pipeline contains data preprocessing as well as a machine learning algorithm, with fine-tuned hyperparameter settings.
GAMA can also combine multiple tuned machine learning pipelines together into an ensemble, which on average should help model performance.
At the moment, GAMA is restricted to classification and regression problems.

Provide insight into the evolutionary process.

# Related Work

There are already many AutoML systems, both open-source and closed-source.
Amongst these are auto-sklearn[@autosklearn], TPOT[@TPOT], ML-Plan[@MLPlan], Auto-WEKA[@Auto-WEKA] and H2O[@H2O].
They differ in optimization strategy, programming language, target audience or underlying machine learning package.

Most closely related to GAMA is TPOT[@TPOT].
TPOT is also a Python package performing AutoML based on genetic programming using scikit-learn pipelines.
However, there are differences.
TPOT uses generation based evolution, whereas GAMA uses asynchronous evolution.


# Notes

In addition, your paper should include:

    A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience
    A clear statement of need that illustrates the purpose of the software
    The software should have an obvious research application
    A list of key references including a link to the software archive
    Mentions (if applicable) of any ongoing research projects using the software
     or recent scholarly publications enabled by it

    The software should be a significant contribution to the available open source software that either enables some new research challenges
     to be addressed or makes addressing research challenges significantly better (e.g., faster, easier, simpler)
    The software should be feature complete (no half-baked solutions) Minor, ‘utility’ packages, including ‘thin’ API clients are not acceptable

JOSS publishes articles about research software.
 This definition includes software that: solves complex modeling problems in a scientific context
  (physics, mathematics, biology, medicine, social science, neuroscience, engineering); 
  supports the functioning of research instruments or the execution of research experiments; 
  extracts knowledge from large data sets; offers a mathematical library, or similar.

https://joss.readthedocs.io/en/latest/review_criteria.html

# Acknowledgements

# References
