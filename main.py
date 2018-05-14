"""
Created on Thu Jan 11 11:45:30 2018
@author: Pieter Gijsbers
"""
import sys
sys.path.append('..\openml-python')
from openml import tasks
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from time import time

from GamaRegressor import GamaRegressor
from GamaClassifier import GamaClassifier
from visualization.visualizer import Visualizer

if __name__ == '__main__':
    mode = 'clf'
    if False:
        phoneme_task_id = 219
        task = tasks.get_task(phoneme_task_id)
        X, y = task.get_X_and_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=42)
    elif True:
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, shuffle=True, random_state=42)
    elif False:
        moneyball_task = 167148
        task = tasks.get_task(moneyball_task)
        X, y = task.get_X_and_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)
        mode = 'reg'

    for _ in range(1):
        start = time()
        gpaml = GamaClassifier(random_state=1, generations=5, population_size=20, n_jobs=1, async=False)
        gpaml.generation_completed(lambda x: print('generation completed!', len(x), 'individuals.'))
        viz = Visualizer()
        gpaml.evaluation_completed(viz.new_evaluation_result)
        gpaml._observer.on_pareto_updated(viz.new_pareto_entry)
        gpaml.fit(X_train, y_train)
        print('dur', time()-start)
    predictions_1 = gpaml.predict(X_test)
    print('Accuracy n=1:', accuracy_score(y_test, predictions_1))
    predictions_5 = gpaml.predict(X_test, auto_ensemble_n=5)
    print('Accuracy n=5:', accuracy_score(y_test, predictions_5))

