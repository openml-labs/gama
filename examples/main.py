"""
Created on Thu Jan 11 11:45:30 2018
@author: Pieter Gijsbers
"""
import sys
sys.path.append('..\openml-python')
from openml import tasks
import numpy as np
import logging

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from time import time

from gama import GamaRegressor
from gama import GamaClassifier

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    #ch = logging.StreamHandler(sys.stdout)
    #ch.setLevel(logging.DEBUG)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #ch.setFormatter(formatter)
    #root.addHandler(ch)

    mode = 'clf'
    if False:
        phoneme_task_id = 219
        task = tasks.get_task(phoneme_task_id)
        X, y = task.get_X_and_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=42)
    elif True:
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, shuffle=True, random_state=42)
    elif True:
        moneyball_task = 167148
        task = tasks.get_task(moneyball_task)
        X, y = task.get_X_and_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)
        mode = 'reg'

    for _ in range(1):
        start = time()
        gpaml = GamaClassifier(random_state=1, population_size=10, n_jobs=7, max_total_time=60)
        gpaml.generation_completed(lambda x: print('generation completed!', len(x), 'individuals.'))
        gpaml.fit(X_train, y_train, auto_ensemble_n=3)
        print('dur', time()-start)

    predictions_5 = gpaml.predict(X_test, auto_ensemble_n=3)
    print('Accuracy n=3:', log_loss(y_test, predictions_5))

    predictions_5 = gpaml.predict(X_test, auto_ensemble_n=7)
    print('Accuracy n=7:', log_loss(y_test, predictions_5))

    predictions_5 = gpaml.predict(X_test, auto_ensemble_n=10)
    print('Accuracy n=10:', log_loss(y_test, predictions_5))

