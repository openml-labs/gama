"""
Created on Thu Jan 11 11:45:30 2018
@author: Pieter Gijsbers
"""
import sys
sys.path.append('.\openml-python')
sys.path.append('..\..\openml-python')
from openml import tasks
import logging
import time
import pickle

from sklearn.model_selection import train_test_split

from gama.GamaRegressor import GamaRegressor
from gama.GamaClassifier import GamaClassifier
from gama.ea import evaluation

n_repeats = 3
task_list = [34537]
time_per_experiment = 50  # seconds
max_time_per_pipeline = 200  # seconds
n_jobs = 6
ensemble_size = 25
results = []
header = ['tid', 'repeat', 'time','score','ensemble_size','random_state','population_size','n_jobs','objectives','max_total_time','max_eval_time']
results.append(header)

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    gama_logger = logging.getLogger('gama')
    gama_logger.setLevel(logging.DEBUG)
    gama_logger.handlers = []
    lfh = logging.FileHandler('gama.log')
    gama_logger.addHandler(lfh)

    for task_id in task_list:
        root.info("Loading task {}".format(task_id))
        task = tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        root.info("Loaded task {} of type {} with measure {}.".format(task.task_id, task.task_type, task.evaluation_measure))

        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=i)
            metric = 'f1_macro' if task.task_type_id == 1 else 'root_mean_squared_error'
            gama_args = dict(
                random_state=1,
                population_size=50,
                n_jobs=n_jobs,
                objectives=(metric, 'size'),
                max_total_time=time_per_experiment,
                max_eval_time=max_time_per_pipeline
            )
            if task.task_type_id == 1:
                gama = GamaClassifier(**gama_args)
            elif task.task_type_id == 2:
                gama = GamaRegressor(**gama_args)
            else:
                root.error("Unknown task type ({}) for task {}".format(task.task_type, task.task_id))

            start = time.time()
            gama.fit(X_train, y_train, ensemble_size)
            end = time.time()

            predictions = gama.predict(X_test)
            score = evaluation.evaluate(evaluation.string_to_metric(metric), y_test, predictions)

            results.append([task.task_id, i, end-start, score, ensemble_size, *list(gama_args.values())])
            root.info("tid: {}, time: {:.1f}s, metric: {}, score: {:.4f}, ensemble_size: {}".format(
                task.task_id, end-start, metric, score, ensemble_size))

            gama.ensemble.build_initial_ensemble(1)# .predict(X_test)
            gama.ensemble.fit(X_train, y_train)
            predictions = gama.ensemble.predict(X_test)
            score = evaluation.evaluate(evaluation.string_to_metric(metric), y_test, predictions)

            results.append([task.task_id, i, end - start, score, 1, *list(gama_args.values())])
            root.info("tid: {}, time: {:.1f}s, metric: {}, score: {:.4f}, ensemble_size: {}".format(
                task.task_id, end-start, metric, score, 1))

            gama.delete_cache()

with open('benchmark_results.pkl','wb') as fh:
    pickle.dump(results, fh)
