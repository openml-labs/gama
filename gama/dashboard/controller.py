""" Provides updates on GAMA's search.
 Next step is to call GAMA directly, but the `fit` call has to be made async. """

import shlex
import subprocess
from collections import defaultdict


class Controller:
    def __init__(self):
        self._subscribers = defaultdict(list)

    def start_gama(
        self,
        metric,
        regularize,
        n_jobs,
        max_total_time_h,
        max_total_time_m,
        max_eval_time_h,
        max_eval_time_m,
        input_file,
        log_dir,
        target,
    ):
        # For some reason, 0 input registers as None.
        max_total_time_h = 0 if max_total_time_h is None else max_total_time_h
        max_total_time_m = 0 if max_total_time_m is None else max_total_time_m
        max_eval_time_h = 0 if max_eval_time_h is None else max_eval_time_h
        max_eval_time_m = 0 if max_eval_time_m is None else max_eval_time_m
        max_total_time = max_total_time_h * 60 + max_total_time_m
        max_eval_time = max_eval_time_h * 60 + max_eval_time_m
        command = (
            f'gama "{input_file}" -v -n {n_jobs} -t {max_total_time} '
            f'--time_pipeline {max_eval_time} -outdir {log_dir} --target "{target}"'
        )
        if regularize != "on":
            command += " --long"
        if metric != "default":
            command += f" -m {metric}"

        command = shlex.split(command)
        # fake_command = ['python', '-h']
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        self._on_gama_started(process, log_dir)

    def _on_gama_started(self, process, log_file):
        for subscriber in self._subscribers["gama_started"]:
            subscriber(process, log_file)

    def gama_started(self, callback_function):
        self._subscribers["gama_started"].append(callback_function)

    def gama_ended(self, callback_function):
        self._subscribers["gama_ended"].append(callback_function)
