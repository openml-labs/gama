import math


def asha(operations=operations, start_candidates=start_candidates,  # General Search Hyperparameters
         reduction_factor=3, minimum_resource=1, maximum_resource=10, minimum_early_stopping_rate=5):  # Algorithm Specific
    rungs = range(math.floor(math.log(maximum_resource/minimum_resource, reduction_factor))
                  - minimum_early_stopping_rate)
    pipelines_by_rung = {rung: [] for rung in rungs}
    promoted_pipelines = {rung: [] for rung in rungs}

    def get_job():
        for rung, pipelines in reversed(pipelines_by_rung.items()):
            candidates = sorted(pipelines)[:math.floor(len(pipelines) / reduction_factor)]
            promotable = [candidate for candidate in candidates if candidate not in promoted_pipelines[rung]]
            if len(promotable) > 0:
                return promotable[0], rung + 1

        if len(start_candidates) > 0:
            return start_candidates.pop(), 0
        else:
            return operations.new(), 0

    # Maybe get function dispatcher too? So don't have to know the number of jobs.
    # e.g. a `for_each`?
    with FunctionDispatcher() as workers:
        pipeline, rung = get_job()
        workers.queue_evaluation((pipeline, rung))
        pipeline, rung = workers.get_next_result()
        pipelines_by_rung[rung] = (loss, pipeline)
