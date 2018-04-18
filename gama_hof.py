class HallOfFame(object):
    
    def __init__(self, filename):
        self._filename = filename
        self._pop = []
        self._last_pop = []
        
    def update(self, pop):
        self._pop += pop
        self._last_pop = pop
        
        with open(self._filename,'a') as fh:
            # print('-gen-')
            # print('\n'.join([str((str(ind), ind.fitness.values[0])) for ind in pop]))
            fh.write('\n'.join([str((str(ind), ind.fitness.values[0])) for ind in pop]))

    def best_n(self, n):
        best_pipelines = sorted(self._pop, key=lambda x: (-x.fitness.values[0], str(x)))
        return best_pipelines[:n]
