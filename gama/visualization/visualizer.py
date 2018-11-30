import matplotlib.pyplot as plt


class Visualizer(object):

    def __init__(self):
        self._all_points = []
        self._max_by_pop = []
        self._pareto = []

        self._fig, axes = plt.subplots(nrows=1, ncols=2)
        self._pareto_plot, self._fitness_plot = axes
        self._pareto_plot.set_title('Pareto')
        self._pareto_plot.set_xlabel('criterion 1')
        self._pareto_plot.set_ylabel('criterion 2')

        self._fitness_plot.set_title('Fitness')
        self._fitness_plot.set_xlabel('#individuals')
        self._fitness_plot.set_ylabel('max fitness')

    def new_evaluation_result(self, individual):
        self._all_points.append(individual)
        max_score = max(map(lambda ind: ind.fitness.values[0], self._all_points))
        self._max_by_pop.append((len(self._all_points), max_score))
        pop_sizes, max_scores = zip(*self._max_by_pop)
        self._fitness_plot.plot(pop_sizes, max_scores)
        self._draw()

    def new_pareto_entry(self, ind):
        self._pareto.append(ind.fitness.values)
        c1s, c2s = zip(*self._pareto)
        self._pareto_plot.scatter(c1s, c2s, c='r')
        c1, c2 = ind.fitness.values
        self._pareto_plot.scatter(c1, c2, c='b')
        self._draw()

    def _draw(self):
        self._fig.show()
        self._fig.canvas.draw()
        plt.pause(0.05)
