Search Methods
--------------

.. module:: gama.search_methods

AsynchronousSuccessiveHalving
*****************************

.. autoclass:: AsynchronousSuccessiveHalving

AsyncEA
*******

.. autoclass:: AsyncEA

RandomSearch
************

.. autoclass:: RandomSearch

Developer Note
**************

Each search method consists of two parts:

    - a class that is used to store its configuration and determine default values
    - a function which implements the search algorithm

Some algorithms' hyperparameters' default values are best specified based on characteristics of the dataset,
we refer to these non-static defaults as 'dynamic defaults'.
Now each search method now consists of the search algorithm and (optionally) a method to determine default
values based on the data.
To keep these two functions together explicitly and keep naming consistent between different search methods,
we bundle them in a class derived from
`BaseSearch <https://github.com/PGijsbers/gama/tree/develop/gama/search_methods/base_search.py#L10>`_.

The function which implements the search algorithm can expect to be interrupted by a stopit.TimeoutException at any time.
This indicates the search algorithm should stop and its output should be set to your class' `output` member.
Additionally, the search algorithm should take at least the following hyperparameters:

 - operators: gama.genetic_programming.OperatorSet
    Has methods to create new individuals, evaluate individuals and more.
 - start_population: List[Individual]
    A list of individuals to be considered before all others.
