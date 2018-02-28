# Rename
An automated machine learning tool based on genetic programming.

## Known issues
[ ] Nested configuration not supported (e.g. specify estimators for RFE)

[ ] Preprocessing may result in invalid data (e.g. feature selection )

[ ] No check on input types (e.g. don't pass negative data to NB)

[ ] How to with invalidly generated pipelines (prevent them? how?)
    > https://docs.python.org/2/faq/design.html#how-fast-are-exceptions
    > https://stackoverflow.com/questions/2522005/cost-of-exception-handlers-in-python
    >> Just testing with if-statement after creation seems best?
    >> Maybe check function should suggest which hyperparameter to change?
    >> Or create a list of valid hyperparameter configurations.-> explicit cross product (?)
    >> must be smarter techniques.
	
## Ideas
Meta-learning
(on-the-fly/meta)-hyperparameter space pruning
Use half-trained models (eg. RF that only had first x trees trained)