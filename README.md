# GAMA
**G**eneral **A**utomated **M**achine learning **A**ssistant  
An automated machine learning tool based on genetic programming.
	
## TODO
[ ] Expand unit and system tests  
[ ] Support other (custom?) (multi-objective) optimization metrics  
[ ] Expand the grammar to include NaN/negative/non-numeric pre/post conditions  
[ ] Support multi-threading  
[ ] Model serializability  
[ ] Progress output to console (in a nice way)  
[ ] Implement bias correction for crossvalidation (eg. BBC-CV)  
[ ] Support time-based cut-off  
[ ] Add visualization  
[ ] Add logging  

## Nice to have
[ ] Python code export  
[ ] Custom operator  
	[ ] Specifically also support some popular packages like LightGBM or XGBoost  
[ ] Use half-trained models (eg. RF that only had first x trees trained)  

## Ideas
[ ] Schema that can define how mutation/crossover should vary over time  
[ ] Will including a Voting/Stacking Estimator 'just work'?  
[ ] (on-the-fly/meta)-hyperparameter space pruning  

## Known issues
[ ] Nested configuration not supported (e.g. specify estimators for RFE)  
[ ] Preprocessing may result in invalid data (e.g. feature selection )  
[ ] No check on input types (e.g. don't pass negative data to NB)       
[ ] Many duplicate pipelines -> Should maybe only be fixed for *some* selection methods.  
