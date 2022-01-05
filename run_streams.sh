#!/bin/sh


echo start of job
for i in 0 1 3 4 5 6
do
	python Streams.py $i 5000 > results/results_LeverageBagging_$i.txt
done
echo end of job
