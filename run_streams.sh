#!/bin/sh


echo start of job
for i in 0 1 3 4 5 6
do
	python Baseline_chacha.py $i > results/results_chacha_$i.txt
done
echo end of job
