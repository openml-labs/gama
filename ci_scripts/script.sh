#!/bin/bash

if [ "$JOB" = "TEST" ]; then
  pytest --cov=gama -sv -n 4 tests/"$SUITE"/
fi
if [ "$JOB" = "CHECK" ]; then
  pre-commit run --all-files
fi
