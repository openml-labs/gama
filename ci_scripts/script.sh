#!/bin/bash

if [ "$JOB" = "test" ]; then
  pytest --cov=gama -sv -n 4 tests/"$SUITE"/
fi
if [ "$JOB" = "check" ]; then
  pre-commit run --all-files
fi
