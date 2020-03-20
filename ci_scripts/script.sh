#!/bin/bash

set -e

if [ "$JOB" = "check" ] || [ "$JOB" = "deploy" ]; then
  pre-commit run --all-files
fi
if [ "$JOB" = "test" ]; then
  pytest --cov=gama -sv -n 4 tests/"$SUITE"/
fi
if [ "$JOB" = "deploy" ]; then
  pytest -sv -n 4 tests/unit/
  pytest -sv -n 4 tests/system/
fi
