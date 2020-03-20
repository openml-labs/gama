#!/bin/bash

if [ "$JOB" = "check" ]; then
  exit $(pre-commit run --all-files)
fi
if [ "$JOB" = "test" ]; then
  exit $(pytest --cov=gama -sv -n 4 tests/"$SUITE"/)
fi
if [ "$JOB" = "deploy" ]; then
  if [ $(pre-commit run --all-files) ] || \
     [ $(pytest --cov=gama -sv -n 4 tests/unit/) ] || \
     [ $(pytest --cov=gama -sv -n 4 tests/system/) ]
  then
    exit $?
  fi
fi
