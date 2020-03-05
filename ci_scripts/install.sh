#!/bin/bash

if [ "$JOB" = "TEST" ]; then
  pip install -e .[test]
fi
if [ "$JOB" = "CHECK" ]; then
  pip install pre-commit
fi
