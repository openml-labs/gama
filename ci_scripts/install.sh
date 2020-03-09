#!/bin/bash

if [ "$JOB" = "TEST" ]; then
  pip install -e .[dev]
fi
if [ "$JOB" = "CHECK" ]; then
  pip install pre-commit
fi
