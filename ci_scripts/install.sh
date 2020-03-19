#!/bin/bash

if [ "$JOB" = "test" ]; then
  pip install -e .[dev]
fi
if [ "$JOB" = "check" ]; then
  pip install pre-commit
fi
