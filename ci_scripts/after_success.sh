#!/bin/bash

if [ "$JOB" = "TEST" ]; then
  # This should be the last test to complete.
  bash <(curl -s https://codecov.io/bash)
fi
