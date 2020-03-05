#!/bin/bash

if [ "$JOB" = "TEST" ] && [ "$SUITE" = "system" ]; then
  # This should be the last test to complete.
  bash <(curl -s https://codecov.io/bash)
fi
