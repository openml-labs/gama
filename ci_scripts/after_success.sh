#!/bin/bash

if [ "$JOB" = "test" ]; then
  # codecov will merge reports automatically
  bash <(curl -s https://codecov.io/bash)
fi
