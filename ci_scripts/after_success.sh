#!/bin/bash

if [ "$JOB" = "test" ]; then
  # codecov will merge reports automatically
  bash <(curl -s https://codecov.io/bash)
fi
if [ "$JOB" = "deploy" ]; then
  sphinx-build -b html docs/source docs/build
fi
