#!/usr/bin/env bash

set -e

git clone https://github.com/PGijsbers/gama.git --single-branch --branch gh-pages docs/build
sphinx-build -b html docs/source docs/build/$TRAVIS_BRANCH
