#!/usr/bin/env bash

set -e

echo "Building documentation"
git clone https://github.com/PGijsbers/gama.git --single-branch --branch gh-pages docs/build
sphinx-build -b html docs/source docs/build/$TRAVIS_BRANCH
