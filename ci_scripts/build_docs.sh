#!/usr/bin/env bash

set -e

git clone https://github.com/PGijsbers/gama.git --single-branch --branch gh-pages docs/build

release_branch_regex="[0-9][0-9].[0-9]+.x"

if [[ $TRAVIS_BRANCH =~ release_branch_regex ]]; then
  echo "Truncating branch name"
  branchname=${TRAVIS_BRANCH%.x}
  sed -i -E "s/url=\S*\//url=$branchname\//" index.html
  sphinx-build -b html docs/source docs/build/$branchname
else
  sphinx-build -b html docs/source docs/build/$TRAVIS_BRANCH
fi
