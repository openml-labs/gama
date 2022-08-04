#!/bin/bash
PYPI_INFO=$( curl -Ls https://test.pypi.org/pypi/gama/json | sed "s/ //g")
GAMA_VERSION=$(cat gama/__version__.py | grep -Eo "\d+\.\d+\.\d+\.dev")
NEW_VERSION=$(python -c "
import json, sys; 
releases = json.loads(sys.argv[-2])['releases'].keys(); 
current_version = sys.argv[-1]; 
related_versions = [ver for ver in releases if ver.startswith(current_version)]
last_version = related_versions[-1] if related_versions else None
print(current_version + str(int(last_version.removeprefix(current_version))+1) if last_version else current_version+'0')
"  $PYPI_INFO $GAMA_VERSION)
echo "s/$GAMA_VERSION/$NEW_VERSION/"
sed -i '' -r "s/$GAMA_VERSION[0-9]+/$NEW_VERSION/" "gama/__version__.py"
