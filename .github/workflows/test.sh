PYPI_INFO=$( curl -Ls https://test.pypi.org/pypi/gama/json | sed "s/ //g")
echo $PYPI_INFO
cat gama/__version__.py
GAMA_VERSION=$(cat gama/__version__.py | grep -Eo "\d+\.\d+\.\d+\.dev")
echo $GAMA_VERSION
NEW_VERSION=$(python -c "import json, sys;releases = json.loads(sys.argv[-2])['releases'].keys();current_version = sys.argv[-1];last_published = next((ver for ver in releases if ver.startswith(current_version)), None);print(current_version + str(int(last_published.removeprefix(current_version))+1) if last_published else current_version+'0')"  $PYPI_INFO $GAMA_VERSION)
echo "s/$GAMA_VERSION/$NEW_VERSION/"
sed -i -e "s/$GAMA_VERSION/$NEW_VERSION/" "gama/__version__.py"
echo done
