#!/bin/sh

# Convert script path into package name
SCRIPT_PATH=$1
SCRIPT_PATH_WITHOUT_EXTENSION="${SCRIPT_PATH%.*}"
SCRIPT_NAME_AS_PACKAGE=$(echo "$SCRIPT_PATH_WITHOUT_EXTENSION" | tr '/' '.')

# Remove script path from argument list
shift;

# Allow to use common packages from main dir
CURRENT_DIR=$(pwd)
export PYTHONPATH=$CURRENT_DIR
echo "PYTHONPATH=$PYTHONPATH"

# Run recipe from script
python -m "$SCRIPT_NAME_AS_PACKAGE" "$@"