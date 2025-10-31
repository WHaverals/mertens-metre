#!/bin/bash

set -e

ENV_NAME="mertens-metre-env"
PYTHON_VERSION="3.12"

echo "creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "installing dependencies"
pip install -r requirements.txt

echo "setup complete!"
echo "activate env with: conda activate $ENV_NAME"

