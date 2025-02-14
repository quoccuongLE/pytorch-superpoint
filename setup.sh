#!/bin/bash
VENV_DIR=${1:-".venv/superpoint"}
if [ -f lock.conda.yaml ]; then
    conda env create -f lock.conda.yaml --prefix $VENV_DIR
else
    conda env create -f conda.yaml --prefix $VENV_DIR
    conda env export > lock.conda.yaml
fi
