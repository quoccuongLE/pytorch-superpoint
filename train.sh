#!/bin/bash

set -a
source .env
set +a

python train.py .vscode/magicpoint_shapes_pair_dev.yaml magicpoint_synth
