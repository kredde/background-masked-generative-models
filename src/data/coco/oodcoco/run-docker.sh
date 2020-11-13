#!/bin/bash
SOURCE_DIR=$(pwd)
DATA_DIR=/data
RESULTS_DIR=$(realpath results)

docker run --rm -it --name oodcoco \
  -u "$(id -u)":"$(id -g)" \
  --group-add 1001 \
  -v "$SOURCE_DIR":/deploy:rw -w /deploy \
  -v "$DATA_DIR":/data \
  -v "$RESULTS_DIR":/results:rw \
  oodcoco python -m oodcoco.oodcoco
