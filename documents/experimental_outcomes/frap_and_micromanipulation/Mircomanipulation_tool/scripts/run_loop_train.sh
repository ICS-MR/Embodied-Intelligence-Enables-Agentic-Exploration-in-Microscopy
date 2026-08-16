#!/bin/bash

CONFIG_DIR="configs"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for cfg in "$CONFIG_DIR"/Splicing_2*.yaml; do
    cfg_name=$(basename "$cfg" .yaml)
    timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
    log_file="$LOG_DIR/${cfg_name}_${timestamp}.log"

    echo "=============================="
    echo ">> Starting training with config: $cfg"
    echo ">> Logging to: $log_file"
    echo "=============================="

    python 3_model_train.py --config "$cfg" > "$log_file" 2>&1

    echo ">> Finished training with $cfg"
    echo ""
done