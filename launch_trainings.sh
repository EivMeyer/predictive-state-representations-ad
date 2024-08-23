#!/bin/bash

# Array of model types
MODEL_TYPES=(
    "PredictiveModelV5"
    "SimpleReconstructiveModel"
    "SingleStepPredictiveModel"
)

# Directory to store log files
LOG_DIR="output/training_logs"
mkdir -p $LOG_DIR

# Loop through each model type and launch a training process
for model in "${MODEL_TYPES[@]}"
do
    echo "Launching training for $model"
    nohup python3 train_model.py training.model_type=$model > "$LOG_DIR/${model}_training.log" 2>&1 &
    echo "Process ID: $!"
done

echo "All training processes have been launched."
echo "You can monitor the progress in the log files in the $LOG_DIR directory."
echo "Use 'tail -f $LOG_DIR/<model_name>_training.log' to follow a specific log file."