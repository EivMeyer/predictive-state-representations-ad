#!/bin/bash

show_readme() {
    cat << EOF
ML Pipeline Script
==================

This script automates the process of dataset collection, representation model training,
and reinforcement learning model training.

Usage:
    ./run_pipeline.sh [OPTIONS] [CONFIG_OVERRIDES]

Options:
    --nohup             Run the script with nohup (keeps running after you log out)
    --output FILE       Specify the output file (default: ml_pipeline_output_TIMESTAMP.log)

Config Overrides:
    key=value           Override any configuration parameter (e.g., training.learning_rate=64)

Examples:
    ./run_pipeline.sh
    ./run_pipeline.sh --nohup --output my_run.log
    ./run_pipeline.sh training.learning_rate=64 dataset.t_pred=50 viewer.window_size=256

The script will guide you through the process, asking for necessary inputs along the way.
You can modify any configuration parameter by passing it as an argument.

EOF
}

get_input() {
    local prompt="$1"
    local default="$2"
    local input
    read -p "$prompt [$default]: " input
    echo "${input:-$default}"
}

check_dataset() {
    local dataset_dir="./output/dataset"
    if [ -d "$dataset_dir" ]; then
        local file_count=$(find "$dataset_dir" -name "batch_*.pt" | wc -l)
        local total_size=$(du -sh "$dataset_dir" | cut -f1)
        local last_modified=$(stat -c %y "$dataset_dir" | cut -d. -f1)
        echo "Existing dataset found:"
        echo "- Number of batch files: $file_count"
        echo "- Total size: $total_size"
        echo "- Last modified: $last_modified"
        read -p "Do you want to use this existing dataset? (y/n) " use_existing
        if [[ $use_existing == [yY] ]]; then
            return 0
        fi
    fi
    return 1
}

select_model() {
    local models_dir="./output/models"
    local model_files=($(find "$models_dir" -name "*.pth" -type f -printf '%T@ %p\n' | sort -n | tail -n 20 | cut -f2- -d" "))
    
    if [ ${#model_files[@]} -eq 0 ]; then
        echo "No existing model files found."
        return 1
    fi

    echo "Select a model file:"
    select model_path in "${model_files[@]}"; do
        if [ -n "$model_path" ]; then
            echo "$model_path"
            return 0
        else
            echo "Invalid selection. Please try again."
        fi
    done
}

show_system_info() {
    echo "System Information:"
    echo "CPU Info:"
    lscpu | grep "Model name\|CPU(s):"
    echo "Memory Info:"
    free -h
    echo "GPU Info:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    else
        echo "NVIDIA GPU not detected or nvidia-smi not available"
    fi
}

# Show README
show_readme

use_nohup=false
output_file=""
config_overrides=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --nohup)
        use_nohup=true
        shift
        ;;
        --output)
        output_file="$2"
        shift
        shift
        ;;
        *=*)
        config_overrides="$config_overrides $1"
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if [ -z "$output_file" ]; then
    output_file="pipeline_output.log"
fi

show_system_info

echo ""
echo "Configuration overrides: $config_overrides"
echo ""

if ! check_dataset; then
    num_workers=$(get_input "Enter the number of workers for dataset collection" "16")
    total_episodes=$(get_input "Enter the total number of episodes to collect" "10000")
    episodes_per_restart=$(get_input "Enter the number of episodes to collect before restarting a worker" "500")
else
    num_workers=""
    total_episodes=""
    episodes_per_restart=""
fi

echo ""
read -p "Do you want to train a new representation model? (y/n) " train_new_model
if [[ $train_new_model == [yY] ]]; then
    representation_epochs=$(get_input "Enter the number of epochs for training the representation model" "1000")
    model_path=""
else
    representation_epochs=""
    model_path=$(select_model)
    if [ -z "$model_path" ]; then
        echo "Error: No existing model selected and not training a new one. Cannot proceed."
        exit 1
    fi
    model_path=$(echo "$model_path" | tail -n 1)
fi

echo ""
rl_timesteps=$(get_input "Enter the total number of timesteps for RL training" "10000000")
rl_num_envs=$(get_input "Enter the number of parallel environments for RL training" "16")

echo "Configuration:"
if [ -n "$num_workers" ]; then
    echo "- Number of workers: $num_workers"
    echo "- Total episodes: $total_episodes"
    echo "- Episodes per restart: $episodes_per_restart"
fi
if [ -n "$representation_epochs" ]; then
    echo "- Representation model epochs: $representation_epochs"
else
    echo "- Using existing model: $model_path"
fi
echo "- RL training timesteps: $rl_timesteps"
echo "- RL parallel environments: $rl_num_envs"
echo "- Output file: $output_file"
echo "- Running with nohup: $use_nohup"
echo "- Configuration overrides: $config_overrides"

read -p "Do you want to proceed with this configuration? (y/n) " confirm
if [[ $confirm != [yY] ]]; then
    echo "Aborted by user."
    exit 1
fi

echo ""

run_pipeline() {
    local num_workers="$1"
    local total_episodes="$2"
    local episodes_per_restart="$3"
    local representation_epochs="$4"
    local model_path="$5"
    local rl_timesteps="$6"
    local rl_num_envs="$7"
    local config_overrides="$8"

    if [ -z "$num_workers" ]; then
        echo "Using existing dataset."
    else
        echo "Step 1: Collecting new dataset..."
        ./parallel_dataset_collection.sh -w "$num_workers" -e "$total_episodes" -r "$episodes_per_restart"  $config_overrides
    fi

    if [ -n "$representation_epochs" ]; then
        echo "Step 2: Training representation model..."
        python train_model.py training.epochs="$representation_epochs" $config_overrides

        latest_model=$(ls -t ./output/models/*/final_model.pth | head -n 1)
        if [ -z "$latest_model" ]; then
            echo "Error: No trained model found. Make sure train_model.py is saving the model correctly."
            exit 1
        fi

        model_path=$(realpath --relative-to=. "$latest_model")
    else
        echo "Step 2: Using existing representation model..."
    fi

    echo "Using representation model: $model_path"

    echo "Step 3: Training RL model..."
    python train_rl_agent.py rl_training.total_timesteps="$rl_timesteps" rl_training.num_envs="$rl_num_envs" representation.model_path="$model_path" $config_overrides

    echo "ML pipeline completed successfully!"
}

# Prepare the command to run
cmd="run_pipeline '$num_workers' '$total_episodes' '$episodes_per_restart' '$representation_epochs' '$model_path' '$rl_timesteps' '$rl_num_envs' '$config_overrides'"

if $use_nohup; then
    echo "Starting pipeline with nohup. Output will be written to $output_file"
    nohup bash -c "$(declare -f run_pipeline); $cmd" > "$output_file" 2>&1 &
    echo "Pipeline started in background. Process ID: $!"
else
    eval "$cmd" | tee "$output_file"
fi