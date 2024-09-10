#!/bin/bash

# Function to get a valid integer input from the user
get_integer_input() {
    local prompt="$1"
    local value
    
    while true; do
        read -p "$prompt" value
        if [[ "$value" =~ ^[0-9]+$ ]]; then
            echo "$value"
            return 0
        else
            echo "Please enter a valid integer."
        fi
    done
}

# Initialize variables
num_workers=""
total_episodes=""
episodes_per_restart=""
config_overrides=""
append_mode=false

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w) num_workers="$2"; shift 2 ;;
        -e) total_episodes="$2"; shift 2 ;;
        -r) episodes_per_restart="$2"; shift 2 ;;
        --append) append_mode=true; shift ;;
        *=*) config_overrides="$config_overrides $1"; shift ;;
        *) echo "Usage: $0 [-w num_workers] [-e total_episodes] [-r episodes_per_restart] [--append] [CONFIG_OVERRIDES]" >&2
           exit 1 ;;
    esac
done

# Get number of workers if not provided
if [ -z "$num_workers" ]; then
    num_workers=$(get_integer_input "Enter the number of workers: ")
fi

# Get total number of episodes if not provided
if [ -z "$total_episodes" ]; then
    total_episodes=$(get_integer_input "Enter the total number of episodes to collect: ")
fi

# Get number of episodes per restart if not provided
if [ -z "$episodes_per_restart" ]; then
    episodes_per_restart=$(get_integer_input "Enter the number of episodes to collect before restarting a worker: ")
fi

echo "Number of workers: $num_workers"
echo "Total number of episodes: $total_episodes"
echo "Episodes per restart: $episodes_per_restart"
echo "Append mode: $append_mode"

base_dir="./output/dataset"

# Create base directory if it doesn't exist
mkdir -p "$base_dir"

# Delete existing dataset only if not in append mode
if [ "$append_mode" = false ]; then
    echo "Deleting existing dataset..."
    rm -rf "$base_dir"/*
else
    echo "Appending to existing dataset..."
fi

# Calculate episodes per worker
episodes_per_worker=$((total_episodes / num_workers))
remainder=$((total_episodes % num_workers))

echo "Collecting $total_episodes episodes using $num_workers workers, restarting every $episodes_per_restart episodes."

# Function to run collect_dataset.py with a specific project directory and number of episodes
collect_dataset() {
    worker_id=$1
    episodes=$2
    run_id=$3
    project_dir="${base_dir}/worker_${worker_id}/run_${run_id}"
    mkdir -p "$project_dir"
    
    append_flag=""
    if [ "$append_mode" = true ]; then
        append_flag="--append"
    fi
    
    python3 collect_dataset.py project_dir="${project_dir}" dataset.num_episodes=${episodes} $append_flag $config_overrides
}

# Function to manage a worker with periodic restarts
manage_worker() {
    worker_id=$1
    total_episodes=$2
    
    collected_episodes=0
    run_id=0
    while [ $collected_episodes -lt $total_episodes ]; do
        episodes_to_collect=$((total_episodes - collected_episodes))
        if [ $episodes_to_collect -gt $episodes_per_restart ]; then
            episodes_to_collect=$episodes_per_restart
        fi
        
        collect_dataset $worker_id $episodes_to_collect $run_id
        
        collected_episodes=$((collected_episodes + episodes_to_collect))
        echo "Worker $worker_id has collected $collected_episodes out of $total_episodes episodes so far"
        
        run_id=$((run_id + 1))
        
        if [ $collected_episodes -lt $total_episodes ]; then
            echo "Restarting worker $worker_id"
        fi
    done
}

# Run workers with periodic restarts
for ((i=1; i<=${num_workers}; i++)); do
    worker_episodes=$episodes_per_worker
    if [ $i -le $remainder ]; then
        worker_episodes=$((worker_episodes + 1))
    fi
    manage_worker $i $worker_episodes &
done

# Wait for all background jobs to finish
wait

echo "All dataset collection jobs completed."

# Call the merge_datasets.sh script
./merge_datasets.sh

echo "Script completed. Dataset collection and merging are done."