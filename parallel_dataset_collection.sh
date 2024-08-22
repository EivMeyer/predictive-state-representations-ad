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

# Get number of workers
if [ $# -ge 1 ]; then
    num_workers=$1
else
    num_workers=$(get_integer_input "Enter the number of workers: ")
fi

# Get total number of episodes
if [ $# -ge 2 ]; then
    total_episodes=$2
else
    total_episodes=$(get_integer_input "Enter the total number of episodes to collect: ")
fi

base_dir="./output"
merge_dir="${base_dir}/merged_dataset"

# Calculate episodes per worker
episodes_per_worker=$((total_episodes / num_workers))
remainder=$((total_episodes % num_workers))

echo "Collecting $total_episodes episodes using $num_workers workers."

# Function to run collect_dataset.py with a specific project directory and number of episodes
collect_dataset() {
    worker_id=$1
    episodes=$2
    project_dir="${base_dir}/worker_${worker_id}"
    python collect_dataset.py project_dir="${project_dir}" dataset.num_episodes=${episodes}
}

# Create merge directory
mkdir -p "${merge_dir}"

# Run collect_dataset.py in parallel
for ((i=1; i<=${num_workers}; i++)); do
    worker_episodes=$episodes_per_worker
    if [ $i -le $remainder ]; then
        worker_episodes=$((worker_episodes + 1))
    fi
    collect_dataset $i $worker_episodes &
done

# Wait for all background jobs to finish
wait

echo "All dataset collection jobs completed."

# Merge datasets
echo "Merging datasets..."
for ((i=1; i<=${num_workers}; i++)); do
    worker_dir="${base_dir}/worker_${i}/dataset"
    if [ -d "$worker_dir" ]; then
        cp -r "${worker_dir}"/* "${merge_dir}/"
    else
        echo "Warning: Dataset directory for worker ${i} not found."
    fi
done

echo "Dataset merge completed. Merged dataset is in ${merge_dir}"

# Automatically remove individual worker directories
echo "Removing individual worker directories..."
for ((i=1; i<=${num_workers}; i++)); do
    rm -rf "${base_dir}/worker_${i}"
done
echo "Individual worker directories removed."

echo "Script completed. Collected $total_episodes episodes in total."