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

# Get number of episodes per restart
episodes_per_restart=$(get_integer_input "Enter the number of episodes to collect before restarting a worker: ")

base_dir="./output"
merge_dir="${base_dir}/dataset"

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
    python collect_dataset.py project_dir="${project_dir}" dataset.num_episodes=${episodes}
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
        
        new_episodes=$(find ${base_dir}/worker_${worker_id}/run_${run_id}/dataset -name "episode_*.pt" 2>/dev/null | wc -l)
        collected_episodes=$((collected_episodes + new_episodes))
        echo "Worker $worker_id has collected $collected_episodes episodes so far"
        
        run_id=$((run_id + 1))
        
        if [ $collected_episodes -lt $total_episodes ]; then
            echo "Restarting worker $worker_id"
        fi
    done
}

# Create merge directory
mkdir -p "${merge_dir}"

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

# Merge datasets with renaming
echo "Merging datasets..."
episode_counter=0
for ((i=1; i<=${num_workers}; i++)); do
    worker_dir="${base_dir}/worker_${i}"
    for run_dir in "$worker_dir"/run_*; do
        if [ -d "$run_dir/dataset" ]; then
            for episode_file in "$run_dir/dataset"/*; do
                if [ -f "$episode_file" ]; then
                    new_name=$(printf "episode_%d.pt" $episode_counter)
                    cp "$episode_file" "${merge_dir}/${new_name}"
                    episode_counter=$((episode_counter + 1))
                fi
            done
        fi
    done
done

echo "Dataset merge completed. Merged dataset is in ${merge_dir}"
echo "Total episodes in merged dataset: $episode_counter"

# Automatically remove individual worker directories
echo "Removing individual worker directories..."
for ((i=1; i<=${num_workers}; i++)); do
    rm -rf "${base_dir}/worker_${i}"
done
echo "Individual worker directories removed."

echo "Script completed. Collected and merged $episode_counter episodes in total."