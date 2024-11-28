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
config_overrides=()
base_dir="."

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w) num_workers="$2"; shift 2 ;;
        -e) total_episodes="$2"; shift 2 ;;
        -o|--output-dir) base_dir="$2"; shift 2 ;;
        --) shift; break ;;  # End of options
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [-w num_workers] [-e total_episodes] [-o|--output-dir base_directory] [CONFIG_OVERRIDES]" >&2
            exit 1 ;;
        *) break ;;  # First non-option argument
    esac
done

# Capture all remaining arguments as config overrides
config_overrides=("$@")

# Get number of workers if not provided
if [ -z "$num_workers" ]; then
    num_workers=$(get_integer_input "Enter the number of workers: ")
fi

# Get total number of episodes if not provided
if [ -z "$total_episodes" ]; then
    total_episodes=$(get_integer_input "Enter the total number of episodes to evaluate: ")
fi

eval_dir="$base_dir/evaluation"
mkdir -p "$eval_dir"

echo "Number of workers: $num_workers"
echo "Total number of episodes: $total_episodes"
echo "Output directory: $eval_dir"

# Calculate episodes per worker
episodes_per_worker=$((total_episodes / num_workers))
remainder=$((total_episodes % num_workers))

echo "Evaluating $total_episodes episodes using $num_workers workers."

# Function to run evaluate_rl_agent.py with specific parameters
evaluate_model() {
    worker_id=$1
    episodes=$2
    worker_dir="${eval_dir}/worker_${worker_id}"
    mkdir -p "$worker_dir"

    python3 evaluate_rl_agent.py \
        evaluation_dir="${worker_dir}" \
        evaluation.num_episodes=${episodes} \
        "${config_overrides[@]}"
}

# Run workers in parallel
for ((i=1; i<=${num_workers}; i++)); do
    worker_episodes=$episodes_per_worker
    if [ $i -le $remainder ]; then
        worker_episodes=$((worker_episodes + 1))
    fi
    evaluate_model $i $worker_episodes &
done

# Wait for all background jobs to finish
wait

echo "All evaluation jobs completed."

# Merge results
echo "Merging evaluation results..."

# Use Python to merge detailed metrics
python3 merge_csv_helper.py merge "${eval_dir}/worker_*/detailed_metrics_*.csv" "${eval_dir}/merged_detailed_metrics.csv"

# Use Python to merge and average aggregate metrics
python3 merge_csv_helper.py average "${eval_dir}/worker_*/aggregate_metrics_*.csv" "${eval_dir}/merged_aggregate_metrics.csv"

echo "Results merged successfully:"
echo "- Detailed metrics: ${eval_dir}/merged_detailed_metrics.csv"
echo "- Aggregate metrics: ${eval_dir}/merged_aggregate_metrics.csv"

# Clean up worker directories
for ((i=1; i<=${num_workers}; i++)); do
    rm -rf "${eval_dir}/worker_${i}"
    echo "Removed worker directory: ${eval_dir}/worker_${i}"
done
echo "All worker directories cleaned up."

echo "Parallel evaluation completed successfully."
