#!/bin/bash

# Initialize base directory with default value
base_dir="./output"

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir) base_dir="$2"; shift 2 ;;
        *) echo "Usage: $0 [-o|--output-dir base_directory]" >&2
           exit 1 ;;
    esac
done

# Define the merge directory
merge_dir="${base_dir}/dataset"

# Function to merge datasets
merge_datasets() {
    local merge_dir=$1
    local file_counter=0

    echo "Merging datasets from all worker directories into $merge_dir..."

    # Create merge directory if it doesn't exist
    mkdir -p "${merge_dir}"

    # Find the highest existing batch number
    highest_batch=$(find "${merge_dir}" -name "batch_*.pt" | sed 's/.*batch_\([0-9]*\)\.pt/\1/' | sort -n | tail -n 1)
    if [ -z "$highest_batch" ]; then
        highest_batch=-1
    fi
    file_counter=$((highest_batch + 1))

    # Find all worker directories
    worker_dirs=$(find "$base_dir" -type d -name "worker_*" | sort)

    # Loop through all worker directories and merge files
    for worker_dir in $worker_dirs; do
        for run_dir in "$worker_dir"/run_*; do
            if [ -d "$run_dir/dataset" ]; then
                for file in "$run_dir/dataset"/*; do
                    if [ -f "$file" ]; then
                        new_name=$(printf "batch_%d.pt" $file_counter)
                        cp "$file" "${merge_dir}/${new_name}"
                        file_counter=$((file_counter + 1))
                    fi
                done
            fi
        done
    done

    echo "Dataset merge completed. Merged dataset is in ${merge_dir}"
    echo "Total files in merged dataset: $file_counter"

    # Automatically remove individual worker directories
    echo "Removing individual worker directories..."
    for worker_dir in $worker_dirs; do
        rm -rf "$worker_dir"
    done
    echo "Individual worker directories removed."

    echo "Merge script completed. Collected and merged $file_counter files in total."
}

# Call the merge function
merge_datasets "$merge_dir"