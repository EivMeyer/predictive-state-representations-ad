#!/bin/bash

# Define the base and merge directories
base_dir="./output"
merge_dir="${base_dir}/dataset"

# Function to merge datasets
merge_datasets() {
    local merge_dir=$1
    local file_counter=0

    echo "Merging datasets from all worker directories into $merge_dir..."
    
    # Create merge directory
    mkdir -p "${merge_dir}"

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
merge_datasets $merge_dir
