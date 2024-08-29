#!/bin/bash

# Check if directory and size threshold are provided, otherwise use defaults
DIR="${1:-.}"
THRESHOLD="${2:-100M}"  # Default threshold is 100M (100 Megabytes)

# Function to print summary for each directory
print_summary() {
  local folder=$1
  # Get disk usage in human-readable format and in bytes
  local size=$(du -sh "$folder" 2>/dev/null | cut -f1)
  local size_bytes=$(du -sb "$folder" 2>/dev/null | cut -f1)
  # Count number of subfiles (excluding directories)
  local file_count=$(find "$folder" -type f | wc -l)
  # Get last modified date of the directory
  local last_modified=$(stat -c %y "$folder" 2>/dev/null | cut -d' ' -f1)

  # Compare size in bytes to the threshold converted to bytes
  local threshold_bytes=$(numfmt --from=iec "$THRESHOLD")

  # Only print if size is above the threshold
  if [ "$size_bytes" -gt "$threshold_bytes" ]; then
    echo "Directory: $folder"
    echo "Size: $size"
    echo "Number of files: $file_count"
    echo "Last modified: $last_modified"
    echo "---------------------------------------------"
  fi
}

# Export the function to use with find's -exec
export -f print_summary

# Export the threshold variable for the function
export THRESHOLD

# Find all directories and subdirectories and print their summaries if above threshold
find "$DIR" -type d -exec bash -c 'print_summary "$0"' {} \;
