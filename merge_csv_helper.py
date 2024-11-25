import pandas as pd
import numpy as np
import sys
from pathlib import Path

def merge_csv_files(input_pattern, output_file):
    """
    Merges multiple CSV files matching the input pattern into a single CSV file.
    """
    files = list(Path('.').glob(input_pattern))
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return

    # Combine all CSV files into one DataFrame
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged detailed metrics saved to {output_file}")

def merge_and_average_csv(input_pattern, output_file):
    """
    Merges multiple CSV files matching the input pattern and calculates averages for numerical columns.
    """
    files = list(Path('.').glob(input_pattern))
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return

    # Combine all CSV files into one DataFrame
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    # Calculate averages for numerical columns
    avg_df = merged_df.mean(numeric_only=True).to_frame().T
    avg_df["source_files"] = len(files)  # Add metadata on the number of files merged

    avg_df.to_csv(output_file, index=False)
    print(f"Merged and averaged metrics saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: merge_csv_helper.py <mode> <input_pattern> <output_file>")
        print("<mode>: 'merge' or 'average'")
        sys.exit(1)

    mode = sys.argv[1]
    input_pattern = sys.argv[2]
    output_file = sys.argv[3]

    if mode == "merge":
        merge_csv_files(input_pattern, output_file)
    elif mode == "average":
        merge_and_average_csv(input_pattern, output_file)
    else:
        print(f"Invalid mode: {mode}. Use 'merge' or 'average'.")
        sys.exit(1)
