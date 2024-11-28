import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json

def save_as_json(dataframe, output_file):
    """
    Saves a DataFrame as a JSON file.
    """
    json_file = output_file.replace('.csv', '.json')
    dataframe.to_json(json_file, orient="records", indent=4)
    print(f"JSON file saved as {json_file}")

def preview_dataframe(df, num_rows=5):
    """
    Prints a readable preview of the DataFrame.
    """
    print("\n==== Preview of the Data ====")
    print(df.head(num_rows).to_json(orient="records", indent=4))

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
    preview_dataframe(merged_df)  # Show a readable preview

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

    all_columns = set().union(*[df.columns for df in dataframes])  # Collect all column names
    aligned_dfs = [df.reindex(columns=all_columns, fill_value=0) for df in dataframes]  # Align columns and fill missing with 0
    merged_df = pd.concat(aligned_dfs, ignore_index=True)

    # Calculate averages for numerical columns
    avg_df = merged_df.mean(numeric_only=True).to_frame().T
    avg_df["source_files"] = len(files)  # Add metadata on the number of files merged

    save_as_json(avg_df, output_file)  # Save as JSON
    print(f"Merged and averaged metrics saved to {output_file}")
    preview_dataframe(avg_df)  # Show a readable preview

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
