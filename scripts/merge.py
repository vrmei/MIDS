import pandas as pd
import os
import argparse

# python scripts/merge.py --input_dir data/owndata/reshape/high-speed/ --output_file data/owndata/merged/high-speed_merged.csv

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Merge all CSV files in a directory into a single file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing CSV files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the merged CSV file.")
    return parser.parse_args()

def merge_files_in_directory(input_dir, output_file):
    """
    Merge all CSV files in a given directory into a single CSV file.

    Parameters:
    - input_dir (str): Path to the input directory containing CSV files.
    - output_file (str): Path to save the merged CSV file.
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    merged_data = []
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            try:
                data = pd.read_csv(file_path, header=None)
                merged_data.append(data)
                print(f"Loaded file: {file_name}")
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    
    if not merged_data:
        print("No valid CSV files found in the directory.")
        return

    # Concatenate all loaded files
    merged_df = pd.concat(merged_data, ignore_index=True)
    
    # Save the merged DataFrame to the output file
    try:
        merged_df.to_csv(output_file, header=False, index=False)
        print(f"Merged data saved to {output_file}")
    except Exception as e:
        print(f"Error saving merged data: {e}")

def main():
    args = parse_arguments()

    print("Merging files from the input directory...")
    merge_files_in_directory(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
