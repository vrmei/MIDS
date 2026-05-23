import pandas as pd
import numpy as np
import argparse
import os

# Example usage:
# python scripts/reshape.py --input_dir data/owndata/attackdata/high-speed/ --output_file data/owndata/reshape/high-speed/merged_reshaped.csv --group_size 100
# python scripts/reshape.py --input_dir data/owndata/attackdata/standby/ --output_file data/owndata/reshape/standby/standby.csv --group_size 100

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reshape CAN data by grouping rows.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing CSV files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the reshaped CSV file.")
    parser.add_argument("--group_size", type=int, default=9, help="Number of rows to group.")
    parser.add_argument("--discard_incomplete", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to discard incomplete groups. Use True or False.")
    return parser.parse_args()

def merge_files_in_directory(input_dir):
    """
    Merge all CSV files in a given directory into a single DataFrame.

    Parameters:
    - input_dir (str): Path to the input directory containing CSV files.

    Returns:
    - pd.DataFrame: Combined DataFrame with all files' content.
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return None

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
        return None

    return pd.concat(merged_data, ignore_index=True)

def reshape_data(data, output_file, group_size=9, discard_incomplete=True):
    """
    Reshape the data by grouping every 'group_size' rows into a single row.

    Parameters:
    - data (pd.DataFrame): Input data to reshape.
    - output_file (str): Path to save the reshaped CSV file.
    - group_size (int): Number of rows to group.
    - discard_incomplete (bool): Whether to discard incomplete groups.
    """
    total_rows = data.shape[0]
    num_groups = total_rows // group_size
    remainder = total_rows % group_size

    if remainder != 0 and not discard_incomplete:
        num_groups += 1
    elif remainder != 0 and discard_incomplete:
        print(f"Discarding the last {remainder} incomplete rows.")

    reshaped_data = []
    reshaped_labels = []

    for group in range(num_groups):
        start_idx = group * group_size
        end_idx = start_idx + group_size

        if end_idx > total_rows:
            if discard_incomplete:
                break
            else:
                end_idx = total_rows

        group_data = data.iloc[start_idx:end_idx]

        # Extract data fields and labels
        data_fields = group_data.iloc[:, :-1].values.flatten()
        labels = group_data.iloc[:, -1].values

        # Aggregate labels:
        non_zero_labels = labels[labels != 0]
        if len(non_zero_labels) > 0:
            aggregated_label = int(non_zero_labels.max())
        else:
            aggregated_label = 0

        reshaped_data.append(data_fields)
        reshaped_labels.append(aggregated_label)

        if (group + 1) % 1000 == 0:
            print(f"Processed {group + 1} groups...")

    # Convert to DataFrame
    reshaped_df = pd.DataFrame(reshaped_data)
    reshaped_df['label'] = reshaped_labels

    # Save to CSV
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        reshaped_df.to_csv(output_file, header=False, index=False)
        print(f"Reshaped data saved to {output_file}")
        print(f"Total {len(reshaped_df)} groups processed.")
    except Exception as e:
        print(f"Error saving reshaped data: {e}")

def main():
    args = parse_arguments()

    print("Merging files from the input directory...")
    data = merge_files_in_directory(args.input_dir)
    if data is None:
        print("No data to process. Exiting.")
        return

    print("Reshaping the merged data...")
    reshape_data(data, args.output_file, args.group_size, args.discard_incomplete)

if __name__ == "__main__":
    main()
