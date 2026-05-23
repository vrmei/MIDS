import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


# python scripts/own_data_process.py batch --input_dir ./data/owndata/origin/high-speed --output_dir ./data/owndata/processed/ --output_file high-speed-merge.csv
# python scripts/own_data_process.py single --file ./data/owndata/origin/high-speed/high-speed4.asc --output_dir ./data/owndata/processed/high-speed
# python scripts/own_data_process.py batch_single --input_dir ./data/owndata/origin/high-speed --output_dir ./data/owndata/processed/high-speed

def process_files_individually(input_dir, output_dir):
    """
    Process all .asc files in a directory individually.
    
    Args:
        input_dir (str): Input folder path containing .asc files.
        output_dir (str): Output folder path for processed CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_pattern = os.path.join(input_dir, '*.asc')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"Warning: No .asc files found in directory {input_dir}.")
        return
    
    print(f"Found {len(file_list)} .asc files. Starting individual processing...")
    
    for file_path in tqdm(file_list, desc="Processing files individually"):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_processed.csv"
        process_single_file(file_path, output_dir, output_filename)
    
    print(f"All files have been processed and saved in {output_dir}")

def process_file(file_path):
    """
    Process a single .asc file and return the processed data as a list of lists.
    
    Args:
        file_path (str): Path to the .asc file.
    
    Returns:
        List[List[int]]: Processed data, where each sublist represents a row of data.
    """
    totaldata = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(4):
                next(f)
            for line_number, line in enumerate(tqdm(f, desc=f"Processing {os.path.basename(file_path)}"), 1):
                tempstr = line.strip().split()
                if len(tempstr) < 14:
                    tempstr += ['00'] * (14 - len(tempstr))
                try:
                    x = int(tempstr[2], 16)
                    payload = [int(tempstr[i], 16) for i in range(6, 14)]
                    currow = [x] + payload + [0]
                    totaldata.append(currow)
                except ValueError:
                    print(f"Warning: Invalid hexadecimal number at line {line_number + 4} in file {file_path}, skipping.")
                except IndexError:
                    print(f"Warning: Insufficient fields at line {line_number + 4} in file {file_path}, skipping.")
    except Exception as e:
        print(f"Error: Unable to read file {file_path}, error: {e}")
    
    return totaldata

def process_all_files(input_dir, output_dir, output_filename='merged_data.csv'):
    """
    Process all .asc files in a directory and save the merged result as a single CSV file.
    
    Args:
        input_dir (str): Input folder path containing .asc files.
        output_dir (str): Output folder path for the merged CSV file.
        output_filename (str): Name of the merged output CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_pattern = os.path.join(input_dir, '*.asc')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"Warning: No .asc files found in directory {input_dir}.")
        return
    
    print(f"Found {len(file_list)} .asc files. Starting processing...")
    all_data = []
    
    for file_path in tqdm(file_list, desc="Processing all files"):
        file_data = process_file(file_path)
        all_data.extend(file_data)
    
    if not all_data:
        print("Warning: No data was processed or merged.")
        return
    
    temparray = np.array(all_data, dtype=np.int16)
    output_path = os.path.join(output_dir, output_filename)
    try:
        np.savetxt(output_path, temparray, fmt='%d', delimiter=',')
        print(f"Data from all files successfully merged and saved to {output_path}")
    except Exception as e:
        print(f"Error: Unable to save merged data to {output_path}, error: {e}")
    
    print("Sample data (first 5 rows):")
    print(temparray[:5])

def process_single_file(file_path, output_dir, output_filename=None):
    """
    Process a single .asc file and save the result as a CSV file.
    
    Args:
        file_path (str): Path to the .asc file.
        output_dir (str): Output folder path for the processed CSV file.
        output_filename (str, optional): Output CSV file name. If not provided, it will be generated based on the input file name.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing single file: {file_path}")
    file_data = process_file(file_path)
    
    if not file_data:
        print("Warning: No data was processed.")
        return
    
    temparray = np.array(file_data, dtype=np.int16)
    if not output_filename:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_processed.csv"
    
    output_path = os.path.join(output_dir, output_filename)
    try:
        np.savetxt(output_path, temparray, fmt='%d', delimiter=',')
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error: Unable to save data to {output_path}, error: {e}")
    
    print("Sample data (first 5 rows):")
    print(temparray[:5])

def main():
    parser = argparse.ArgumentParser(description="Process .asc files and convert them to CSV format.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    parser_batch = subparsers.add_parser('batch', help='Process multiple .asc files and merge into a single output file')
    parser_batch.add_argument('--input_dir', type=str, required=True, help='Input folder path containing .asc files')
    parser_batch.add_argument('--output_dir', type=str, required=True, help='Output folder path')
    parser_batch.add_argument('--output_file', type=str, default='merged_data.csv', help='Merged output CSV file name')

    parser_single = subparsers.add_parser('single', help='Process a single .asc file')
    parser_single.add_argument('--file', type=str, required=True, help='Path to the .asc file to process')
    parser_single.add_argument('--output_dir', type=str, required=True, help='Output folder path')
    parser_single.add_argument('--output_file', type=str, default=None, help='Output CSV file name (optional)')

    parser_individual = subparsers.add_parser('batch_single', help='Process all files in a directory individually')
    parser_individual.add_argument('--input_dir', type=str, required=True, help='Input folder path containing .asc files')
    parser_individual.add_argument('--output_dir', type=str, required=True, help='Output folder path')

    args = parser.parse_args()

    if args.command == 'batch':
        process_all_files(args.input_dir, args.output_dir, args.output_file)
    elif args.command == 'single':
        process_single_file(args.file, args.output_dir, args.output_file)
    elif args.command == 'batch_single':
        process_files_individually(args.input_dir, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
