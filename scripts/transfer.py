import numpy as np
import pandas as pd

def csv_to_npy(input_csv, output_npy):
    """
    Convert a CSV file to a NumPy array and save it as an .npy file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_npy (str): Path to save the output .npy file.
    """
    try:
        data = pd.read_csv(input_csv)
        
        data_array = data.values
        
        np.save(output_npy, data_array)
        print(f"Successfully converted {input_csv} to {output_npy}")
    except Exception as e:
        print(f"Error converting {input_csv} to .npy: {e}")

if __name__ == "__main__":
    input_csv_path = "data/owndata/merged/all.csv"  
    output_npy_path = "data/owndata/merged/all.npy" 
    csv_to_npy(input_csv_path, output_npy_path)
