import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

# python scripts/data_visualizer.py --input_file .\data\owndata\processed\high-speed\high-speed1_processed.csv --save_dir .\data\data_visualization\high-speed

class DataVisualizer:
    def __init__(self, input_file, save_dir='plots', seed=42):
        """
        Initialize the DataVisualizer class.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - save_dir (str): Directory to save the plots.
        - seed (int): Random seed for reproducibility.
        """
        self.input_file = input_file
        self.save_dir = save_dir
        self.seed = seed
        np.random.seed(self.seed)
        self.data = None
        self.canid_col = 0
        self.payload_cols = list(range(1, 9))
        self.label_col = 9
        self.distribution_before = {}

    def read_data(self):
        """Read the CSV dataset and initialize unique value sets."""
        try:
            self.data = pd.read_csv(self.input_file, header=None, delimiter=',')
        except Exception as e:
            print(f"Error reading file: {e}")
            raise

        # Ensure the dataset has at least 10 columns (CANID + 8 payloads + label)
        if self.data.shape[1] < 10:
            raise ValueError("The dataset has fewer than 10 columns. Please check the data format.")

        print(f"Data read successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")

    def capture_distribution(self):
        """Capture the distribution of CANIDs and payloads."""
        # CANID distribution
        canid_counts = self.data[self.canid_col].value_counts().sort_index()
        self.distribution_before['CANID'] = canid_counts

        # Payload distribution
        for col in self.payload_cols:
            payload_counts = self.data[col].value_counts().sort_index()
            self.distribution_before[f'Payload_{col}'] = payload_counts

    def plot_distribution(self):
        """Generate and save distribution plots for CANIDs and payloads."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        for key, counts in self.distribution_before.items():
            plt.figure(figsize=(12, 8))
            sns.barplot(x=counts.index, y=counts.values, palette='viridis')
            plt.title(f'Low-Speed {key} Distribution')
            plt.xlabel(key)
            plt.ylabel('Count')

            # Adjust x-axis tick labels to avoid overlap
            max_ticks = 20  # Maximum number of x-axis ticks to display
            if len(counts) > max_ticks:
                step = (len(counts) // max_ticks + 1)
                plt.xticks(ticks=range(0, len(counts), step), labels=counts.index[::step], rotation=45)
            else:
                plt.xticks(rotation=45)

            plt.tight_layout()
            plot_filename = f"{key}_distribution.png"
            plt.savefig(os.path.join(self.save_dir, plot_filename))
            plt.close()
            print(f"Saved distribution plot: {os.path.join(self.save_dir, plot_filename)}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize CANID or payload distributions.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save distribution plots.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.isfile(args.input_file):
        print(f"Input file does not exist: {args.input_file}")
        return

    visualizer = DataVisualizer(input_file=args.input_file, save_dir=args.save_dir)

    print("Reading data...")
    visualizer.read_data()

    print("Capturing distribution...")
    visualizer.capture_distribution()

    print("Plotting and saving distribution plots...")
    visualizer.plot_distribution()

    print("Visualization completed successfully.")

if __name__ == "__main__":
    main()
