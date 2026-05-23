import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from collections import Counter

import torch
from torch.utils import data
from tqdm import tqdm
import random

from modules.model import TransformerClassifier_WithPositionalEncoding, CNN, KNN, MLP  # Added MLP

# Fix random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Example usage:
# python train/eval.py --model_path output/AttnSeomodel.pth

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
parser.add_argument("--model_type", type=str, default='MLP', help="Which model to use (KNN, CNN, Attn, SVM, MLP)")
parser.add_argument("--data_src", type=str, default='own', help="The dataset name")
parser.add_argument("--attack_type", type=str, default='DoS', help="Which attack in: DoS, Fuzz, or Gear")
parser.add_argument("--propotion", type=float, default=0.8, help="Train proportion of the whole dataset")
parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
opt = parser.parse_args()
print(opt)

# Load data
if opt.data_src == 'Seo':
    if opt.attack_type == 'DoS': 
        source_data = pd.read_csv('data/CNN_data/DoS_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/DoS_label.csv')

    elif opt.attack_type == 'Fuzz': 
        source_data = pd.read_csv('data/CNN_data/Fuzz_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/Fuzz_label.csv')

    elif opt.attack_type == 'gear': 
        source_data = pd.read_csv('data/CNN_data/gear_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/gear_label.csv')

elif opt.data_src == 'own':
    source_data = pd.read_csv('data/owndata/attackdata/1_x_2_280.csv')
    datalen = int(opt.propotion * len(source_data))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GetDataset(data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root.values
        self.label = data_label.values

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label
    
    def __len__(self):
        return len(self.data)

if opt.data_src == 'Seo':
    train_data = source_data.iloc[:datalen, :]
    train_label = source_label.iloc[:datalen, :]
    test_data = source_data.iloc[datalen:, :]
    test_label = source_label.iloc[datalen:, :]
elif opt.data_src == 'own':
    train_data = source_data.iloc[:datalen, :]
    test_data = source_data.iloc[datalen:, :]
    train_label = train_data.iloc[:, -1]     # Last column as label
    test_label = test_data.iloc[:, -1]       # Last column as label
    train_data = train_data.iloc[:, :-1]     # All columns except last as data
    test_data = test_data.iloc[:, :-1]       # All columns except last as data

source_label = source_data.iloc[:, -1]
torch_data_test = GetDataset(test_data, test_label)
testdataloader = data.DataLoader(torch_data_test, batch_size=32, shuffle=False)
torch_data_train = GetDataset(train_data, train_label)
traindataloader = data.DataLoader(torch_data_train, batch_size=32, shuffle=False)

# Load model
if opt.model_type == 'CNN':
    model = CNN().to(device)
    model.load_state_dict(torch.load(opt.model_path))  # Load model parameters
    model.to(device)  # Move model to device
elif opt.model_type == 'KNN':
    model = KNeighborsClassifier()
elif opt.model_type == 'Attn':
    model = TransformerClassifier_WithPositionalEncoding(
        input_dim=1, 
        num_heads=8, 
        num_layers=4, 
        hidden_dim=128, 
        max_seq_len=500,
        num_classes=opt.n_classes
    )  # Replace with actual model class
    model.load_state_dict(torch.load(opt.model_path))  # Load model parameters
    model.to(device)  # Move model to device
elif opt.model_type == 'SVM':
    model = SVC(kernel='linear', C=1.0)
elif opt.model_type == 'MLP':
    input_dim = train_data.shape[1]
    hidden_dim = 4096  # You can adjust this
    num_classes = opt.n_classes
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(opt.model_path))  # Load model parameters
    model.to(device)  # Move model to device
else:
    raise ValueError("Invalid model_type specified.")

# Evaluation process
if opt.model_type in ['KNN', 'SVM']:
    train_data_np = np.array(train_data.values, dtype='float32')
    test_data_np = np.array(test_data.values, dtype='float32')
    train_label_np = np.array(train_label.values, dtype='float32').flatten()
    test_label_np = np.array(test_label.values, dtype='float32').flatten()

    if opt.model_type == 'KNN':
        model.fit(train_data_np, train_label_np)
    elif opt.model_type == 'SVM':
        model.fit(train_data_np, train_label_np)

    # Use tqdm to display prediction progress
    predictions = []
    for i in tqdm(range(len(test_data_np)), desc="Evaluating"):
        prediction = model.predict([test_data_np[i]])
        predictions.append(prediction[0])

    predictions = np.array(predictions)
    accuracy = accuracy_score(test_label_np, predictions)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

else:
    # For other models (CNN, Transformer, MLP)
    model.eval()
    acc, nums = 0, 0

    with torch.no_grad():
        for idx, (data_x, data_y) in enumerate(testdataloader):
            batch_size = data_x.size(0)
            if opt.model_type == 'CNN':
                data_x = data_x.view(batch_size, 1, 9, 9)
            elif opt.model_type == 'LSTM':
                data_x = data_x.view(batch_size, -1, data_x.shape[1])  # Adjust shape for LSTM
            elif opt.model_type == 'Attn':
                data_x = data_x.view(batch_size, data_x.shape[1], 1) 
            elif opt.model_type == 'MLP':
                data_x = data_x.view(batch_size, -1)  # Flatten if necessary
            data_x = data_x.to(torch.float32).to(device)
            data_y = data_y.to(torch.long).to(device)

            outputs = model(data_x)
            if opt.model_type in ['CNN', 'MLP', 'Attn']:
                if opt.n_classes == 1:
                    # Binary classification with sigmoid
                    preds = torch.sigmoid(outputs).squeeze()
                    predicts = (preds > 0.5).long()
                else:
                    # Multi-class classification with softmax
                    _, predicts = torch.max(outputs, 1)
            else:
                # Handle other possible model types
                _, predicts = torch.max(outputs, 1)

            acc += (predicts == data_y).sum().item()
            nums += data_y.size(0)

    accuracy = acc / nums
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

# Prepare output string
output_str = f"""
Model Type: {opt.model_type}
Data Source: {opt.data_src}
Attack Type: {opt.attack_type}
Final Accuracy: {accuracy * 100:.2f}%
"""

# Write evaluation results to a file
with open("eval_log.txt", "a") as f:
    f.write(output_str)

print(output_str)
