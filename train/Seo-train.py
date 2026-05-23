import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

import torch.utils
from torch.utils import data
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm  
from modules.model import TransformerClassifier_WithPositionalEncoding, CNN, KNN, MLP, LSTMClassifier, BERT  
import random


from transformers import BertTokenizer

# CUDA_VISIBLE_DEVICES=1 python train/train.py

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--loss_type", type=str, default='CE', help="the loss func(MSE, CE)")
parser.add_argument("--model_type", type=str, default='MLP', help="which model will be used (KNN, CNN, BERT ,Attn, DecisionTree, MLP, LSTM)")
parser.add_argument("--data_src", type=str, default='own', help="the dataset name")
parser.add_argument("--attack_type", type=str, default='Gear', help="which attack in: DoS, Fuzz, or Gear")
parser.add_argument("--propotion", type=float, default=0.8, help="the count of train divided by the count of whole")
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--freeze_bert", action='store_true', help="Whether to freeze BERT parameters")
opt = parser.parse_args()
print(opt)

path = 'data/owndata/attackdata/1_x_2_all.csv'
if opt.data_src == 'Seo':
    if opt.attack_type == 'DoS':
        source_data = pd.read_csv('data/CNN_data/DoS_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/DoS_label.csv')

    elif opt.attack_type == 'Fuzz':
        source_data = pd.read_csv('data/CNN_data/Fuzz_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/Fuzz_label.csv')

    elif opt.attack_type == 'Gear':
        source_data = pd.read_csv('data/CNN_data/gear_data.csv')
        datalen = int(opt.propotion * len(source_data))
        source_label = pd.read_csv('data/CNN_data/gear_label.csv')

elif opt.data_src == 'own':
    source_data = pd.read_csv(path)
    datalen = int(opt.propotion * len(source_data))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GetDataset(data.Dataset):
    def __init__(self, data_root, data_label, tokenizer=None, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data_root.values.astype(str) if tokenizer else data_root.values.astype(np.float32)
        self.label = data_label.values.astype(np.int64)

    def __getitem__(self, index):
        if self.tokenizer:
            encoded = self.tokenizer.encode_plus(
                self.data[index],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoded['input_ids'].squeeze(0)  # [max_length]
            attention_mask = encoded['attention_mask'].squeeze(0)  # [max_length]
            label = self.label[index]
            return input_ids, attention_mask, label
        else:
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
    train_data = source_data.iloc[:datalen,:]
    test_data = source_data.iloc[datalen:,:]
    train_label = train_data.iloc[:, -1]     
    test_label = test_data.iloc[:, -1]     
    train_data = train_data.iloc[:, :-1] 
    test_data = test_data.iloc[:, :-1] 


torch_data_train = GetDataset(train_data, train_label)
torch_data_test = GetDataset(test_data, test_label)

traindataloader = data.DataLoader(torch_data_train, batch_size=64, shuffle=True)
testdataloader = data.DataLoader(torch_data_test, batch_size=64, shuffle=True)

output_str = f"""
Model Type: {opt.model_type}
Data Source: {opt.data_src}
Data Path: {path}
Attack Type: {opt.attack_type}
Training Proportion: {opt.propotion}
Number of Classes: {opt.n_classes}
"""
print(output_str)

output_str += '\n\n'

if opt.model_type == 'CNN':
    model = CNN().to(device)

elif opt.model_type == 'KNN':
    model = KNN()
    # Convert training and testing data to NumPy arrays
    train_data_np = np.array(train_data.values, dtype='float32')
    test_data_np = np.array(test_data.values, dtype='float32')
    train_label_np = np.array(train_label.values, dtype='int64').flatten()
    test_label_np = np.array(test_label.values, dtype='int64').flatten()

    X_train = train_data_np
    y_train = train_label_np  # Labels

    X_test = test_data_np
    y_test = test_label_np  # Test labels

    # Initialize KNN model
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train KNN model
    knn.fit(X_train, y_train)

    # Predict using the test data
    predictions = knn.predict(X_test)

    # Calculate confusion matrix components
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))

    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    FNR = fn / (fn + tp) if (fn + tp) != 0 else 0
    ER = (fp + fn) / total
    Recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    Precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0

    # Output results
    output_str += f'Final Accuracy: {accuracy * 100:.2f}%\n'
    output_str += f'False Negative Rate (FNR): {FNR:.4f}\n'
    output_str += f'Error Rate (ER): {ER:.4f}\n'
    output_str += f'Recall: {Recall:.4f}\n'
    output_str += f'F1 Score: {F1:.4f}\n'
    with open("log.txt", "a") as f:
        f.write(output_str)
    print(f"Final Accuracy of KNN: {accuracy}")
    print(f"Data path: {path}")
    print(f"False Negative Rate (FNR): {FNR}")
    print(f"Error Rate (ER): {ER}")
    print(f"Recall: {Recall}")
    print(f"F1 Score: {F1}")
    exit()

elif opt.model_type == 'DecisionTree':
    # Convert training and testing data to NumPy arrays
    train_data_np = np.array(train_data.values, dtype='float32')
    test_data_np = np.array(test_data.values, dtype='float32')
    train_label_np = np.array(train_label.values, dtype='int64').flatten()
    test_label_np = np.array(test_label.values, dtype='int64').flatten()

    X_train = train_data_np
    y_train = train_label_np  # Labels

    X_test = test_data_np
    y_test = test_label_np  # Test labels

    # Initialize Decision Tree model
    dt_model = DecisionTreeClassifier()

    # Train Decision Tree model
    dt_model.fit(X_train, y_train)

    # Predict using the test data
    predictions = dt_model.predict(X_test)

    # Calculate confusion matrix components
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))

    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    FNR = fn / (fn + tp) if (fn + tp) != 0 else 0
    ER = (fp + fn) / total
    Recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    Precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0

    # Output results
    output_str += f'Final Accuracy: {accuracy * 100:.2f}%\n'
    output_str += f'False Negative Rate (FNR): {FNR:.4f}\n'
    output_str += f'Error Rate (ER): {ER:.4f}\n'
    output_str += f'Recall: {Recall:.4f}\n'
    output_str += f'F1 Score: {F1:.4f}\n'
    with open("log.txt", "a") as f:
        f.write(output_str)
    print(f"Data path: {path}")
    print(f"Final Accuracy of Decision Tree: {accuracy}")
    print(f"False Negative Rate (FNR): {FNR}")
    print(f"Error Rate (ER): {ER}")
    print(f"Recall: {Recall}")
    print(f"F1 Score: {F1}")
    exit()

elif opt.model_type == 'MLP':
    # Assuming MLP is defined in modules.model
    model = MLP(input_dim=9, hidden_dim=4096, num_classes=opt.n_classes).to(device)

elif opt.model_type == 'LSTM':
    # Assuming LSTMClassifier is defined in modules.model
    model = LSTMClassifier(input_dim=144, hidden_dim=64, num_classes=opt.n_classes).to(device)

elif opt.model_type == 'Attn':
    model = TransformerClassifier_WithPositionalEncoding(input_dim=1, num_heads=8, num_layers=4, hidden_dim=1024, num_classes=opt.n_classes).to(device)

elif opt.model_type == 'BERT':
    model = BERT(input_dim=1, num_heads=8, num_layers=4, hidden_dim=128, num_classes=opt.n_classes).to(device)


else:
    raise ValueError("Invalid model_type specified.")

if opt.loss_type == 'MSE':
    criterion = torch.nn.MSELoss()
elif opt.loss_type == 'CE':
    if opt.data_src == 'own':
        weights = [1.0, 1.0]
        class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

test_acc = []
train_loss = []
test_loss = []
train_epochs_loss = []
test_epochs_loss = []
acc, nums = 0, 0
maxF1 = 0
for epoch in range(opt.n_epochs):
    acc = nums = 0
    train_epoch_loss = []
    model.train()
    for idx, batch in enumerate(traindataloader):
        try:
            data_x, data_y = batch
            batch_size = data_x.size(0)
            if opt.model_type == 'CNN':
                data_x = data_x.view(batch_size, 1, 12, 12)
            elif opt.model_type == 'LSTM':
                data_x = data_x.view(batch_size, -1, data_x.shape[1])  # Adjust shape for LSTM
            elif opt.model_type == 'Attn' or opt.model_type == 'BERT':
                data_x = data_x.view(batch_size, data_x.shape[1], 1) 
            data_x = data_x.to(torch.float32).to(device)
            if opt.data_src == 'Seo':
                data_y = data_y.squeeze(1).to(torch.long)
            data_y = data_y.to(device)
        except Exception as e:
            print(f"Error processing batch {idx}: {e}")
            continue

        outputs = model(data_x)
        optimizer.zero_grad()
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()

        if opt.model_type == 'BERT':
            _, predicts = torch.max(outputs, 1)
            acc += (predicts == data_y).sum().cpu()
            nums += data_y.size(0)
        else:
            _, predicts = torch.max(outputs, 1)
            acc += (predicts == data_y).sum().cpu()
            nums += data_y.size()[0]

        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx % max(1, (len(traindataloader)//100)) == 0:
            print("epoch= {}/{}, {}/{} of train, loss={}".format(
                epoch, opt.n_epochs, idx, len(traindataloader), loss.item()))
    print("Training Accuracy:", 100 * acc / nums)
    train_epochs_loss.append(np.average(train_epoch_loss))
    acc = nums = 0

    # Initialize confusion matrix components
    tp, tn, fp, fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(testdataloader):
            try:
                data_x, data_y = batch
                batch_size = data_x.size(0)
                if opt.model_type == 'CNN':
                    data_x = data_x.view(batch_size, 1, 12, 12)
                elif opt.model_type == 'LSTM':
                    data_x = data_x.view(batch_size, -1, data_x.shape[1])  # Adjust shape for LSTM
                elif opt.model_type == 'Attn' or opt.model_type == 'BERT':
                    data_x = data_x.view(batch_size, data_x.shape[1], 1) 
                data_x = data_x.to(torch.float32).to(device)
                if opt.data_src == 'Seo':
                    data_y = data_y.squeeze(1).to(torch.long)
                data_y = data_y.to(device)
            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                continue

            outputs = model(data_x)
            loss = criterion(outputs, data_y)
            test_epochs_loss.append(loss.item())
            test_loss.append(loss.item())

            _, predicts = torch.max(outputs, 1)
            acc += (predicts == data_y).sum().cpu()
            nums += data_y.size(0)

            # Calculate confusion matrix components
            tp += ((predicts == 1) & (data_y == 1)).sum().item()
            tn += ((predicts == 0) & (data_y == 0)).sum().item()
            fp += ((predicts == 1) & (data_y == 0)).sum().item()
            fn += ((predicts == 0) & (data_y == 1)).sum().item()

    # Calculate metrics
    total = tp + tn + fp + fn
    FNR = fn / (fn + tp) if (fn + tp) != 0 else 0
    ER = (fp + fn) / total
    Recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    Precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) != 0 else 0
    accuracy = (tp + tn) / total

    print("epoch= {}/{}, {}/{} of test, acc=".format(
        epoch, opt.n_epochs, idx, len(testdataloader)), "%.4f" % float(acc/nums))
    print(f"Data path: {path}")
    print(f"False Negative Rate (FNR): {FNR:.4f}")
    print(f"Error Rate (ER): {ER:.4f}")
    print(f"Recall: {Recall:.4f}")
    print(f"F1 Score: {F1:.4f}")
    if F1 > maxF1:
        maxF1 = F1
        if opt.model_type == 'BERT':
            torch.save(model.state_dict(), f"./output/{opt.model_type}_{opt.data_src}_model_F1_{F1:.2f}.pth")
        else:
            torch.save(model.state_dict(), f"./output/{opt.model_type}{opt.data_src}modelF1{F1:.2F}.pth")
    test_epochs_loss.append(np.average(test_epochs_loss))
    test_acc.append((acc/nums))

output_str = f"""
Model Type: {opt.model_type}
Data Source: {opt.data_src}
Attack Type: {opt.attack_type}
Training Proportion: {opt.propotion}
Number of Epochs: {opt.n_epochs}
Number of Classes: {opt.n_classes}
Learning Rate: {opt.lr}

Final Accuracy: {test_acc}
Max Accuracy: {max(test_acc)}
Mean Accuracy: {sum(test_acc) / len(test_acc)}
False Negative Rate (FNR): {FNR:.4f}
Error Rate (ER): {ER:.4f}
Recall: {Recall:.4f}
F1 Score: {F1:.4f}
"""

with open("log.txt", "a") as f:
    f.write(output_str)

print(test_acc, "max:", max(test_acc), "   mean:", sum(test_acc) / len(test_acc))
print(f"False Negative Rate (FNR): {FNR}")
print(f"Error Rate (ER): {ER}")
print(f"Recall: {Recall}")
print(f"F1 Score: {F1}")
if opt.model_type == 'BERT':
    torch.save(model.state_dict(), f"./output/{opt.model_type}_{opt.data_src}_model.pth")
else:
    torch.save(model.state_dict(), f"./output/{opt.model_type}{opt.data_src}model.pth")
