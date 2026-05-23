import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch import nn
import random
from tqdm import tqdm
from modules.model import AttackDetectionModel_Sincos, AttackDetectionModel_no_pos, AttackDetectionModel_no_attn, AttackDetectionModel_no_conv_pos, AttackDetectionModel_no_embedding_pos
from modules.model import AttackDetectionModel, AttackDetectionModel_Sincos_Fre_EMB, AttackDetectionModel_Sincos_Fre, AttackDetectionModel_LSTM
from modules.model import MambaCAN, MambaCAN_noconv, MambaCAN_noid, MambaCAN_Only, MambaCAN_2Direction, MambaCAN_2Direction_1conv, MambaCAN_2Direction_Fre
import logging
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to compute and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, fold, output_dir="./output"):
    """
    Plots and saves confusion matrix.
    Args:
        y_true (list or numpy array): Ground truth labels.
        y_pred (list or numpy array): Predicted labels.
        class_names (list): List of class names.
        fold (int): Fold index for file naming.
        output_dir (str): Directory to save the confusion matrix image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix for Fold {fold + 1}")
    plt.savefig(f"{output_dir}/confusion_matrix_fold_{fold + 1}_{log_file}.png")
    plt.close()


# Set random seeds
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Setup logging
log_file = "ablation_AttackDetectionModel_no_embedding_pos_acc.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--loss_type", type=str, default='CE', help="Loss function type (MSE, CE)")
parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--n_classes", type=int, default=4, help="Number of output classes")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--k_folds", type=int, default=5, help="Number of K-folds for cross-validation")
opt = parser.parse_args()

# Set device
device = torch.device("cuda:0")

# Load dataset
path = 'data/owndata/merged/all.npy'
source_data = np.load(path)
data = source_data[:, :-1]
label = source_data[:, -1]

# Dataset class
class GetDataset(Dataset):
    def __init__(self, data_root, data_label):
        self.data = torch.tensor(data_root, dtype=torch.float32)
        self.label = torch.tensor(data_label, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# K-Fold Cross Validation
kf = KFold(n_splits=opt.k_folds, shuffle=True, random_state=42)

def safe_division(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

# Training and Evaluation Loop for K-Fold
sum_rec, sum_pre, sum_F1, sum_acc = 0, 0, 0, 0
all_time = []

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    logging.info(f"\nTraining fold {fold + 1}/{opt.k_folds}")

    # Split data into train and validation sets for this fold
    train_data, val_data = data[train_idx], data[val_idx]
    train_label, val_label = label[train_idx], label[val_idx]

    # Create DataLoader for this fold
    torch_data_train = GetDataset(train_data, train_label)
    torch_data_val = GetDataset(val_data, val_label)
    traindataloader = DataLoader(torch_data_train, batch_size=1024, shuffle=True)
    valdataloader = DataLoader(torch_data_val, batch_size=1024, shuffle=False)

    # Initialize the model for each fold
    model = AttackDetectionModel_no_embedding_pos(num_classes=4).to(device)

    # Loss function
    if opt.loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif opt.loss_type == 'CE':
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(opt.n_classes),
            y=np.concatenate([labels.numpy() for _, labels in traindataloader])
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training Loop for the current fold
    for epoch in range(opt.n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        logging.info(f"Epoch [{epoch + 1}/{opt.n_epochs}]")
        train_bar = tqdm(traindataloader, desc="Training", leave=False)
        total_time = 0.0
        num_items = len(train_bar)
        for data_x, data_y in train_bar:
            start_time = time.time()
            data_x, data_y = data_x.to(device), data_y.to(device)
            data_y = data_y.to(torch.long)

            optimizer.zero_grad()
            outputs = model(data_x)
            end_time = time.time()
            total_time += end_time - start_time

            loss = criterion(outputs, data_y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_loss += loss.item() * data_y.size(0)
            train_correct += (predicted == data_y).sum().item()
            train_total += data_y.size(0)

            train_bar.set_postfix(loss=train_loss / train_total, accuracy=train_correct / train_total)

        train_accuracy = train_correct / train_total
        average_time = total_time / num_items
        logging.info(f"Training Loss: {train_loss / train_total:.4f}, Accuracy: {train_accuracy:.4f}")
        logging.info(f"Item time: {average_time:.4f}")
        all_time.append(average_time)

        # Evaluation Loop for the current fold
        model.eval()
        val_correct = 0
        val_total = 0
        num_classes = opt.n_classes
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        all_labels = []
        all_preds = []

        val_bar = tqdm(valdataloader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for data_x, data_y in val_bar:
                data_x, data_y = data_x.to(device), data_y.to(device)
                outputs = model(data_x)
                _, predicted = torch.max(outputs, 1)

                val_correct += (predicted == data_y).sum().item()
                val_total += data_y.size(0)

                all_labels.extend(data_y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                for cls in range(num_classes):
                    tp[cls] += ((predicted == cls) & (data_y == cls)).sum().item()
                    fp[cls] += ((predicted == cls) & (data_y != cls)).sum().item()
                    fn[cls] += ((predicted != cls) & (data_y == cls)).sum().item()

        val_accuracy = val_correct / val_total
        macro_precision = sum([safe_division(tp[cls], tp[cls] + fp[cls]) for cls in range(num_classes)]) / num_classes
        macro_recall = sum([safe_division(tp[cls], tp[cls] + fn[cls]) for cls in range(num_classes)]) / num_classes
        macro_f1 = sum([
            safe_division(
                2 * safe_division(tp[cls], tp[cls] + fp[cls]) * safe_division(tp[cls], tp[cls] + fn[cls]),
                safe_division(tp[cls], tp[cls] + fp[cls]) + safe_division(tp[cls], tp[cls] + fn[cls])
            )
            for cls in range(num_classes)
        ]) / num_classes

        if epoch == opt.n_epochs - 1:  # Last epoch
            sum_pre += macro_precision
            sum_rec += macro_recall
            sum_F1 += macro_f1
            sum_acc += val_accuracy

        logging.info(f"Macro-Averaged Metrics: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1 Score={macro_f1:.4f}, Accuracy={val_accuracy:.4f}")

    # Generate confusion matrix after all validation batches
    plot_confusion_matrix(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        class_names=[f"Class {i}" for i in range(num_classes)],
        fold=fold,
        output_dir="./output"
    )
    # Save the model after each fold
    model_save_path = f"./output/{log_file}_fold{fold + 1}_model.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved at {model_save_path}")

# Final metrics
logging.info(f"Final Metrics: Precision={sum_pre / opt.k_folds:.4f}, Recall={sum_rec / opt.k_folds:.4f}, F1 Score={sum_F1 / opt.k_folds:.4f}, Accuracy={sum_acc / opt.k_folds:.4f}")
all_times_array = np.array(all_time)
logging.info(f"Average item time: {all_times_array.mean():.4f}")