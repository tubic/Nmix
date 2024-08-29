import os
import csv
import torch
import argparse
import numpy as np
from utils import *
from model import RNASeqClassifier, Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import pandas as pd

# Create ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--base', type=str, default='A')
parser.add_argument('--ratio', type=int, default=1)
parser.add_argument('--weight_factor', type=float, default=1.0)
args = parser.parse_args()

# Set seed and device
seed = 42
set_seed(seed)
epochs = 500
patience = 10
device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device("cpu")

# Path settings
result_dir = f'../train_result/ratio_{args.ratio}/{args.base}'
model_dir = f'../train_model/ratio_{args.ratio}/{args.base}'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Variable initialization
train_metrics_file = '../train_metrics.csv'

# Explicitly define performance metric field names
metric_names = ['ACC', 'SEN', 'SPE', 'PRE', 'MCC', 'AUROC', 'F1']

# Function: Check if file is empty and write header
def init_csv_file(filepath, fieldnames):
    is_empty = not os.path.exists(filepath) or os.stat(filepath).st_size == 0
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_empty:
            writer.writeheader()

# Initialize CSV file
train_fieldnames = ['ratio', 'base', 'weight_factor', 'fold'] + metric_names
init_csv_file(train_metrics_file, train_fieldnames)

all_metrics = []
weight_model_dir = f"{model_dir}/weight_factor_{args.weight_factor}"
weight_result_dir = f"{result_dir}/weight_factor_{args.weight_factor}"
os.makedirs(weight_model_dir, exist_ok=True)
os.makedirs(weight_result_dir, exist_ok=True)

# Function to train the model
def train_model(model, train_loader, valid_loader, optimizer, scheduler, fold_idx, 
                epochs, device, patience, max_grad_norm, model_dir, result_dir, pos_weight):
    best_mcc = -float('inf')
    best_metric = [0, 0, 0, 0, 0, 0]
    no_improvement_count = 0
    model_path = os.path.join(model_dir, f"fold_{fold_idx}.pth")
    prob_path = os.path.join(result_dir, f"fold_{fold_idx}.csv")
    criterion = Loss(pos_weight=torch.tensor([pos_weight], device=device))
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Training loop
        for feature1, feature2, feature3, labels in train_loader:
            feature1, feature2, feature3, labels = feature1.to(device), feature2.to(device), feature3.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feature1, feature2, feature3)
            # Adjust the shape of the output to match the shape of the labels
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.view(-1)
            elif outputs.dim() == 0:
                outputs = outputs.view(1)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation loop
        model.eval()
        valid_probs = []
        valid_preds = []
        valid_labels = []

        with torch.no_grad():
            for feature1, feature2, feature3, labels in valid_loader:
                feature1, feature2, feature3, labels = feature1.to(device), feature2.to(device), feature3.to(device), labels.to(device)
                outputs = model(feature1, feature2, feature3)

                preds = torch.round(torch.sigmoid(outputs)).detach()
                probs = torch.sigmoid(outputs).detach()

                valid_probs.extend(probs.cpu().numpy())
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())

        valid_metric = compute_metrics(valid_labels, np.array(valid_probs), valid_preds)
        # Update best model if improvement is seen
        if valid_metric['MCC'] > best_mcc or epoch == 0:
            best_mcc = valid_metric['MCC']
            best_metric = valid_metric
            no_improvement_count = 0
            torch.save(model.cpu(), model_path)
            model.to(device)

            print(f"Epoch {epoch} | Fold {fold_idx}")
            print(best_metric)
            # Save best model's validation predictions, probabilities, and labels
            prob_results = pd.DataFrame({
                'prob': [prob[0] for prob in valid_probs],  # Convert each probability value from list to single number
                'pred': [str(int(pred.item())) for pred in valid_preds],  # Save as string format "1" or "0"
                'label': [str(int(label)) for label in valid_labels]  # Save as string format "1" or "0"
            })
            prob_results.to_csv(prob_path, index=False)

        else:
            no_improvement_count += 1

        # Early stopping condition
        if no_improvement_count >= patience:
            print(f"Fold {fold_idx} - Early stopping triggered after {no_improvement_count} epochs without improvement.")
            return best_metric

    return best_metric

# Read dataset
def read_fold_data(base, ratio, fold_num):
    path = f"../dataset/{base}/train_{ratio}/fold_{fold_num}.csv"
    print(path)
    return read_dataset(path)

# Start training
for fold_idx in range(5):
    train_folds = [i for i in range(5) if i != fold_idx]
    print("Train_dataset:")
    train_datasets = [read_fold_data(args.base, args.ratio, i + 1) for i in train_folds]
    print("Val_dataset:")
    val_dataset = read_fold_data(args.base, args.ratio, fold_idx + 1)
    
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = RNASeqClassifier(drop_out=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    metrics = train_model(model, train_loader, val_loader, optimizer, scheduler, 
                          max_grad_norm=10, epochs=epochs, device=device, 
                          patience=patience, result_dir=weight_result_dir, model_dir=weight_model_dir, 
                          fold_idx=fold_idx, pos_weight=args.weight_factor)
    all_metrics.append(metrics)
    
    # Save performance metrics for each fold
    metrics = {key: round(value * 100, 1) for key, value in metrics.items() if isinstance(value, (int, float))}
    metrics['ratio'] = args.ratio
    metrics['base'] = args.base
    metrics['weight_factor'] = args.weight_factor
    metrics['fold'] = fold_idx
    with open(train_metrics_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=train_fieldnames)
        writer.writerow(metrics)

# Calculate average performance metrics
avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
avg_metrics = {key: round(value * 100, 1) for key, value in avg_metrics.items() if isinstance(value, (int, float))}
avg_metrics['ratio'] = args.ratio
avg_metrics['base'] = args.base
avg_metrics['weight_factor'] = args.weight_factor
avg_metrics['fold'] = 'avg'
    
# Save average performance metrics
with open(train_metrics_file, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=train_fieldnames)
    writer.writerow(avg_metrics)
