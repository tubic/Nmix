import os
import torch
import pandas as pd
import numpy as np
from utils import set_seed, read_dataset, compute_metrics
from torch.utils.data import DataLoader

seed = 42
set_seed(seed)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device("cpu")

def load_models(model_dir):
    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]
    models = []
    for model_path in model_paths:
        model = torch.load(model_path).to(device)
        model.eval()
        models.append(model)
    return models

def val_model(models, val_loader):
    all_fold_probs = []
    all_fold_preds = []
    all_labels = []

    for model in models:
        fold_probs = []
        fold_preds = []
        fold_labels = []

        with torch.no_grad():
            for feature1, feature2, feature3, labels in val_loader:
                feature1, feature2, feature3, labels = feature1.to(device), feature2.to(device), feature3.to(device), labels.to(device)
                outputs = model(feature1, feature2, feature3)
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.view(-1)
                elif outputs.dim() == 0:
                    outputs = outputs.view(1)
                labels = labels.float()
                preds = torch.round(torch.sigmoid(outputs)).detach()
                probs = torch.sigmoid(outputs).detach()
                fold_probs.append(probs.cpu().numpy())
                fold_preds.append(preds.cpu().numpy())
                fold_labels.append(labels.cpu().numpy())

        fold_probs = np.concatenate(fold_probs)
        fold_preds = np.concatenate(fold_preds)
        fold_labels = np.concatenate(fold_labels)

        all_fold_probs.append(fold_probs)
        all_fold_preds.append(fold_preds)
        all_labels = fold_labels

    return all_fold_probs, all_fold_preds, all_labels

def evaluate_predictions(all_fold_probs, all_fold_preds, all_labels):
    metrics_per_fold = [compute_metrics(all_labels, probs, preds) for probs, preds in zip(all_fold_probs, all_fold_preds)]
    average_metrics = pd.DataFrame(metrics_per_fold).mean().to_dict()
    average_metrics = {key: round(value * 100, 1) for key, value in average_metrics.items() if isinstance(value, (int, float))}
    return average_metrics

def save_results(results, filename):
    df = pd.DataFrame(results)
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['model', 'dataset'], keep='last')
        combined_df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)

def process_and_save_results():
    model_types = ['A', 'C', 'G', 'U', 'all']
    results = []

    for model_type in model_types:
        print(f"Processing models for {model_type}")
        model_dir = f'../model/{model_type}/'
        models = load_models(model_dir)

        for dataset_type in model_types:
            print(f"Testing on {dataset_type} dataset")
            val_set_path = f"../dataset/{dataset_type}/val.csv"
            val_dataset = read_dataset(val_set_path)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            all_fold_probs, all_fold_preds, all_labels = val_model(models, val_loader)
            average_metrics = evaluate_predictions(all_fold_probs, all_fold_preds, all_labels)
            
            results.append({
                'model': model_type,
                'dataset': dataset_type,
                **average_metrics
            })

    save_results(results, 'pair_val_metrics.csv')

process_and_save_results()

print("Results have been saved to pair_val_metrics.csv")