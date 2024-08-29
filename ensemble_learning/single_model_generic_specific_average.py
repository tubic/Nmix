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

def test_model(models, test_loader):
    all_fold_probs = []
    all_fold_preds = []
    all_labels = []

    for model in models:
        fold_probs = []
        fold_preds = []
        fold_labels = []

        with torch.no_grad():
            for feature1, feature2, feature3, labels in test_loader:
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

def process_specific_models():
    results = []
    all_probs = []
    all_preds = []
    all_labels = []

    for base in ['A', 'C', 'G', 'U']:
        print(f"Processing models for {base}")
        model_dir = f'../model/{base}/'
        models = load_models(model_dir)

        test_set_path = f"../dataset/{base}/test.csv"
        test_dataset = read_dataset(test_set_path)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_fold_probs, all_fold_preds, labels = test_model(models, test_loader)
        average_metrics = evaluate_predictions(all_fold_probs, all_fold_preds, labels)
        
        results.append({
            'base': base,
            **average_metrics
        })

        # Save predictions and labels for each base
        all_probs.extend(np.mean(all_fold_probs, axis=0))
        all_preds.extend(np.round(np.mean(all_fold_preds, axis=0)))
        all_labels.extend(labels)

    # Compute overall performance
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_metrics = compute_metrics(all_labels, all_probs, all_preds)
    overall_metrics = {key: round(value * 100, 1) for key, value in overall_metrics.items() if isinstance(value, (int, float))}

    results.append({
        'base': 'all',
        **overall_metrics
    })

    df = pd.DataFrame(results)
    df.to_csv('../specific_average.csv', index=False)
    print("Specific model results have been saved to specific_average.csv")

def process_generic_model():
    results = []
    model_dir = '../model/all/'
    models = load_models(model_dir)

    for base in ['A', 'C', 'G', 'U', 'all']:
        print(f"Testing generic model on {base} dataset")
        test_set_path = f"../dataset/{base}/test.csv"
        test_dataset = read_dataset(test_set_path)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_fold_probs, all_fold_preds, all_labels = test_model(models, test_loader)
        average_metrics = evaluate_predictions(all_fold_probs, all_fold_preds, all_labels)
        
        results.append({
            'base': base,
            **average_metrics
        })

    df = pd.DataFrame(results)
    df.to_csv('../generic_average.csv', index=False)
    print("Generic model results have been saved to generic_average.csv")

process_specific_models()
process_generic_model()

print("All results have been saved.")
