import os
import torch
import argparse
import pandas as pd
import numpy as np
import importlib
from utils import set_seed, read_dataset, compute_metrics
from torch.utils.data import DataLoader

# Create ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
args = parser.parse_args()

# Set seed and device
seed = 42
set_seed(seed)
device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device("cpu")

def load_models(model_dir):
    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]
    models = []
    
    for model_path in model_paths:
        # Load the entire model
        model = torch.load(model_path, map_location=device)
        model = model.to(device)
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
    # Average performance across 5 folds
    metrics_per_fold = [compute_metrics(all_labels, probs, preds) for probs, preds in zip(all_fold_probs, all_fold_preds)]
    average_metrics = pd.DataFrame(metrics_per_fold).mean().to_dict()
    average_metrics = {key: round(value * 100, 1) for key, value in average_metrics.items() if isinstance(value, (int, float))}
    
    return average_metrics

def save_results(results, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    
    # If file exists, read existing data
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        new_df = pd.DataFrame(results)
        
        # Merge existing and new data, remove duplicates
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['model', 'base'], keep='last')
        
        # Save merged data
        combined_df.to_csv(file_path, index=False)
    else:
        # If file doesn't exist, save new data directly
        pd.DataFrame(results).to_csv(file_path, index=False)

def process_and_save_results(train_metrics_df, test_metrics_df):
    # Get all completed training combinations
    completed_combinations = train_metrics_df[['model', 'base']].drop_duplicates()

    # Get already tested combinations
    tested_combinations = test_metrics_df[['model', 'base']]

    # Find new combinations to test
    new_combinations = pd.merge(completed_combinations, tested_combinations, how='left', indicator=True)
    new_combinations = new_combinations[new_combinations['_merge'] == 'left_only'].drop('_merge', axis=1)

    new_results = []

    for _, row in new_combinations.iterrows():
        model = row['model']
        base = row['base']
        
        print(f"Processing: model={model}, base={base}")

        # Model directory
        model_dir = f'../train_model/{model}/{base}/'
        models = load_models(model_dir)

        # Read validation set
        test_set_path = f"../dataset/{base}/test.csv"
        test_dataset = read_dataset(test_set_path)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_fold_probs, all_fold_preds, all_labels = test_model(models, test_loader)
        average_metrics = evaluate_predictions(all_fold_probs, all_fold_preds, all_labels)
        print(average_metrics)

        new_results.append({
            'model': model,
            'base': base,
            **average_metrics
        })

    # Save results to csv file
    directory = '../'
    save_results(new_results, directory, 'test_metrics.csv')

# Main program
train_metrics_df = pd.read_csv('../train_metrics.csv').drop_duplicates()

# Read existing validation set results (if they exist)
test_metrics_path = '../test_metrics.csv'
if os.path.exists(test_metrics_path):
    test_metrics_df = pd.read_csv(test_metrics_path)
else:
    test_metrics_df = pd.DataFrame(columns=['model', 'base'])

process_and_save_results(train_metrics_df, test_metrics_df)

print("Results have been saved for test set.")

