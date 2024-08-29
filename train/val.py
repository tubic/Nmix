import os
import torch
import argparse
import pandas as pd
import numpy as np
from utils import set_seed, read_dataset, compute_metrics
from model import RNASeqClassifier
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

seed = 42
set_seed(seed)
device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device("cpu")

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

def save_results(results, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        new_df = pd.DataFrame(results)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['base', 'ratio', 'weight_factor'], keep='last')
        combined_df.to_csv(file_path, index=False)
    else:
        pd.DataFrame(results).to_csv(file_path, index=False)

def process_and_save_results(train_metrics_df, val_metrics_df):
    completed_combinations = train_metrics_df[train_metrics_df['fold'] == 'avg'][['base', 'ratio', 'weight_factor']].drop_duplicates()
    valed_combinations = val_metrics_df[['base', 'ratio', 'weight_factor']]
    new_combinations = pd.merge(completed_combinations, valed_combinations, how='left', indicator=True)
    new_combinations = new_combinations[new_combinations['_merge'] == 'left_only'].drop('_merge', axis=1)

    new_results = []

    for _, row in new_combinations.iterrows():
        base = row['base']
        ratio = row['ratio']
        weight_factor = row['weight_factor']
        
        print(f"Processing: base={base}, ratio={ratio}, weight_factor={weight_factor}")

        model_dir = f'../train_model/ratio_{ratio}/{base}/weight_factor_{weight_factor}/'
        models = load_models(model_dir)

        val_set_path = f"../dataset/{base}/val.csv"
        val_dataset = read_dataset(val_set_path)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        all_fold_probs, all_fold_preds, all_labels = val_model(models, val_loader)
        average_metrics = evaluate_predictions(all_fold_probs, all_fold_preds, all_labels)
        print(average_metrics)

        new_results.append({
            'base': base,
            'ratio': ratio,
            'weight_factor': weight_factor,
            **average_metrics
        })

    save_results(new_results, '../', 'val_metrics.csv')

train_metrics_df = pd.read_csv('../train_metrics.csv').drop_duplicates()

val_metrics_path = '../val_metrics.csv'
if os.path.exists(val_metrics_path):
    val_metrics_df = pd.read_csv(val_metrics_path)
else:
    val_metrics_df = pd.DataFrame(columns=['base', 'ratio', 'weight_factor'])

process_and_save_results(train_metrics_df, val_metrics_df)

print("Results have been saved for val set.")