import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import set_seed, read_dataset, compute_metrics

# Set random seed
set_seed(42)

# Set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def load_models(model_dir):
    models = []
    for i in range(5):
        model_path = os.path.join(model_dir, f'fold_{i}.pth')
        model = torch.load(model_path, map_location=device)
        model.eval()
        models.append(model)
    return models

def test_models(models, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feature1, feature2, feature3, labels in test_loader:
            feature1, feature2, feature3 = feature1.to(device), feature2.to(device), feature3.to(device)
            fold_preds = []

            for model in models:
                outputs = model(feature1, feature2, feature3)
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.view(-1)
                elif outputs.dim() == 0:
                    outputs = outputs.view(1)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
                fold_preds.append(preds)

            # Hard voting (majority vote)
            ensemble_preds = np.round(np.mean(fold_preds, axis=0)).astype(int)
            all_preds.extend(ensemble_preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

def evaluate_hard_voting(models, base):
    print(f"Processing test set for base: {base}")

    # Read test set
    test_set_path = f"../dataset/{base}/test.csv"
    test_dataset = read_dataset(test_set_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Perform prediction
    preds, labels = test_models(models, test_loader)

    # Compute metrics
    metrics = compute_metrics(labels, None, preds)
    metrics = {key: round(value * 100, 1) for key, value in metrics.items() if isinstance(value, (int, float))}
    
    return {
        'base': base,
        **metrics
    }

def save_results(results, file_path):
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

# Main execution logic

# Load models for base=all
model_dir = '../model/all/'
models = load_models(model_dir)
print("Loaded models for base=all")

# Get all base directories
base_dirs = [d for d in os.listdir('../dataset/') if os.path.isdir(os.path.join('../dataset/', d))]
results = []

for base in base_dirs:
    result = evaluate_hard_voting(models, base)
    results.append(result)
    print(f"Base: {base}, Metrics: {result}")

save_results(results, '../generic_hard_voting.csv')

print("All evaluations completed. Results saved to ../generic_hard_voting.csv")
