import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import set_seed, read_dataset, compute_metrics
from bayes_opt import BayesianOptimization
from sklearn.metrics import matthews_corrcoef

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

def get_predictions(models, data_loader):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for feature1, feature2, feature3, labels in data_loader:
            feature1, feature2, feature3 = feature1.to(device), feature2.to(device), feature3.to(device)
            fold_probs = []

            for model in models:
                outputs = model(feature1, feature2, feature3)
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.view(-1)
                elif outputs.dim() == 0:
                    outputs = outputs.view(1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.append(probs)

            all_probs.append(np.array(fold_probs))
            all_labels.extend(labels.numpy())

    return np.concatenate(all_probs, axis=1).T, np.array(all_labels)

def objective_function(w1, w2, w3, w4, w5):
    weights = np.array([w1, w2, w3, w4, w5])
    weights = weights / np.sum(weights)  # Normalize weights
    
    weighted_probs = np.sum(val_probs * weights, axis=1)
    preds = (weighted_probs > 0.5).astype(int)
    
    return matthews_corrcoef(val_labels, preds)

def optimize_weights(val_probs, val_labels):
    pbounds = {'w1': (0, 1), 'w2': (0, 1), 'w3': (0, 1), 'w4': (0, 1), 'w5': (0, 1)}
    
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
    )
    
    optimizer.maximize(init_points=20, n_iter=50)
    
    best_weights = optimizer.max['params']
    weights = np.array([best_weights[f'w{i}'] for i in range(1, 6)])
    weights = weights / np.sum(weights)
    weights = np.round(weights, 4)
    max_index = np.argmax(weights)
    weights[max_index] += 1 - np.sum(weights)
    
    return weights

def evaluate_weighted_ensemble(models, data_loader, weights):
    probs, labels = get_predictions(models, data_loader)
    weighted_probs = np.sum(probs * weights, axis=1)
    preds = (weighted_probs > 0.5).astype(int)
    
    return weighted_probs, preds, labels

def process_base(base):
    print(f"Processing base: {base}")

    model_dir = f'../model/{base}/'
    models = load_models(model_dir)

    val_set_path = f"../dataset/{base}/val.csv"
    val_dataset = read_dataset(val_set_path)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    global val_probs, val_labels
    val_probs, val_labels = get_predictions(models, val_loader)

    best_weights = optimize_weights(val_probs, val_labels)
    print(f"Best weights for {base}: {best_weights}")

    test_set_path = f"../dataset/{base}/test.csv"
    test_dataset = read_dataset(test_set_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    weighted_probs, preds, labels = evaluate_weighted_ensemble(models, test_loader, best_weights)
    
    return {
        'base': base,
        'weights': best_weights.tolist(),
        'probs': weighted_probs,
        'preds': preds,
        'labels': labels
    }

def save_results(results, file_path):
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

def save_weights(weights_data, file_path):
    df = pd.DataFrame(weights_data)
    df.to_csv(file_path, index=False)
    print(f"Weights saved to {file_path}")

# Main execution logic
base_dirs = ['A', 'C', 'G', 'U']
results = []
weights_data = []
all_probs = []
all_preds = []
all_labels = []

for base in base_dirs:
    result = process_base(base)
    
    metrics = compute_metrics(result['labels'], result['probs'], result['preds'])
    metrics = {key: round(value * 100, 1) for key, value in metrics.items() if isinstance(value, (int, float))}
    
    results.append({
        'base': base,
        'weights': result['weights'],
        **metrics
    })
    print(f"Base: {base}, Metrics: {metrics}")
    
    weights_dict = {'base': base}
    weights_dict.update({f'weight_{i+1}': w for i, w in enumerate(result['weights'])})
    weights_data.append(weights_dict)
    
    all_probs.extend(result['probs'])
    all_preds.extend(result['preds'])
    all_labels.extend(result['labels'])

# Compute overall performance
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
overall_metrics = compute_metrics(all_labels, all_probs, all_preds)
overall_metrics = {key: round(value * 100, 1) for key, value in overall_metrics.items() if isinstance(value, (int, float))}
overall_result = {
    'base': 'all',
    'weights': [],
    **overall_metrics
}
results.append(overall_result)
print(f"Overall Metrics: {overall_metrics}")

save_results(results, '../specific_bayes_ensemble.csv')
save_weights(weights_data, '../specific_weights.csv')
