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
    
    # Round weights to 4 decimal places
    weights = np.round(weights, 4)
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    # Round again to handle possible floating-point issues
    weights = np.round(weights, 4)
    
    # Adjust the largest weight to ensure the total sum is 1
    max_index = np.argmax(weights)
    weights[max_index] += 1 - np.sum(weights)
    
    return weights

def evaluate_weighted_ensemble(models, data_loader, weights):
    probs, labels = get_predictions(models, data_loader)
    weighted_probs = np.sum(probs * weights, axis=1)
    preds = (weighted_probs > 0.5).astype(int)
    
    metrics = compute_metrics(labels, weighted_probs, preds)
    metrics = {key: round(value * 100, 1) for key, value in metrics.items() if isinstance(value, (int, float))}
    
    return metrics

def save_weights_to_csv(weights, output_file):
    """Save weights to a CSV file"""
    df = pd.DataFrame({f'weight_{i+1}': [w] for i, w in enumerate(weights)})
    df.to_csv(output_file, index=False)
    print(f"Weights saved to {output_file}")

# Main execution logic
print("Loading 'all' base models and optimizing weights...")

# Load 'all' base models
all_model_dir = '../model/all/'
all_models = load_models(all_model_dir)

# Read 'all' base validation set
all_val_set_path = "../dataset/all/val.csv"
all_val_dataset = read_dataset(all_val_set_path)
all_val_loader = DataLoader(all_val_dataset, batch_size=64, shuffle=False)

# Get validation set predictions
global val_probs, val_labels
val_probs, val_labels = get_predictions(all_models, all_val_loader)

# Optimize weights
best_weights = optimize_weights(val_probs, val_labels)
print(f"Best weights for 'all' base: {best_weights}")

# Save weights to CSV file
weights_output_file = '../generic_weights.csv'
save_weights_to_csv(best_weights, weights_output_file)

# Evaluate each base
base_dirs = [d for d in os.listdir('../dataset/') if os.path.isdir(os.path.join('../dataset/', d))]
results = []

for base in base_dirs:
    print(f"Evaluating base: {base}")
    
    # Read test set
    test_set_path = f"../dataset/{base}/test.csv"
    test_dataset = read_dataset(test_set_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate test set
    test_metrics = evaluate_weighted_ensemble(all_models, test_loader, best_weights)
    
    result = {
        'base': base,
        **test_metrics
    }
    results.append(result)
    print(f"Base: {base}, Metrics: {result}")

# Save results
output_file = '../generic_bayes_ensemble.csv'
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
