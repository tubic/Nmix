from utils import *
from model import *
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def process_and_save_subset(base, model, is_trained):
    file_prefix = 'trained' if is_trained else 'untrained'
    file_name = f'tsne_data_{file_prefix}_{base}.npz'
    
    if os.path.exists(file_name):
        print(f"Loading existing data for {file_prefix} {base}")
        data = np.load(file_name)
        return data['tsne_results'], data['labels']
    
    print(f"Processing data for {file_prefix} {base}")
    # Read test dataset
    test_set_file = f"./dataset/{base}/test.csv"
    test_df = pd.read_csv(test_set_file)
    test_dataset = read_dataset(test_set_file)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get labels from test.csv
    labels = test_df['label'].values

    intermediate_features_list = []

    with torch.no_grad():
        for batch in test_loader:
            x1, x2, x3, _ = batch
            _, features = model.forward(x1, x2, x3)
            intermediate_features_list.append(features.cpu())

        intermediate_features_all = torch.cat(intermediate_features_list, dim=0)

    intermediate_features_np = intermediate_features_all.numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(intermediate_features_np)

    # Save results
    np.savez(file_name, tsne_results=tsne_results, labels=labels)

    return tsne_results, labels

def plot_tsne(ax, tsne_results, labels, base, is_trained):
    point_size = 10
    color_1 = '#41A2A2'  # Color for non-Nm sites
    color_2 = '#F36A66'  # Color for Nm sites

    for label in np.unique(labels):
        indices = labels == label
        color = color_1 if label == 0 else color_2
        label_name = "Non-Nm sites" if label == 0 else "Nm sites"
        ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                    edgecolors=color, facecolors='none', s=point_size, 
                    marker='o', label=label_name)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    status = "Trained" if is_trained else "Untrained"
    ax.set_title(f"{status} Nmix - {base}m Subset")
    legend = ax.legend()

    for handle in legend.legendHandles:
        handle.set_sizes([30])

# Process data
bases = ['A', 'C', 'G', 'U']

for base in bases:
    # Untrained model
    untrained_model = RNASeqClassifier()
    process_and_save_subset(base, untrained_model, False)

    # Trained model
    trained_model = torch.load('./model/all/fold_0.pth')
    process_and_save_subset(base, trained_model, True)

# Create main figure and subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

for i, base in enumerate(bases):
    # Untrained model
    tsne_results, labels = process_and_save_subset(base, None, False)
    plot_tsne(axs[0, i], tsne_results, labels, base, False)

    # Trained model
    tsne_results, labels = process_and_save_subset(base, None, True)
    plot_tsne(axs[1, i], tsne_results, labels, base, True)

plt.tight_layout()
plt.savefig('tsne_all_subsets.png', dpi=500, bbox_inches='tight')
plt.show()