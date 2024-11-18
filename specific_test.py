import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef, roc_auc_score, f1_score
from torch.utils.data import DataLoader
from utils import set_seed, create_predict_dataset
from model import RNASeqClassifier
import RNA
import argparse

# Set random seed to ensure reproducibility
set_seed(42)

def load_models(model_base_dir):
    models = {}
    for nucleotide in ['A', 'C', 'G', 'U']:
        models[nucleotide] = []
        model_dir = os.path.join(model_base_dir, nucleotide)
        for i in range(5):
            model_path = os.path.join(model_dir, f'fold_{i}.pth')
            model = torch.load(model_path)
            model.eval()
            models[nucleotide].append(model)
    return models

def load_weights(weights_file):
    df = pd.read_csv(weights_file)
    weights = {}
    for _, row in df.iterrows():
        base = row['base']
        weights[base] = row[['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5']].values
        weights[base] = weights[base] / np.sum(weights[base])  # Ensure weights sum to 1
    return weights

def read_csv(input_file):
    df = pd.read_csv(input_file)
    sequences = df['seq'].str.replace('T', 'U').str.upper().tolist()
    labels = df['label'].tolist()
    ids = df['ID'].tolist() if 'ID' in df.columns else list(range(len(df)))
    return ids, sequences, labels

def split_data_by_center(ids, sequences, structures, labels):
    data = {'A': [], 'C': [], 'G': [], 'U': []}
    for id, seq, struct, label in zip(ids, sequences, structures, labels):
        center = seq[20]  # 21st base (index starts at 0)
        data[center].append((id, seq, struct, label))
    return data

def predict(models, weights, data_loader):
    all_probs = []
    
    with torch.no_grad():
        for feature1, feature2, feature3, *_ in data_loader:
            fold_probs = []
            for model in models:
                outputs = model(feature1, feature2, feature3)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.append(probs)
            weighted_probs = np.average(fold_probs, axis=0, weights=weights)
            all_probs.extend(weighted_probs)
    return np.array(all_probs, dtype=float).flatten()

def main(args):
    print("Loading models and weights...")
    models = load_models('./model')
    weights = load_weights('./model/specific_weights.csv')

    print(f"Reading test set from {args.input}...")
    ids, sequences, labels = read_csv(args.input)

    print("Predicting secondary structures...")
    structures = [RNA.fold(seq)[0] for seq in sequences]

    print("Splitting data by center nucleotide...")
    split_data = split_data_by_center(ids, sequences, structures, labels)

    all_results = []
    all_true_labels = []
    all_predictions = []
    all_probabilities = []
    for nucleotide in ['A', 'C', 'G', 'U']:
        if split_data[nucleotide]:
            print(f"Processing {nucleotide} sequences...")
            subset_ids, subset_seqs, subset_structs, subset_labels = zip(*split_data[nucleotide])
            
            test_dataset = create_predict_dataset(subset_seqs, subset_structs)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            probabilities = predict(models[nucleotide], weights[nucleotide], test_loader)
            predictions = [1 if prob > 0.5 else 0 for prob in probabilities]

            all_true_labels.extend(subset_labels)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

            subset_results = pd.DataFrame({
                'ID': subset_ids,
                'Sequence': subset_seqs,
                'True Label': subset_labels,
                'Probability': np.round(probabilities, 3),
                'Prediction': predictions
            })
            all_results.append(subset_results)

    # Merge results and sort by original order
    results = pd.concat(all_results)
    results = results.set_index('ID').loc[ids].reset_index()

    # Calculate and print performance metrics
    acc = accuracy_score(all_true_labels, all_predictions)
    recall = recall_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions)
    mcc = matthews_corrcoef(all_true_labels, all_predictions)
    auroc = roc_auc_score(all_true_labels, all_probabilities)
    f1 = f1_score(all_true_labels, all_predictions)

    print(f"Accuracy: {acc:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"AUROC: {auroc:.3f}")
    print(f"F1 Score: {f1:.3f}")

    results.to_csv(args.output, index=False)
    print(f"Prediction results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nmix Test")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input CSV file containing sequences and labels")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output CSV file")
    args = parser.parse_args()

    main(args)
