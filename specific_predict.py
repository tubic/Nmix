import os
import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
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

def read_fasta(fasta_file):
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).replace('T', 'U').upper()
        sequences.append(seq)
        ids.append(record.id)
    return ids, sequences

def split_data_by_center(ids, sequences, structures):
    data = {'A': [], 'C': [], 'G': [], 'U': []}
    for id, seq, struct in zip(ids, sequences, structures):
        center = seq[20]  # 21st base (index starts at 0)
        data[center].append((id, seq, struct))
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
    ids, sequences = read_fasta(args.input)

    print("Predicting secondary structures...")
    structures = [RNA.fold(seq)[0] for seq in sequences]

    print("Splitting data by center nucleotide...")
    split_data = split_data_by_center(ids, sequences, structures)

    all_results = []
    for nucleotide in ['A', 'C', 'G', 'U']:
        if split_data[nucleotide]:
            print(f"Processing {nucleotide} sequences...")
            subset_ids, subset_seqs, subset_structs = zip(*split_data[nucleotide])
            
            test_dataset = create_predict_dataset(subset_seqs, subset_structs)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            probabilities = predict(models[nucleotide], weights[nucleotide], test_loader)
            subset_results = pd.DataFrame({
                'ID': subset_ids,
                'Sequence': subset_seqs,
                'Probability': np.round(probabilities, 3),
                'Prediction': ['Nm site' if prob > 0.5 else 'non-Nm site' for prob in probabilities]
            })
            all_results.append(subset_results)

    # Merge results and sort by original order
    results = pd.concat(all_results)
    results = results.set_index('ID').loc[ids].reset_index()

    results.to_csv(args.output, index=False)
    print(f"Prediction results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nmix")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output CSV file")
    args = parser.parse_args()

    main(args)