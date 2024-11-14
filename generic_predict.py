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

def load_models(model_dir):
    models = []
    for i in range(5):
        model_path = os.path.join(model_dir, f'fold_{i}.pth')
        model = torch.load(model_path)
        model.eval()
        models.append(model)
    return models

def load_weights(weights_file):
    df = pd.read_csv(weights_file)
    weights = df[['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5']].values[0]
    return weights / np.sum(weights)  # Ensure weights sum to 1

def read_fasta(fasta_file):
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).replace('T', 'U').upper()
        sequences.append(seq)
        ids.append(record.id)
    return ids, sequences

def predict(models, weights, data_loader):
    all_probs = []
    
    with torch.no_grad():
        for feature1, feature2, feature3, *_ in data_loader:
            fold_probs = []
            for model in models:
                outputs = model(feature1, feature2, feature3)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.append(probs)
            # Calculate weighted average probabilities using weights
            weighted_probs = np.average(fold_probs, axis=0, weights=weights)
            all_probs.extend(weighted_probs)
    return np.array(all_probs).flatten()

def main(args):
    print("Loading models and weights...")
    models = load_models('./model/all')
    weights = load_weights('./model/generic_weights.csv')

    print(f"Reading test set from {args.input}...")
    ids, sequences = read_fasta(args.input)

    print("Predicting secondary structures...")
    structures = [RNA.fold(seq)[0] for seq in sequences]  # Only take the structure, not the energy

    print("Preparing test data...")
    test_dataset = create_predict_dataset(sequences, structures)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Making predictions...")
    probabilities = predict(models, weights, test_loader)

    # Create prediction results
    predictions = ['Nm site' if prob > 0.5 else 'non-Nm site' for prob in probabilities]

    # Save results
    results = pd.DataFrame({
        'ID': ids,
        'Sequence': sequences,
        'Probability': probabilities.round(3),
        'Prediction': predictions
    })

    results.to_csv(args.output, index=False)
    print(f"Prediction results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nmix")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output CSV file")
    args = parser.parse_args()

    main(args)