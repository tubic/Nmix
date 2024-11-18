import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef, roc_auc_score, f1_score
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

def read_csv(input_file):
    df = pd.read_csv(input_file)
    sequences = df['seq'].str.replace('T', 'U').str.upper().tolist()
    labels = df['label'].tolist()
    ids = df['ID'].tolist() if 'ID' in df.columns else list(range(len(df)))
    return ids, sequences, labels

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
    ids, sequences, labels = read_csv(args.input)

    print("Predicting secondary structures...")
    structures = [RNA.fold(seq)[0] for seq in sequences]  # Only take the structure, not the energy

    print("Preparing test data...")
    test_dataset = create_predict_dataset(sequences, structures)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Making predictions...")
    probabilities = predict(models, weights, test_loader)

    # Generate prediction labels using 0.5 as the threshold
    predictions = [1 if prob > 0.5 else 0 for prob in probabilities]

    # Calculate and print performance metrics
    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    auroc = roc_auc_score(labels, probabilities)
    f1 = f1_score(labels, predictions)

    print(f"Accuracy: {acc:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"AUROC: {auroc:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Create prediction results
    results = pd.DataFrame({
        'ID': ids,
        'Sequence': sequences,
        'True Label': labels,
        'Probability': probabilities.round(3),
        'Prediction': predictions
    })

    results.to_csv(args.output, index=False)
    print(f"Prediction results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nmix Test")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input CSV file containing sequences and labels")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output CSV file")
    args = parser.parse_args()

    main(args)
