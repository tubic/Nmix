import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix

# Set a random seed for reproducibility
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed as {seed}")

# Convert nucleotide sequence to z-curve representation
def z_curve(seq):
    x, y, z = [0], [0], [0]
    for nucleotide in seq:
        delta_x = 1 if nucleotide in ['A', 'G'] else -1
        delta_y = 1 if nucleotide in ['A', 'C'] else -1
        delta_z = 1 if nucleotide in ['A', 'U', 'T'] else -1
        x.append(x[-1] + delta_x)
        y.append(y[-1] + delta_y)
        z.append(z[-1] + delta_z)
    return x[1:], y[1:], z[1:]

# Convert nucleotide sequence to one-hot encoding
def one_hot(seq):
    a_vec = [1 if nucleotide == 'A' else 0 for nucleotide in seq]
    c_vec = [1 if nucleotide == 'C' else 0 for nucleotide in seq]
    g_vec = [1 if nucleotide == 'G' else 0 for nucleotide in seq]
    u_vec = [1 if nucleotide in ['U', 'T'] else 0 for nucleotide in seq]
    return a_vec, c_vec, g_vec, u_vec

# Convert RNA secondary structure to a matrix
def rna_structure_to_matrix(structure):
    n = len(structure)
    matrix = np.zeros((n, n), dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            matrix[i][j] = 1
            matrix[j][i] = 1
    return matrix

# Dataset for Z-Curve representation
class Z_Curve_Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x, y, z = z_curve(seq)
        return torch.tensor([x, y, z], dtype=torch.float), self.labels[idx]

# Dataset for One-Hot encoding
class One_Hot_Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        a, c, g, u = one_hot(seq)
        return torch.tensor([a, c, g, u], dtype=torch.float), self.labels[idx]

# Dataset for RNA structure matrix
class StructureMatrixDataset(Dataset):
    def __init__(self, structures, labels):
        self.structures = structures
        self.labels = labels

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        matrix = rna_structure_to_matrix(structure)
        return torch.tensor(matrix, dtype=torch.float), self.labels[idx]

# Dataset combining three different datasets
class TripletDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        assert len(dataset1) == len(dataset2) == len(dataset3)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __getitem__(self, index):
        x1, y1 = self.dataset1[index]
        x2, y2 = self.dataset2[index]
        x3, y3 = self.dataset3[index]
        return x1, x2, x3, y1  

    def __len__(self):
        return len(self.dataset1)

# Compute various performance metrics
def compute_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sen = tp / (tp + fn) if tp + fn > 0 else 0  # Sensitivity
    spe = tn / (tn + fp) if tn + fp > 0 else 0  # Specificity
    pre = tp / (tp + fp) if tp + fp > 0 else 0  # Precision

    metrics = {
        'ACC': acc,
        'SEN': sen,
        'SPE': spe,
        'PRE': pre,
        'MCC': mcc,
        'AUROC': auc,
        'F1': f1
    }
    return metrics

# Read dataset from a CSV file
def read_dataset(data_file):
    df = pd.read_csv(data_file)
    x = df['seq']
    y = df['label']
    s = df['structure']
    Z = Z_Curve_Dataset(x, y)
    H = One_Hot_Dataset(x, y)
    S = StructureMatrixDataset(s, y)
    dataset = TripletDataset(H, Z, S)
    return dataset

# Create a dataset for prediction
def create_predict_dataset(sequences, structures):
    data = {'seq': sequences, 'label': [0] * len(sequences), 'structure': structures}
    df = pd.DataFrame(data)
    x = df['seq']
    y = df['label']
    s = df['structure']
    Z = Z_Curve_Dataset(x, y)
    H = One_Hot_Dataset(x, y)
    S = StructureMatrixDataset(s, y)
    dataset = TripletDataset(H, Z, S)
    return dataset








