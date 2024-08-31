![logo](./logo.webp)

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Nmix-Specific](#nmix-specific)
   - [Nmix-Generic](#nmix-generic)
3. [Input Format](#input-format)
4. [Output Format](#output-format)
5. [Example](#example)
6. [Online Prediction](#online-prediction)
7. [Key Dependencies](#key-dependencies)
8. [Troubleshooting](#troubleshooting)
9. [Citation](#citation)
10. [Contact](#contact)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/tubic/Nmix.git
   cd Nmix
   ```

2. Create a conda environment from the `requirements.txt` file:

   ```
   conda create --name nmix --file requirements.txt
   ```

3. Activate the conda environment:

   ```
   conda activate nmix
   ```

## Usage

Nmix provides two prediction models: Nmix-Specific and Nmix-Generic. Both models take a FASTA file as input and produce a CSV file as output.

### Nmix-Specific

Nmix-Specific uses separate models for each nucleotide (A, C, G, U) at the center position.

```
cd ./local_prediction/
python ./specific_predict.py -gpu [GPU_ID] -i [INPUT_FASTA] -o [OUTPUT_CSV]
```

### Nmix-Generic

Nmix-Generic uses a single model for all sequences regardless of the center nucleotide.

```
cd ./local_prediction/
python ./generic_predict.py -gpu [GPU_ID] -i [INPUT_FASTA] -o [OUTPUT_CSV]
```

Parameters:

- `-gpu`: Specify which GPU to use. Use 0 for the first GPU, 1 for the second, etc. If no GPU is available, CPU will be used automatically.
- `-i` or `--input`: Path to the input FASTA file.
- `-o` or `--output`: Path to the output CSV file.

## Input Format

The input should be a FASTA file containing RNA sequences. Each sequence should be 41 nucleotides long, with the potential Nm site at the center position (21st nucleotide).

Example:

```
>seq_1
UGGUCGCAAUGUCCUUGUGAAAGAUCUGAAGACUCACCCUG
>seq_2
GGUCGGCGUGGUCCCUGGUCCAGUCGGAGAGCCAGGUGGGU
>seq_3
AAUGGGGUCAGCCUUCCACUGGGCACAUUUCUGCCCACCUU
>seq_4
GAGCUUUUUGUAUUUAUGUAGCUAUUUAUCACAGACUAGCC
```

Note: If your sequences contain 'T', they will be automatically converted to 'U'.

## Output Format

The output is a CSV file with the following columns:

- ID: Sequence identifier from the FASTA file
- Sequence: The input RNA sequence
- Probability: The predicted probability of the site being an Nm site (between 0 and 1)
- Prediction: 'Nm site' if the probability is > 0.5, otherwise 'non-Nm site'

## Example

A sample FASTA file (`sample.fasta`) is provided in the repository. You can use it to test the prediction:

```
cd ./local_prediction/
python ./specific_predict.py -gpu 0 -i sample.fasta -o sample_specific_output.csv
python ./generic_predict.py -gpu 0 -i sample.fasta -o sample_generic_output.csv
```

## Online Prediction

If you have a small number of sequences to predict, you can also use our online prediction tool at https://tubic.org/Nm. This web-based interface provides a convenient way to get predictions without setting up the local environment.

## Key Dependencies

Nmix relies on several key packages. Here are some of the important dependencies and their versions:

- Python: 3.9.18
- PyTorch: 1.12.1+cu113
- Biopython: 1.83
- NumPy: 1.26.4
- Pandas: 2.1.1
- scikit-learn: 1.3.2
- CUDA Toolkit: 11.3.1

For a complete list of dependencies, please refer to the `requirements.txt` file.

## Troubleshooting

- If you encounter a CUDA out of memory error, try reducing the batch size in the script (default is 64).
- Ensure that your input sequences are 41 nucleotides long with the potential Nm site at the center.
- If you're using a CPU-only machine, you can omit the `-gpu` parameter or set it to -1.
- Make sure you have activated the conda environment (`conda activate nmix`) before running the scripts.

## Citation

If you use Nmix in your research, please cite our paper: [paper citation to be added]

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/tubic/Nmix).
