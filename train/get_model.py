import os
import pandas as pd
import shutil

# Read the CSV file
df = pd.read_csv('../val_metrics.csv')

# Find the row with the maximum MCC for each base
best_models = df.loc[df.groupby('base')['MCC'].idxmax()]

# Ensure the target directory exists
os.makedirs('../model', exist_ok=True)

for _, row in best_models.iterrows():
    base = row['base']
    ratio = int(row['ratio'])  # Ensure ratio is an integer
    weight_factor = round(row['weight_factor'], 1)  # Ensure weight_factor has 1 decimal place
    
    # Source directory
    src_dir = f'../train_model/ratio_{ratio}/{base}/weight_factor_{weight_factor:.1f}'
    
    # Destination directory
    dst_dir = f'../model/{base}'
    
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"Processing {base}:")
    print(f"  Source: {src_dir}")
    print(f"  Destination: {dst_dir}")
    
    # Copy model files
    for i in range(5):
        src_file = os.path.join(src_dir, f'fold_{i}.pth')
        dst_file = os.path.join(dst_dir, f'fold_{i}.pth')
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"  Copied: fold_{i}.pth")
        else:
            print(f"  Warning: {src_file} does not exist")
    
    print(f"Finished processing {base}")
    print()

print("All models have been retrieved and copied.")