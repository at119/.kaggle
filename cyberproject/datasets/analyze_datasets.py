# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd

# Ensure UTF-8 output in case of special characters
sys.stdout.reconfigure(encoding='utf-8')

# === 1. Load main datasets ===
credit = pd.read_csv(r"C:\Users\user\Desktop\.kaggle\cyberproject\datasets\creditcard.csv")
paysim = pd.read_csv(r"C:\Users\user\Desktop\.kaggle\cyberproject\datasets\paysim\PS_20174392719_1491204439457_log.csv")

# === 2. Load all Bank dataset variants automatically ===
bank_folder = r"C:\Users\user\Desktop\.kaggle\cyberproject\datasets\bank"
bank_files = [f for f in os.listdir(bank_folder) if f.endswith(".csv")]

bank_datasets = {}
for file in bank_files:
    path = os.path.join(bank_folder, file)
    bank_datasets[file] = pd.read_csv(path)

# === 3. Print dataset sizes ===
print("=== Dataset Sizes (rows, columns) ===")
print(f"Credit Card Fraud Detection: {credit.shape}")
print(f"PaySim Synthetic Transactions: {paysim.shape}")
for name, df in bank_datasets.items():
    print(f"Bank Dataset - {name}: {df.shape}")
print("-" * 70)

# === 4. Detailed info for Credit Card Fraud Detection ===
print("Credit Card Fraud Detection Dataset Info")
print(f"Shape: {credit.shape}")
print(f"First 5 columns: {credit.columns[:5].tolist()}")
print(f"Missing values: {credit.isnull().sum().sum()}")
print(f"Labeled: {'Class' in credit.columns}")
print()

# === 5. Summary for each Bank variant ===
for name, df in bank_datasets.items():
    print(f"{name} - Columns: {len(df.columns)}, Rows: {len(df)}")
    print(f"First 5 columns: {df.columns[:5].tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("-" * 70)

# === 6. Create comparison table and save to /logs ===
comparison_data = [
    ["Credit Card Fraud Detection", credit.shape[0], credit.shape[1], "Yes", "Real anonymized card transactions"],
    ["PaySim Synthetic Transactions", paysim.shape[0], paysim.shape[1], "Yes", "Synthetic mobile money data"],
]

for name, df in bank_datasets.items():
    comparison_data.append([f"Bank - {name}", df.shape[0], df.shape[1], "Yes", "Synthetic banking fraud variants"])

comparison_df = pd.DataFrame(
    comparison_data,
    columns=["Dataset", "Samples (rows)", "Features (columns)", "Labeled", "Description"]
)

# Make sure logs folder exists
log_folder = r"C:\Users\user\Desktop\.kaggle\cyberproject\logs"
os.makedirs(log_folder, exist_ok=True)

# Save comparison CSV
comparison_path = os.path.join(log_folder, "dataset_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)

print("\nComparison table saved to:", comparison_path)
print("\n=== Dataset Comparison Table ===")
print(comparison_df)
