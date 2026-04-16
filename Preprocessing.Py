## -*- coding: -*-
## FuMOGAE , GAE is applied to six omic modalities (CNA, GSE, DNA-Methylation, miRNA, Mutation, and
## Co-expression) after hybrid Boruta + mRMR feature selection. Latent embeddings extracted from these
## six GAEs are concatenated with the Clinical (CLN) features to form a unified representation.
## The final breast cancer subtype classification is performed using a hybrid Choquet–Sugeno fuzzy integral.
#
#
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from boruta import BorutaPy
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import re

# Load the datasets
cna_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/Subtype/cna_1253SUBTYPE.csv",header=None).dropna(how="all")
clinical_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/Subtype/clinical_1253SUBTYPE.csv", header=None)
gse_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/Subtype/gse_1253SUBTYPE.csv", header=None)
Dna_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/Subtype/dna_1253SUBTYPE.csv", header=None)
mi_rna_data=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/Subtype/mi_rna_1253SUBTYPE.csv", header=None)
mut_data=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/Subtype/mutation_1253SUBTYPE.csv", header=None)
coexp_data=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/Subtype/coexp_1253SUBTYPE.csv",header=None)
cna_data.head()
import pandas as pd

# Define label mapping
label_mapping = {
    'LumB': 0,
    'Normal': 1,
    'Her2': 2,
    'LumA': 3,
    'Basal': 4
}

# -----------------------------
# 🔹 Apply label mapping
# -----------------------------
clinical_data.iloc[:, -1] = clinical_data.iloc[:, -1].map(label_mapping)
cna_data.iloc[:, -1] = cna_data.iloc[:, -1].map(label_mapping)
gse_data.iloc[:, -1] = gse_data.iloc[:, -1].map(label_mapping)
Dna_data.iloc[:, -1] = Dna_data.iloc[:, -1].map(label_mapping)
mi_rna_data.iloc[:, -1] = mi_rna_data.iloc[:, -1].map(label_mapping)
mut_data.iloc[:, -1] = mut_data.iloc[:, -1].map(label_mapping)
coexp_data.iloc[:, -1] = coexp_data.iloc[:, -1].map(label_mapping)


# -----------------------------
# 🔹 Save updated datasets
# -----------------------------
clinical_data.to_csv('clinical_data_with_labels_as_numbers.csv', index=False)
cna_data.to_csv('cna_data_with_labels_as_numbers.csv', index=False)
gse_data.to_csv('gse_data_with_labels_as_numbers.csv', index=False)
Dna_data.to_csv('Dna_data_with_labels_as_numbers.csv', index=False)
mi_rna_data.to_csv('mi_rna_data_with_labels_as_numbers.csv', index=False)
mut_data.to_csv('mut_data_with_labels_as_numbers.csv', index=False)
coexp_data.to_csv('coexp_data_with_labels_as_numbers.csv', index=False)
Dna_data.head()
print("Clinical:", clinical_data.shape)
print("CNA:", cna_data.shape)
print("GSE:", gse_data.shape)
print("DNA:", Dna_data.shape)
print("miRNA:", mi_rna_data.shape)
print("Mutation:", mut_data.shape)
print("Coexp:", coexp_data.shape)
import pandas as pd
import numpy as np

# =====================================================
# 🔥 MUTATION PREPROCESSING
# =====================================================

mut_df = mut_data.copy()

# Save patient IDs (optional)
mut_ids = mut_df.iloc[:, 0]

# Remove ID column
mut_df = mut_df.iloc[:, 1:]

# Convert mutation → binary (0 / 1)
mut_df = mut_df.applymap(lambda x: 0 if str(x) == "0" else 1)

# Ensure binary
mut_df = (mut_df > 0).astype(int)

# 🔥 Frequency filtering (IMPORTANT)
freq = mut_df.sum(axis=0)
mut_df = mut_df.loc[:, freq > 10]

# =====================================================
# 🔥 ADD LABEL BACK (VERY IMPORTANT)
# =====================================================

labels = mut_data.iloc[:, -1].values  # original label column

final_mut_df = pd.concat(
    [
        mut_ids.reset_index(drop=True),      # keep ID (optional)
        mut_df.reset_index(drop=True),
        pd.Series(labels, name="label")
    ],
    axis=1
)

# =====================================================
# 🔥 SAVE FILE (FOR BULK PIPELINE)
# =====================================================

final_mut_df.to_csv("final_preprocessed_mutation.csv", index=False)

print("✅ Mutation saved successfully!")
print("Shape:", final_mut_df.shape)
print("Unique values:", np.unique(mut_df.values))
# ============================================================
# 🔥 UNIFIED PREPROCESSING FOR ALL MODALITIES (FINAL)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ============================================================
# 1. LOAD ALL DATASETS
# ============================================================

file_paths = {
    "clinical": "clinical_data_with_labels_as_numbers.csv",
    "cna": "cna_data_with_labels_as_numbers.csv",
    "gse": "gse_data_with_labels_as_numbers.csv",
    "dna": "Dna_data_with_labels_as_numbers.csv",
    "mirna": "mi_rna_data_with_labels_as_numbers.csv",
    "mutation": "final_preprocessed_mutation.csv",  # already processed
    "coexp": "coexp_data_with_labels_as_numbers.csv"
}

datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# ============================================================
# 2. PREPROCESS FUNCTION (CORE LOGIC)
# ============================================================

def preprocess_dataset(df, name):
    df = df.copy()

    print(f"\n🔹 Processing: {name}")

    # ------------------------------------------
    # STEP 1: Remove Patient ID (first column)
    # ------------------------------------------
    df = df.iloc[:, 1:]   # remove first column (ID)

    # ------------------------------------------
    # STEP 2: Convert features to numeric
    # ------------------------------------------
    for col in df.columns[:-1]:  # exclude label
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ------------------------------------------
    # STEP 3: Handle missing values
    # ------------------------------------------
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(0, inplace=True)

    # ------------------------------------------
    # STEP 4: Split features & label
    # ------------------------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # ------------------------------------------
    # STEP 5: Normalization
    # ------------------------------------------
    if name == "clinical":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------
    # STEP 6: Recreate dataframe
    # ------------------------------------------
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    # ------------------------------------------
    # STEP 7: Safety check
    # ------------------------------------------
    if final_df.isna().sum().sum() > 0:
        print(f"❌ NaN still present in {name}")
    else:
        print(f"✅ {name} clean | Shape: {final_df.shape}")

    return final_df

# ============================================================
# 3. APPLY PREPROCESSING TO ALL DATASETS
# ============================================================

processed_data = {}

for name, df in datasets.items():
    processed_data[name] = preprocess_dataset(df, name)

# ============================================================
# 4. SAVE FINAL PREPROCESSED FILES
# ============================================================
for name, original_df in datasets.items():

    processed_df = processed_data[name]

    # Extract original patient ID
    patient_id = original_df.iloc[:, 0].reset_index(drop=True)

    # -----------------------------
    # 🔹 Save WITHOUT ID (for model)
    # -----------------------------
    processed_df.to_csv(f"final_preprocessed_{name}.csv", index=False)

    # -----------------------------
    # 🔹 Save WITH ID (for tracking)
    # -----------------------------
    df_with_id = pd.concat(
        [patient_id, processed_df],
        axis=1
    )

    df_with_id.to_csv(f"final_preprocessed_{name}_with_id.csv", index=False)

print("\n🎉 Saved BOTH versions (with & without ID)")
print("Clinical:", clinical_data.shape)
print("CNA:", cna_data.shape)
print("GSE:", gse_data.shape)
print("DNA:", Dna_data.shape)
print("miRNA:", mi_rna_data.shape)
print("Mutation:", mut_data.shape)
print("Coexp:", coexp_data.shape)
