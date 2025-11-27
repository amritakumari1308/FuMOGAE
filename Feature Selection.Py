## -*- coding: -*-
## FuMOGAE , hybrid Boruta + mRMR feature selection is applied to six omic modalities (CNA, GSE,
## DNA-Methylation, miRNA, Mutation, and Co-expression). GAE is then used to extract latent embeddings
## from the selected features of these six modalities, and the resulting embeddings are concatenated with
## the Clinical (CLN) features to form a unified representation. The final breast cancer subtype
## classification is performed using a hybrid Choquet–Sugeno fuzzy integral.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from mrmr import mrmr_classif

# ----------------------------
# Convert helper
# ----------------------------
def extract_features(df_or_arr):
    """Return features only (skip PatientID first col, skip label last col)."""
    if isinstance(df_or_arr, pd.DataFrame):
        return df_or_arr.iloc[:, 1:-1].to_numpy()
    arr = np.asarray(df_or_arr)
    return arr[:, 1:-1]

# ----------------------------
# Labels from clinical dataset
# ----------------------------
if isinstance(clinical_data, pd.DataFrame):
    labels = clinical_data.iloc[:, -1].to_numpy().astype(int)
else:
    labels = np.asarray(clinical_data)[:, -1].astype(int)

# ----------------------------
# Features for all modalities
# ----------------------------
X_clinical = extract_features(clinical_data)
X_dna      = extract_features(Dna_data)
X_mirna    = extract_features(mi_rna_data)
X_mutation = extract_features(mut_data)
X_coexp    = extract_features(coexp_data)

def to_numpy(X): return X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

X_clinical = to_numpy(X_clinical)
X_dna      = to_numpy(X_dna)
X_mirna    = to_numpy(X_mirna)
X_mutation = to_numpy(X_mutation)
X_coexp    = to_numpy(X_coexp)

selected_clinical = np.asarray(X_clinical, dtype=np.float32)
selected_mirna    = np.asarray(X_mirna,   dtype=np.float32)
selected_mutation = np.asarray(X_mutation,dtype=np.float32)
selected_coexp    = np.asarray(X_coexp,   dtype=np.float32)

print("[Shapes]")
print("|Clinical:", selected_clinical.shape,
      "| miRNA:", selected_mirna.shape,
      "| Mutation:", selected_mutation.shape,
      "| COEXP:", selected_coexp.shape)



np.save("selected_outputs/selected_clinical.npy",  selected_clinical)
np.save("selected_outputs/selected_mirna.npy",    selected_mirna)
np.save("selected_outputs/selected_mutation.npy", selected_mutation)
np.save("selected_outputs/selected_coexp.npy",    selected_coexp)


pd.DataFrame(selected_clinical).to_csv("selected_outputs/selected_clinical.csv", index=False)
pd.DataFrame(selected_mirna).to_csv("selected_outputs/selected_mirna.csv", index=False)
pd.DataFrame(selected_mutation).to_csv("selected_outputs/selected_mutation.csv", index=False)
pd.DataFrame(selected_coexp).to_csv("selected_outputs/selected_coexp.csv", index=False)
import numpy as np
import pandas as pd

# Load your preprocessed clinical CSV
clinical_data = pd.read_csv("clinical_final_preprocessed.csv")

# Extract the label column (last column)
labels = clinical_data.iloc[:, -1].to_numpy().astype(int)

# Save to numpy
np.save("labels.npy", labels)

print("✅ Saved: labels.npy")
print("Shape:", labels.shape)
print("Unique classes:", np.unique(labels))
import numpy as np
import pandas as pd

# Load the CSV
selected_cna=pd.read_csv(
    "/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/deeplearning/5th_sem/feature_selected/boruta_and_mrmr/selected_cna.csv"
)
selected_gse=pd.read_csv(
    "/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/deeplearning/5th_sem/feature_selected/boruta_and_mrmr/selected_gse.csv"
)

selected_dna = pd.read_csv(
    "/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/deeplearning/5th_sem/feature_selected/boruta_and_mrmr/selected_dna_data.csv"
)



# Convert to NumPy
selected_dna_np = selected_dna.to_numpy(dtype=np.float32)
selected_cna_np = selected_cna.to_numpy(dtype=np.float32)
selected_gse_np = selected_gse.to_numpy(dtype=np.float32)

# Save as .npy
np.save("selected_outputs/selected_dna.npy", selected_dna_np)
np.save("selected_outputs/selected_cna.npy", selected_cna_np)
np.save("selected_outputs/selected_gse.npy", selected_gse_np)

print("✅ Saved: selected_dna_np.npy")
print("Shape:", selected_dna_np.shape)

print("✅ Saved: selected_cna_np.npy")
print("Shape:", selected_cna_np.shape)

print("✅ Saved: selected_gse_np.npy")
print("Shape:", selected_gse_np.shape)
selected_clinical_np= np.load("selected_outputs/selected_clinical.npy")
selected_cna_np= np.load("selected_outputs/selected_cna.npy")
selected_gse_np= np.load("selected_outputs/selected_gse.npy")
selected_dna_np= np.load("selected_outputs/selected_dna.npy")
selected_mirna_np= np.load("selected_outputs/selected_mirna.npy")
selected_mutation_np= np.load("selected_outputs/selected_mutation.npy")
selected_coexp_np= np.load("selected_outputs/selected_coexp.npy")


print("Clinical:",selected_clinical_np.shape)
print("CNA:",selected_cna_np.shape)
print("GSE:",selected_gse_np.shape)
print("DNA:",selected_dna_np.shape)
print("miRNA:",selected_mirna_np.shape)
print("Mutation:",selected_mutation_np.shape)
print("CO-EXPRESSION:",selected_coexp_np.shape)
