## -*- coding: -*-
## FuMOGAE , hybrid Boruta + mRMR feature selection is applied to six omic modalities (CNA, GSE,
## DNA-Methylation, miRNA, Mutation, and Co-expression). GAE is then used to extract latent embeddings
## from the selected features of these six modalities, and the resulting embeddings are concatenated with
## the Clinical (CLN) features to form a unified representation. The final breast cancer subtype
## classification is performed using a hybrid Choquet–Sugeno fuzzy integral.
# ============================================================
# 🔥 FINAL FEATURE SELECTION PIPELINE (CLEAN + STABLE)
# ============================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from boruta import BorutaPy
from mrmr import mrmr_classif

np.random.seed(42)

# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

clinical = pd.read_csv("final_preprocessed_clinical.csv")
cna      = pd.read_csv("final_preprocessed_cna.csv")
gse      = pd.read_csv("final_preprocessed_gse.csv")
dna      = pd.read_csv("final_preprocessed_dna.csv")
mirna    = pd.read_csv("final_preprocessed_mirna.csv")
mutation = pd.read_csv("final_preprocessed_mutation.csv")
coexp    = pd.read_csv("final_preprocessed_coexp.csv")

# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_X(df):
    return df.iloc[:, :-1].values

labels = clinical.iloc[:, -1].values.astype(int)

X_data = {
    "clinical": extract_X(clinical),
    "cna": extract_X(cna),
    "gse": extract_X(gse),
    "dna": extract_X(dna),
    "mirna": extract_X(mirna),
    "mutation": extract_X(mutation),
    "coexp": extract_X(coexp)
}

# ============================================================
# SUMMARY STORAGE
# ============================================================

summary = []

# ============================================================
# FEATURE SELECTION FUNCTION
# ============================================================

def feature_selection(X, y, name, k_features, use_prefilter=False):

    print(f"\n🔹 Processing: {name}")

    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    y = np.asarray(y, dtype=int)

    original_features = X.shape[1]
    feature_names = [f"{name}_f{i}" for i in range(original_features)]

    # ---------------- PREFILTER (DNA only) ----------------
    if use_prefilter and X.shape[1] > 3000:
        print("⚡ Applying MI prefilter...")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        top_idx = np.argsort(-mi_scores)[:3000]
        X = X[:, top_idx]
        feature_names = [feature_names[i] for i in top_idx]

    # ---------------- BORUTA ----------------
    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=5,
        random_state=42
    )

    boruta = BorutaPy(
        rf,
        n_estimators='auto',
        max_iter=50,
        random_state=42,
        verbose=0
    )

    boruta.fit(X, y)

    if boruta.support_.sum() == 0:
        print("⚠️ Boruta fallback → MI")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        idx = np.argsort(-mi_scores)[:k_features]
        X_boruta = X[:, idx]
        boruta_features = [feature_names[i] for i in idx]
    else:
        mask = boruta.support_
        X_boruta = X[:, mask]
        boruta_features = [f for f, m in zip(feature_names, mask) if m]

    # ---------------- mRMR ----------------
    k = min(k_features, X_boruta.shape[1])

    df_X = pd.DataFrame(X_boruta, columns=boruta_features)
    y_series = pd.Series(y)

    selected_features = mrmr_classif(df_X, y_series, K=k)
    X_final = df_X[selected_features].values

    # ---------------- SAVE ----------------
    pd.Series(selected_features).to_csv(f"selected_{name}_features.csv", index=False)

    df_out = pd.DataFrame(X_final, columns=selected_features)
    df_out["label"] = y
    df_out.to_csv(f"selected_{name}_data.csv", index=False)

    # ---------------- SUMMARY ----------------
    summary.append({
        "Modality": name,
        "Original": original_features,
        "After_Boruta": X_boruta.shape[1],
        "After_mRMR": len(selected_features),
        "Final_Selected": X_final.shape[1]
    })

    print(f"✅ {name}: {original_features} → {X_boruta.shape[1]} → {X_final.shape[1]}")

    return X_final


# ============================================================
# APPLY PIPELINE (FIXED LOOP)
# ============================================================

selected_data = {}

for name, X in X_data.items():

    # -------- CLINICAL --------
    if name == "clinical":
        print("\n✅ Skipping feature selection for clinical")

        feature_names = [f"clinical_f{i}" for i in range(X.shape[1])]
        pd.Series(feature_names).to_csv("selected_clinical_features.csv", index=False)

        df_out = pd.DataFrame(X, columns=feature_names)
        df_out["label"] = labels
        df_out.to_csv("selected_clinical_data.csv", index=False)

        summary.append({
            "Modality": name,
            "Original": X.shape[1],
            "After_Boruta": X.shape[1],
            "After_mRMR": X.shape[1],
            "Final_Selected": X.shape[1]
        })

        selected_data[name] = X
        continue

    # -------- SKIP MUTATION & COEXP --------
    if name in ["mutation", "coexp"]:
        print(f"\n✅ Skipping feature selection for {name}")

        df_out = pd.DataFrame(X)
        df_out["label"] = labels
        df_out.to_csv(f"selected_{name}_data.csv", index=False)

        summary.append({
            "Modality": name,
            "Original": X.shape[1],
            "After_Boruta": X.shape[1],
            "After_mRMR": X.shape[1],
            "Final_Selected": X.shape[1]
        })

        selected_data[name] = X
        continue

    # -------- DNA (WITH PREFILTER) --------
    if name == "dna":
        selected_data[name] = feature_selection(
            X, labels, name, k_features=1000, use_prefilter=True
        )

    # -------- OTHERS --------
    else:
        selected_data[name] = feature_selection(
            X, labels, name, k_features=200, use_prefilter=False
        )


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(summary)

print("\n📊 FEATURE SELECTION SUMMARY")
print(summary_df)

summary_df.to_csv("feature_selection_summary.csv", index=False)

print("\n🎉 ALL DONE SUCCESSFULLY!")
