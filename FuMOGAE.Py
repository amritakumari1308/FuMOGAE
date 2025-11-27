## -*- coding: -*-
## FuMOGAE , GAE is applied to six omic modalities (CNA, GSE, DNA-Methylation, miRNA, Mutation, and
## Co-expression) after hybrid Boruta + mRMR feature selection. Latent embeddings extracted from these
## six GAEs are concatenated with the Clinical (CLN) features to form a unified representation.
## The final breast cancer subtype classification is performed using a hybrid Choquet–Sugeno fuzzy integral.
------------METABRIC DATASET(CLINICAL,CNA,GSE,MUTATION,COEXPRESSION) Modalities classification
# ===============================================
# ONE FILE: Part 1 (your original) + Part 2 (all classifiers, EMB & RAWEMB)
# ===============================================

# ----------------------- Imports -----------------------
import os, random, math, re, glob
import numpy as np
import pandas as pd
from itertools import combinations

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Reproducibility ---------------------- #
SEED = 50
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_all()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- Helpers ---------------------- #
def to_numpy(arr):
    return arr.to_numpy() if hasattr(arr, "to_numpy") else np.asarray(arr)

def ensure_float32(x):
    x = to_numpy(x)
    return x.astype(np.float32, copy=False)

def mean_sd_str(values):
    vals = np.asarray(values, dtype=float)
    m = float(np.mean(vals)) if len(vals) else 0.0
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, sd, f"{m:.4f} ± {sd:.4f}"

# ---------------------- Subtype Mapping ---------------------- #
subtype_names = {0: 'LumB', 1: 'Normal', 2: 'Her2', 3: 'LumA', 4: 'Basal'}
ALL_CLASSES = list(subtype_names.keys())

# ---------------------- Evaluation ---------------------- #
def evaluate_subtypes(y_true, y_pred, y_prob, subtype_names):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_true_bin = label_binarize(y_true, classes=list(subtype_names.keys()))
    rows = []
    cm = confusion_matrix(y_true, y_pred, labels=list(subtype_names.keys()))

    for i, label in subtype_names.items():
        mask = (y_true == i)
        acc = accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else 0.0
        prec = precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        rec  = recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        f1   = f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        auc  = roc_auc_score(y_true_bin[:, i], y_prob[:, i]) if np.sum(y_true_bin[:, i]) > 0 else 0.0

        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - TP - FP - FN
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        rows.append({
            "Subtype": label,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "AUC": round(auc, 4),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4)
        })

    cm_df = pd.DataFrame(cm, index=subtype_names.values(), columns=subtype_names.values())
    return pd.DataFrame(rows), cm_df

def save_confusion_heatmap(cm_df, title, path_png):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

# ------------------------------ Model ------------------------------ #
class FFNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.4):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.cls = nn.Linear(hidden_dim // 4, num_classes)

    def forward(self, x):
        z = self.feat(x)
        logits = self.cls(z)
        return logits

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - 1e-8:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------------------ λ-Choquet & Sugeno ------------------------------ #
def solve_lambda(weights, tol=1e-7, max_iter=100):
    w = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    def f(lmbd):
        return float(np.prod(1.0 + lmbd * w) - (1.0 + lmbd))
    lo = -0.999999
    hi = 1.0
    f_lo, f_hi = f(lo), f(hi)
    while f_lo * f_hi > 0 and hi < 1e6:
        hi *= 2.0
        f_hi = f(hi)
    if f_lo * f_hi > 0:
        return 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))

def choquet_integral(values, weights):
    v = np.asarray(values, dtype=float)
    g = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    order = np.argsort(-v)
    v_sorted = v[order]
    g_sorted = g[order]
    lam = solve_lambda(g_sorted)
    mu_prev = 0.0
    res = 0.0
    for k in range(len(v_sorted)):
        mu_curr = mu_prev + g_sorted[k] + lam * mu_prev * g_sorted[k]
        res += v_sorted[k] * (mu_curr - mu_prev)
        mu_prev = mu_curr
    return float(res)

def sugeno_integral(values, weights):
    v = np.asarray(values, dtype=float)
    g = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    order = np.argsort(-v)
    v_sorted = v[order]
    g_sorted = g[order]
    lam = solve_lambda(g_sorted)
    mu_prev = 0.0
    best = 0.0
    for k in range(len(v_sorted)):
        mu_curr = mu_prev + g_sorted[k] + lam * mu_prev * g_sorted[k]
        best = max(best, min(v_sorted[k], mu_curr))
        mu_prev = mu_curr
    return float(best)

def fuse_probs(list_of_prob_arrays, weights, alpha):
    m = len(list_of_prob_arrays)
    if m == 1:
        return list_of_prob_arrays[0]
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    n, c = list_of_prob_arrays[0].shape
    fused = np.zeros((n, c), dtype=float)
    for i in range(n):
        for cls in range(c):
            vals = [arr[i, cls] for arr in list_of_prob_arrays]
            ch = choquet_integral(vals, w)
            su = sugeno_integral(vals, w)
            fused[i, cls] = alpha * ch + (1 - alpha) * su
    fused = np.maximum(fused, 1e-9)
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused

# ------------------------------ Training utils ------------------------------ #
def train_one(
    X_train, y_train, X_val, y_val,
    hidden_dim=256, max_epochs=200, lr=1e-2, wd=1e-3, patience=10
):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)

    smote = SMOTE(random_state=SEED)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr_sm), y=y_tr_sm)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    model = FFNNClassifier(X_tr_sm.shape[1], hidden_dim, len(ALL_CLASSES)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    early = EarlyStopping(patience=patience)

    X_tr_t = torch.tensor(X_tr_sm, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_sm, dtype=torch.long, device=device)
    X_v_t  = torch.tensor(X_v, dtype=torch.float32, device=device)
    y_v_t  = torch.tensor(y_val, dtype=torch.long, device=device)

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr_t)
        loss = crit(logits, y_tr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_v_t)
            v_loss = crit(v_logits, y_v_t).item()
        sched.step(v_loss)
        early(v_loss, model)
        if early.early_stop:
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    return model, scaler

def predict_proba(model, scaler, X):
    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xt), dim=1).cpu().numpy()
    return probs

def inner_cv_weight(X_train, y_train, n_splits=5):
    skf_in = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr_idx, va_idx in skf_in.split(X_train, y_train):
        model, scaler = train_one(X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx])
        probs = predict_proba(model, scaler, X_train[va_idx])
        y_pred = probs.argmax(axis=1)
        accs.append(accuracy_score(y_train[va_idx], y_pred))
    return float(np.mean(accs))

def retrain_on_outer_train(X_outer_train, y_outer_train, val_size=0.15):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_outer_train, y_outer_train))
    model, scaler = train_one(X_outer_train[tr_idx], y_outer_train[tr_idx],
                              X_outer_train[va_idx], y_outer_train[va_idx])
    return model, scaler

# ---- Data presence/loader helpers ----
def _as_float32(x): return np.asarray(x, dtype=np.float32, copy=False)
def _as_int(x):     return np.asarray(x, dtype=int, copy=False)

def get_var_or_load(name, fname, dtype="float"):
    """Use variable if already in memory; otherwise load from .npy."""
    g = globals()
    if name in g:
        arr = g[name]
    else:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing variable '{name}' and file '{fname}' not found.")
        arr = np.load(fname, allow_pickle=False)
        g[name] = arr  # cache in globals
    return _as_float32(arr) if dtype == "float" else _as_int(arr)

# ---- labels (int) ----
labels = get_var_or_load("labels", "labels.npy", dtype="int")

# ---- 7 latent modalities (float32) ----
latent_clinical  = get_var_or_load("latent_clinical",  "latent_clinical.npy",  dtype="float")
latent_cna       = get_var_or_load("latent_cna",       "latent_cna.npy",       dtype="float")
latent_gse       = get_var_or_load("latent_gse",       "latent_gse.npy",       dtype="float")
latent_dna       = get_var_or_load("latent_dna",       "latent_dna.npy",       dtype="float")
latent_mirna     = get_var_or_load("latent_mirna",     "latent_mirna.npy",     dtype="float")
latent_mutation  = get_var_or_load("latent_mutation",  "latent_mutation.npy",  dtype="float")
latent_coexp     = get_var_or_load("latent_coexp",     "latent_coexp.npy",     dtype="float")

# ---- sanity: all rows must match labels ----
n = labels.shape[0]
for nm, arr in [
    ("latent_clinical",  latent_clinical),
    ("latent_cna",       latent_cna),
    ("latent_gse",       latent_gse),
    ("latent_dna",       latent_dna),
    ("latent_mirna",     latent_mirna),
    ("latent_mutation",  latent_mutation),
    ("latent_COEXP",     latent_coexp),
]:
    if arr.shape[0] != n:
        raise ValueError(f"Row mismatch: {nm} has {arr.shape[0]} rows, but labels has {n} rows.")

# ---- Your modality combos (unchanged concept) ----
modality_combinations = {
    "Clinical": [latent_clinical],
    #"SElected_clinical": [selected_clinical_np],
    "CNA":      [latent_cna],
    "GSE":      [latent_gse],
    #"DNA":      [latent_dna],
   # "miRNA":    [latent_mirna],
    "Mutation": [latent_mutation],
    "COEXP":    [latent_coexp],
    "All_sel":      [selected_clinical_np, latent_cna, latent_gse, latent_mutation, latent_coexp],
    #"All_sel_sel":  [selected_clinical_np, latent_cna, latent_gse, latent_dna, latent_mirna, latent_mutation, selected_coexp_np]
}

print("\nConfigured single-modality models:")
for nm, lst in modality_combinations.items():
    print(f" - {nm}: {[a.shape for a in lst]}")

# =========================================================
# (Optional) RUN your original Part-1 main loop
# =========================================================
RUN_BASELINE_PART1 = True   # set True if you want to run your original FFNN-only loop

ALPHAS = [ 0.8]
N_OUTER = 10
N_INNER = 5

if RUN_BASELINE_PART1:
    all_results = []
    os.makedirs("fold_metrics", exist_ok=True)
    for alpha in ALPHAS:
        print(f"\n\n{'#'*30} Fusion α = {alpha} {'#'*30}")
        per_alpha_rows = []
        for name, modality_list in modality_combinations.items():
            print(f"\n{'='*60}\nCombo: {name} | α={alpha}\n{'='*60}")
            skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42)
            fold_acc, fold_macro, fold_weighted = [], [], []
            y_true_all, y_pred_all, y_prob_all = [], [], []
            for fold, (tr_idx, te_idx) in enumerate(skf.split(labels, labels), start=1):
                y_tr, y_te = labels[tr_idx], labels[te_idx]
                weights = []
                per_mod_test_probs = []
                for mod_arr in modality_list:
                    X = ensure_float32(mod_arr)
                    X_tr, X_te = X[tr_idx], X[te_idx]
                    w = inner_cv_weight(X_tr, y_tr, n_splits=N_INNER)
                    weights.append(w)
                    model, scaler = retrain_on_outer_train(X_tr, y_tr, val_size=0.15)
                    probs_te = predict_proba(model, scaler, X_te)
                    per_mod_test_probs.append(probs_te)
                fused = fuse_probs(per_mod_test_probs, weights, alpha)
                y_pred = fused.argmax(axis=1)
                y_true_all.extend(y_te)
                y_pred_all.extend(y_pred)
                y_prob_all.extend(fused)
                acc = accuracy_score(y_te, y_pred)
                mf1 = f1_score(y_te, y_pred, average='macro')
                wf1 = f1_score(y_te, y_pred, average='weighted')
                fold_acc.append(acc); fold_macro.append(mf1); fold_weighted.append(wf1)
                print(f"Fold {fold:02d}  Acc={acc:.4f}  MacroF1={mf1:.4f}  WeightedF1={wf1:.4f}")
                subtype_df_fold, _ = evaluate_subtypes(y_te, y_pred, fused, subtype_names)
                print(subtype_df_fold.to_string(index=False))

            y_true_all = np.asarray(y_true_all); y_pred_all = np.asarray(y_pred_all); y_prob_all = np.asarray(y_prob_all)
            subtype_df, cm_df = evaluate_subtypes(y_true_all, y_pred_all, y_prob_all, subtype_names)
            base = f"{name.replace('+','_')}_alpha_{alpha}"
            np.save(f"all_y_true_{base}.npy", y_true_all)
            np.save(f"all_y_prob_{base}.npy", y_prob_all)
            subtype_df.to_csv(f"summary_{base}_metrics.csv", index=False)
            cm_df.to_csv(f"summary_{base}_confusion_matrix.csv", index=False)
            os.makedirs("confusion_matrix_plots", exist_ok=True)
            save_confusion_heatmap(cm_df, f"Confusion Matrix: {name} (α={alpha})",
                                   f"confusion_matrix_plots/summary_{base}_confusion_matrix.png")

            acc_mean, acc_sd, acc_ms = mean_sd_str(fold_acc)
            mf1_mean, mf1_sd, mf1_ms = mean_sd_str(fold_macro)
            wf1_mean, wf1_sd, wf1_ms = mean_sd_str(fold_weighted)

            fold_df = pd.DataFrame({"fold": list(range(1, N_OUTER + 1)), "Accuracy": fold_acc, "MacroF1": fold_macro, "WeightedF1": fold_weighted})
            fold_df.to_csv(f"fold_metrics/{base}.csv", index=False)

            print(f"\n== {name} | α={alpha} ==")
            print(f"Acc     mean ± SD: {acc_ms}")
            print(f"MacroF1 mean ± SD: {mf1_ms}")
            print(f"WeightF1 mean ± SD: {wf1_ms}")

            per_alpha_rows.append({
                "Classifier": f"Choquet+Sugeno-FFNN (α={alpha})",
                "Dataset": name,
                "Accuracy": round(float(np.mean(fold_acc)), 4),
                "Macro F1": round(float(np.mean(fold_macro)), 4),
                "Weighted F1": round(float(np.mean(fold_weighted)), 4),
                "Acc_mean": round(acc_mean, 4), "Acc_sd": round(acc_sd, 4), "Acc_mean±SD": acc_ms,
                "MacroF1_mean": round(mf1_mean, 4), "MacroF1_sd": round(mf1_sd, 4), "MacroF1_mean±SD": mf1_ms,
                "WeightedF1_mean": round(wf1_mean, 4), "WeightedF1_sd": round(wf1_sd, 4), "WeightedF1_mean±SD": wf1_ms
            })
            print(f"Saved metrics & CM for {name} (α={alpha})")

        df = pd.DataFrame(per_alpha_rows)
        df.to_csv(f"choquet_sugeno_fusion_alpha_{str(alpha).replace('.', '_')}.csv", index=False)
        all_results.extend(per_alpha_rows)

    grand = pd.DataFrame(all_results)
    grand.to_csv("choquet_sugeno_fusion_all_alphas_summary_GAE_7datasets.csv", index=False)
    print("\n✅ Done. Saved:")
    print("  - fold_metrics/<combo>_alpha_<α>.csv (per-fold metrics)")
    print("  - choquet_sugeno_fusion_all_alphas_summary_GAE_7datasets.csv (grand means + mean±SD)")

# =========================================================
# PART 2: α=0.8 for all classifiers & modes (EMB, RAWEMB)
# RAWEMB strictly uses your already-saved combined_<modality>.csv (or in-memory combined_<modality>)
# =========================================================
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def _exists(name: str) -> bool:
    return (name in globals()) and (globals()[name] is not None)

def _get_var(name: str):
    if not _exists(name):
        raise RuntimeError(f"In-memory variable '{name}' is required but not found.")
    return globals()[name]



# RAW+EMB combos (maps to combined_<modality>.csv)
RAWEMB_COMBOS = {
    "Clinical": [combined_clinical],
    #"SElected_clinical": [selected_clinical_np],
    "CNA":      [combined_cna],
    "GSE":      [combined_gse],
    #"DNA":      [combined_dna],
    #"miRNA":    [combined_mirna],
    "Mutation": [combined_mutation],
    "COEXP":    [combined_coexp],
    "All_comb":      [selected_clinical_np, combined_cna, combined_gse, combined_mutation, combined_coexp],
    #"All_sel_sel":  [selected_clinical_np, latent_cna, latent_gse, latent_dna, latent_mirna, latent_mutation, selected_coexp_np]
}

# Classifiers
def make_svm():  return SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale", random_state=SEED)
def make_rf():   return RandomForestClassifier(n_estimators=400, max_depth=None, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
def make_mlp():  return MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu", solver="adam",
                                      alpha=1e-4, learning_rate="adaptive", max_iter=200, random_state=SEED)
def make_xgb():  return XGBClassifier(
    n_estimators=600, max_depth=6, subsample=0.9, colsample_bytree=0.8,
    learning_rate=0.03, reg_lambda=1.0, objective="multi:softprob",
    num_class=len(ALL_CLASSES), eval_metric="mlogloss", random_state=SEED, n_jobs=-1
)

CLASSIFIERS = {
    "FFNN": "FFNN",   # use your PyTorch FFNN from Part 1
    "SVM":  make_svm,
    "RF":   make_rf,
    "MLP":  make_mlp,
    "XGB":  make_xgb,
}

MODEL_PREFIX = "CS"     # short name instead of SUPREME
ALPHA = 0.8
MODES = ["EMB", "RAWEMB"]

def fit_predict_ffnn(X_train, y_train, X_val, y_val, X_test):
    model, scaler = retrain_on_outer_train(X_train, y_train, val_size=0.15)
    probs = predict_proba(model, scaler, X_test)
    y_pred = probs.argmax(axis=1)
    return probs, y_pred

def fit_predict_sklearn(clf, X_train, y_train, X_val, y_val, X_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    smote = SMOTE(random_state=SEED)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)

    clf.fit(X_tr_sm, y_tr_sm)
    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("Classifier lacks predict_proba")
    probs = clf.predict_proba(X_te)
    y_pred = probs.argmax(axis=1)
    return probs, y_pred

# -------- Main unified runner (EMB + RAWEMB, α=0.8, all classifiers) --------
os.makedirs("fold_metrics_all", exist_ok=True)

def run_all_classifiers_and_modes():
    s1_rows = []
    EMB_COMBOS = modality_combinations  # use your Part-1 combos directly

    for mode in MODES:
        print(f"\n\n{'#'*22} MODE = {mode} {'#'*22}")
        combos = EMB_COMBOS if mode == "EMB" else RAWEMB_COMBOS

        for combo_name, modality_list in combos.items():
            print(f"\n{'='*70}\nDataset Combo: {combo_name} | Mode: {mode} | α={ALPHA}\n{'='*70}")

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for clf_name, factory in CLASSIFIERS.items():
                model_code = f"{MODEL_PREFIX}-{clf_name}"
                print(f"\n--- Running {model_code} on {combo_name} ({mode}) ---")

                fold_acc, fold_macro, fold_weighted = [], [], []

                for fold, (tr_idx, te_idx) in enumerate(skf.split(labels, labels), start=1):
                    y_tr, y_te = labels[tr_idx], labels[te_idx]

                    weights = []
                    per_mod_test_probs = []

                    for mod_arr in modality_list:
                        X = ensure_float32(mod_arr)
                        X_tr, X_te = X[tr_idx], X[te_idx]

                        w = inner_cv_weight(X_tr, y_tr, n_splits=5)  # same concept as Part-1
                        weights.append(w)

                        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                        tr2_idx, va_idx = next(sss.split(X_tr, y_tr))

                        if clf_name == "FFNN":
                            probs_te, y_pred_te = fit_predict_ffnn(X_tr, y_tr, X_tr[va_idx], y_tr[va_idx], X_te)
                        else:
                            clf = factory() if callable(factory) else factory
                            probs_te, y_pred_te = fit_predict_sklearn(clf, X_tr, y_tr, X_tr[va_idx], y_tr[va_idx], X_te)

                        per_mod_test_probs.append(probs_te)

                    fused = fuse_probs(per_mod_test_probs, weights, ALPHA)
                    y_pred = fused.argmax(axis=1)

                    acc = accuracy_score(y_te, y_pred)
                    mf1 = f1_score(y_te, y_pred, average='macro')
                    wf1 = f1_score(y_te, y_pred, average='weighted')

                    fold_acc.append(acc); fold_macro.append(mf1); fold_weighted.append(wf1)
                    print(f"Fold {fold:02d}  Acc={acc:.4f}  MacroF1={mf1:.4f}  WeightedF1={wf1:.4f}")

                    s1_rows.append({
                        "Dataset": combo_name,
                        "Mode": mode,                        # EMB vs RAWEMB
                        "Model": model_code,                 # CS-FFNN / CS-SVM / ...
                        "Alpha": ALPHA,
                        "Fold": fold,
                        "Accuracy": acc,
                        "MacroF1": mf1,
                        "WeightedF1": wf1,
                    })

                base = f"{combo_name.replace('+','_')}_{mode}_{model_code}_alpha_{str(ALPHA).replace('.', '_')}"
                pd.DataFrame({
                    "fold": list(range(1, len(fold_acc) + 1)),
                    "Accuracy": fold_acc,
                    "MacroF1": fold_macro,
                    "WeightedF1": fold_weighted
                }).to_csv(f"fold_metrics_all/{base}.csv", index=False)

    s1_df = pd.DataFrame(s1_rows)
    s1_df.to_csv("ALL_folds_models_modes_alpha0p8.csv", index=False)
    print("\n✅ Saved master per-fold file: ALL_folds_models_modes_alpha0p8.csv")

    grp = s1_df.groupby(["Dataset", "Mode"])["MacroF1"]
    s1_like = pd.DataFrame({
        "Min_MacroF1": grp.min(),
        "Median_MacroF1": grp.median(),
        "Max_MacroF1": grp.max(),
    }).reset_index()
    s1_like.to_csv("Table_S1_like_MacroF1_by_Dataset_Mode_alpha0p8.csv", index=False)
    print("✅ Saved S1-like summary: Table_S1_like_MacroF1_by_Dataset_Mode_alpha0p8.csv")

    grp2 = s1_df.groupby(["Dataset", "Mode", "Model"])["MacroF1"]
    per_model = pd.DataFrame({
        "Mean_MacroF1": grp2.mean(),
        "SD_MacroF1": grp2.std(ddof=1),
        "Min_MacroF1": grp2.min(),
        "Median_MacroF1": grp2.median(),
        "Max_MacroF1": grp2.max(),
    }).reset_index()
    per_model.to_csv("PerModel_MacroF1_by_Dataset_Mode_Model_alpha0p8.csv", index=False)
    print("✅ Saved per-model details: PerModel_MacroF1_by_Dataset_Mode_Model_alpha0p8.csv")

# ------------------ RUN unified Part 2 now ------------------
run_all_classifiers_and_modes()



# ===============================================
# TCGA-wide tables (datasets as columns; methods as rows)
# Builds:
#   - S3: three wide tables (MacroF1 / WeightedF1 / Accuracy)
#   - S1: one wide table (MacroF1)
# Requires: ALL_folds_models_modes_alpha0p8.csv
# ===============================================
import os
import numpy as np
import pandas as pd

MASTER = "ALL_folds_models_modes_alpha0p8.csv"
ALPHA_TAG = "alpha0p8"

# TCGA = 7 modalities
DATASETS_TCGA = ["Clinical", "CNA", "GSE", "DNA", "miRNA", "Mutation", "COEXP"]
VALID_MODES = ["EMB", "RAWEMB"]  # you produce both

if not os.path.exists(MASTER):
    raise FileNotFoundError(
        f"Missing '{MASTER}'. Run your EMB/RAWEMB runners first to produce this file."
    )

df = pd.read_csv(MASTER)

# sanity
req = {"Dataset", "Mode", "Model", "Fold", "Accuracy", "MacroF1", "WeightedF1"}
missing = req - set(df.columns)
if missing:
    raise ValueError(f"Master file missing columns: {missing}")

# keep only TCGA modalities and valid modes
df = df[df["Dataset"].isin(DATASETS_TCGA) & df["Mode"].isin(VALID_MODES)].copy()
if df.empty:
    raise ValueError("No TCGA rows for EMB/RAWEMB in the master file.")

# ---------------- helpers ----------------
def mean_sd_fmt(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return "0.0000 ± 0.0000"
    m = np.mean(x)
    sd = np.std(x, ddof=1) if x.size > 1 else 0.0
    return f"{m:.4f} ± {sd:.4f}"

def summarize_metric(g, col):
    arr = g[col].astype(float).values
    return pd.Series({
        "mean":    float(np.mean(arr)) if arr.size else 0.0,
        "sd":      float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min":     float(np.min(arr)) if arr.size else 0.0,
        "median":  float(np.median(arr)) if arr.size else 0.0,
        "max":     float(np.max(arr)) if arr.size else 0.0,
        "mean±sd": mean_sd_fmt(arr),
    })

def metric_block(g):
    """Summarize Accuracy, MacroF1, WeightedF1 for one (Dataset, Mode, Model)."""
    acc = summarize_metric(g, "Accuracy").add_prefix("Accuracy_")
    mac = summarize_metric(g, "MacroF1").add_prefix("MacroF1_")
    wgt = summarize_metric(g, "WeightedF1").add_prefix("WeightedF1_")
    return pd.concat([acc, mac, wgt])

# ========= Build tall per-(Dataset,Mode,Model) summary (like S3 base) =========
s3_tall = (
    df.groupby(["Dataset", "Mode", "Model"], as_index=False)
      .apply(metric_block)
      .reset_index(drop=True)
)

# ---------------- S3 (wide) ----------------
# We’ll make THREE separate wide tables: MacroF1 / WeightedF1 / Accuracy
def make_s3_wide(metric_name: str, value_suffix="mean±sd"):
    """
    metric_name ∈ {"MacroF1","WeightedF1","Accuracy"}
    Produces a wide table with rows = (Mode, Model) and columns = datasets.
    Each cell is '<mean> ± <sd>' for that metric aggregated over folds.
    """
    col_name = f"{metric_name}_{value_suffix}"  # e.g., "MacroF1_mean±sd"
    # pivot with datasets as columns
    wide = s3_tall.pivot_table(
        index=["Mode", "Model"],
        columns="Dataset",
        values=col_name,
        aggfunc="first"
    )
    # ensure proper dataset column order
    wide = wide.reindex(columns=DATASETS_TCGA)
    # sort rows for readability
    wide = wide.sort_index(level=["Mode", "Model"])
    wide = wide.reset_index()
    return wide

s3_macro = make_s3_wide("MacroF1")
s3_wf1   = make_s3_wide("WeightedF1")
s3_acc   = make_s3_wide("Accuracy")

s3_macro.to_csv(f"TCGA_S3_wide_MacroF1_{ALPHA_TAG}.csv", index=False)
s3_wf1.to_csv(  f"TCGA_S3_wide_WeightedF1_{ALPHA_TAG}.csv", index=False)
s3_acc.to_csv(  f"TCGA_S3_wide_Accuracy_{ALPHA_TAG}.csv", index=False)
print("✅ Saved:")
print("  - TCGA_S3_wide_MacroF1_", ALPHA_TAG, ".csv", sep="")
print("  - TCGA_S3_wide_WeightedF1_", ALPHA_TAG, ".csv", sep="")
print("  - TCGA_S3_wide_Accuracy_", ALPHA_TAG, ".csv", sep="")

# ---------------- S1 (wide) ----------------
# You asked: “Same for S1, dataset in columns and in rows method svm rf raw without raw”
# → Rows = (Mode, Model), Columns = datasets, Values = MacroF1 (mean±sd)
# This is basically the same pivot as S3 MacroF1, but saved as S1-wide.
s1_wide = s3_macro.copy()  # identical structure fits your requested shape
s1_wide.to_csv(f"TCGA_S1_wide_MacroF1_{ALPHA_TAG}.csv", index=False)
print("✅ Saved: TCGA_S1_wide_MacroF1_", ALPHA_TAG, ".csv", sep="")


# ===============================================
# Multiclass ROC–AUC plots for FFNN (EMB + RAWEMB)
# Looks for both filename styles you used.
# Saves PNG plots + per-class AUC CSVs.
# ===============================================
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ---- config ----
os.makedirs("roc_auc_plots_metabric", exist_ok=True)

# Subtype map (fixed order)
subtype_names = {0:'LumB', 1:'Normal', 2:'Her2', 3:'LumA', 4:'Basal'}
classes = list(subtype_names.keys())
class_labels = [subtype_names[i] for i in classes]

# Datasets to try (you can add/remove)
DATASETS = [
    "Clinical","CNA","GSE","DNA","miRNA","Mutation","COEXP",
    "All_sel","All_comb"
]
# Modes we support in file naming
MODES = ["EMB", "RAWEMB"]

# Alpha tokens to try in filenames (cover both 0.8 and 0_8 variants)
ALPHA_TOKENS = ["0.8", "0_8"]

def find_npys_for(dataset, mode):
    """
    Return (y_true_path, y_prob_path) or (None, None) for the given dataset & mode,
    trying both naming conventions used in your Part-1 (EMB) and Part-2 (RAWEMB).
    """
    # 1) EMB (Part-1) — no model code in filename
    if mode == "EMB":
        # e.g., all_y_true_Clinical_alpha_0.8.npy
        for a in ALPHA_TOKENS:
            yt = f"all_y_true_{dataset.replace('+','_')}_alpha_{a}.npy"
            yp = f"all_y_prob_{dataset.replace('+','_')}_alpha_{a}.npy"
            if os.path.exists(yt) and os.path.exists(yp):
                return yt, yp

    # 2) RAWEMB (Part-2 FFNN) — includes mode + model code
    if mode == "RAWEMB":
        # e.g., all_y_true_Clinical_RAWEMB_CS-FFNN_alpha_0_8.npy
        for a in ALPHA_TOKENS:
            yt = f"all_y_true_{dataset.replace('+','_')}_RAWEMB_CS-FFNN_alpha_{a}.npy"
            yp = f"all_y_prob_{dataset.replace('+','_')}_RAWEMB_CS-FFNN_alpha_{a}.npy"
            if os.path.exists(yt) and os.path.exists(yp):
                return yt, yp

    return None, None

def plot_multiclass_roc(y_true, y_prob, title, out_png, out_csv):
    """
    y_true: shape (n,)
    y_prob: shape (n, C)
    Saves ROC plot and AUC table (per class + micro + macro).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # binarize
    y_bin = label_binarize(y_true, classes=classes)

    # compute per-class ROC + AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i, lab in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            # no positives for this class in pooled set
            fpr[i], tpr[i], roc_auc[i] = np.array([0,1]), np.array([0,1]), float("nan")
        else:
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average (average AUC across classes that have positives)
    valid_aucs = [roc_auc[i] for i in classes if not np.isnan(roc_auc[i])]
    roc_auc["macro"] = float(np.mean(valid_aucs)) if len(valid_aucs) else float("nan")

    # ---- plot ----
    plt.figure(figsize=(9, 6))
    # per-class curves
    for i, name in enumerate(class_labels):
        if np.isnan(roc_auc[i]):
            # skip plotting if undefined
            continue
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{name} (AUC = {roc_auc[i]:.3f})")
    # micro
    plt.plot(fpr["micro"], tpr["micro"], lw=2.0, linestyle="--", label=f"micro-average (AUC = {roc_auc['micro']:.3f})")
    # diagonal
    plt.plot([0,1], [0,1], lw=1.0, linestyle=":", color="black")

    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # ---- table (CSV) ----
    rows = []
    for i, name in enumerate(class_labels):
        rows.append({"Class": name, "AUC": None if np.isnan(roc_auc[i]) else float(roc_auc[i])})
    rows.append({"Class": "micro", "AUC": float(roc_auc["micro"]) if not np.isnan(roc_auc["micro"]) else None})
    rows.append({"Class": "macro", "AUC": float(roc_auc["macro"]) if not np.isnan(roc_auc["macro"]) else None})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# ---- run over all datasets & modes ----
any_found = False
for dataset in DATASETS:
    for mode in MODES:
        ytp, ypp = find_npys_for(dataset, mode)
        if not ytp:
            # silently skip if not present
            continue
        y_true = np.load(ytp).astype(int)
        y_prob = np.load(ypp).astype(float)

        title = f"ROC–AUC (FFNN, {dataset}, {mode})"
        safe_name = f"{dataset.replace('+','_')}_{mode}_FFNN"
        out_png = os.path.join("roc_auc_plots", f"ROC_{safe_name}.png")
        out_csv = os.path.join("roc_auc_plots", f"ROC_{safe_name}.csv")

        print(f"[OK] {dataset} / {mode} -> {os.path.basename(out_png)}")
        plot_multiclass_roc(y_true, y_prob, title, out_png, out_csv)
        any_found = True

if not any_found:
    print("⚠️ No matching FFNN files found. Expected either:")
    print("  EMB:     all_y_true_<Dataset>_alpha_0.8.npy + all_y_prob_<Dataset>_alpha_0.8.npy")
    print("  RAWEMB:  all_y_true_<Dataset>_RAWEMB_CS-FFNN_alpha_0_8.npy + all_y_prob_... .npy")
    print("Datasets tried:", ", ".join(DATASETS))



# ===============================================
# PRINT multiclass ROC–AUC for FFNN (EMB + RAWEMB)
# - Prints per-class, micro, macro AUC to stdout
# - Optional plots shown inline (SHOW_PLOTS=True)
# - Still saves PNG + CSV if you want artifacts
# ===============================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# -------- config --------
SAVE_ARTIFACTS = True
SHOW_PLOTS     = True   # set False if you only want printed AUCs (no plot popups)
OUTDIR         = "roc_auc_plots_metabric"
os.makedirs(OUTDIR, exist_ok=True)

# consistent class order (adjust if mapping differs)
subtype_names = {0:'LumB', 1:'Normal', 2:'Her2', 3:'LumA', 4:'Basal'}
CLASSES      = list(subtype_names.keys())
CLASS_LABELS = [subtype_names[i] for i in CLASSES]

DATASETS = [
    "Clinical","CNA","GSE","DNA","miRNA","Mutation","COEXP",
    "All_sel","All_comb"
]
MODES = ["EMB","RAWEMB"]
ALPHA_TOKENS = ["0.8","0_8"]  # filename variants you’ve used

# -------- file finder --------
def find_npys_for(dataset, mode):
    ds_safe = dataset.replace("+","_")
    if mode == "EMB":
        for a in ALPHA_TOKENS:
            yt = f"all_y_true_{ds_safe}_alpha_{a}.npy"
            yp = f"all_y_prob_{ds_safe}_alpha_{a}.npy"
            if os.path.exists(yt) and os.path.exists(yp):
                return yt, yp
    if mode == "RAWEMB":
        for a in ALPHA_TOKENS:
            yt = f"all_y_true_{ds_safe}_RAWEMB_CS-FFNN_alpha_{a}.npy"
            yp = f"all_y_prob_{ds_safe}_RAWEMB_CS-FFNN_alpha_{a}.npy"
            if os.path.exists(yt) and os.path.exists(yp):
                return yt, yp
    return None, None

# -------- ROC helpers --------
def compute_multiclass_roc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_bin  = label_binarize(y_true, classes=CLASSES)

    fpr, tpr, roc_auc = {}, {}, {}
    # per-class
    for i in range(len(CLASSES)):
        if y_bin[:, i].sum() == 0:
            fpr[i], tpr[i], roc_auc[i] = np.array([0,1]), np.array([0,1]), float("nan")
        else:
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # macro
    valid = [roc_auc[i] for i in range(len(CLASSES)) if not np.isnan(roc_auc[i])]
    roc_auc["macro"] = float(np.mean(valid)) if valid else float("nan")
    return fpr, tpr, roc_auc

def print_auc_block(dataset, mode, roc_auc):
    print(f"\n===== AUCs — {dataset} [{mode}] (FFNN) =====")
    rows = []
    for i, name in enumerate(CLASS_LABELS):
        v = None if np.isnan(roc_auc[i]) else float(roc_auc[i])
        rows.append([name, v])
    rows.append(["micro", float(roc_auc["micro"])])
    rows.append(["macro", float(roc_auc["macro"])])
    df_print = pd.DataFrame(rows, columns=["Class", "AUC"])
    # pretty-print with 4 decimals, including NaNs
    with pd.option_context('display.float_format', lambda x: f"{x:.4f}"):
        print(df_print.to_string(index=False))

def save_auc_csv(dataset, mode, roc_auc):
    rows = []
    for i, name in enumerate(CLASS_LABELS):
        v = None if np.isnan(roc_auc[i]) else float(roc_auc[i])
        rows.append({"Class": name, "AUC": v})
    rows.append({"Class": "micro", "AUC": float(roc_auc["micro"])})
    rows.append({"Class": "macro", "AUC": float(roc_auc["macro"])})
    out_csv = os.path.join(OUTDIR, f"AUC_{dataset}_{mode}_FFNN.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

def maybe_plot_and_save(dataset, mode, fpr, tpr, roc_auc):
    if not (SAVE_ARTIFACTS or SHOW_PLOTS):
        return
    plt.figure(figsize=(9,6))
    # per-class curves
    for i, name in enumerate(CLASS_LABELS):
        if np.isnan(roc_auc[i]):  # skip classes with no positives
            continue
        plt.plot(fpr[i], tpr[i], lw=1.6, label=f"{name} (AUC={roc_auc[i]:.3f})")
    # micro-average
    plt.plot(fpr["micro"], tpr["micro"], lw=2.2, linestyle="--",
             label=f"micro (AUC={roc_auc['micro']:.3f})")
    # diagonal
    plt.plot([0,1],[0,1], linestyle=":", lw=1.0, color="black")
    plt.title(f"ROC–AUC (FFNN, {dataset}, {mode})")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if SAVE_ARTIFACTS:
        out_png = os.path.join(OUTDIR, f"ROC_{dataset}_{mode}_FFNN.png")
        plt.savefig(out_png, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# -------- runner (prints AUCs; optional plot/save) --------
summary_rows = []  # micro/macro per dataset/mode

any_found = False
for dataset in DATASETS:
    for mode in MODES:
        yt, yp = find_npys_for(dataset, mode)
        if not yt:
            continue
        y_true = np.load(yt).astype(int)
        y_prob = np.load(yp).astype(float)

        fpr, tpr, roc_auc = compute_multiclass_roc(y_true, y_prob)

        # 1) PRINT to stdout (what you asked for)
        print_auc_block(dataset, mode, roc_auc)

        # 2) Optional: save CSV + plot
        if SAVE_ARTIFACTS:
            out_csv = save_auc_csv(dataset, mode, roc_auc)
            # (kept, in case you want a record) print(f"Saved: {out_csv}")
        maybe_plot_and_save(dataset, mode, fpr, tpr, roc_auc)

        # 3) Collect summary for quick comparison table
        summary_rows.append({
            "Dataset": dataset,
            "Mode": mode,
            "AUC_micro": float(roc_auc["micro"]),
            "AUC_macro": float(roc_auc["macro"])
        })
        any_found = True

# -------- summary print --------
if any_found:
    df_sum = pd.DataFrame(summary_rows).sort_values(["Dataset","Mode"])
    with pd.option_context('display.float_format', lambda x: f"{x:.4f}"):
        print("\n===== Summary (micro & macro AUC) across datasets =====")
        print(df_sum.to_string(index=False))
    if SAVE_ARTIFACTS:
        df_sum.to_csv(os.path.join(OUTDIR, "AUC_summary_micro_macro_FFNN.csv"), index=False)
else:
    print("⚠️ No EMB/RAWEMB FFNN .npy files found.")


# ===============================================
# Violin plots for EMB + RAWEMB results (FFNN)
# Uses per-fold results from ALL_folds_models_modes_alpha0p8.csv
# ===============================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MASTER = "ALL_folds_models_modes_alpha0p8.csv"
if not os.path.exists(MASTER):
    raise FileNotFoundError(f"Missing {MASTER}. Run your fold runner first.")

# Load all results
df = pd.read_csv(MASTER)

# sanity check
required_cols = {"Dataset","Mode","Model","Fold","Accuracy","MacroF1","WeightedF1"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

# only EMB + RAWEMB
df = df[df["Mode"].isin(["EMB","RAWEMB"])]

# output folder
os.makedirs("violin_plots_metabric", exist_ok=True)

# -------- function to make violin plots --------
def plot_violin(metric, out_png):
    plt.figure(figsize=(12,6))
    sns.violinplot(
        data=df,
        x="Dataset",
        y=metric,
        hue="Mode",
        split=True,
        inner="quartile",
        palette="Set2"
    )
    plt.title(f"Distribution of {metric} across folds — EMB vs RAWEMB")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.xlabel("Dataset")
    plt.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"✅ Saved {out_png}")

# -------- run for all metrics --------
for metric in ["Accuracy","MacroF1","WeightedF1"]:
    out_png = os.path.join("violin_plots", f"Violin_{metric}_EMB_vs_RAWEMB.png")
    plot_violin(metric, out_png)

# ===============================================
# Confusion Matrix for FFNN (EMB + RAWEMB)
# - Prints raw confusion matrix (numbers)
# - Displays annotated heatmap
# ===============================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---- config ----
subtype_names = {0:'LumB', 1:'Normal', 2:'Her2', 3:'LumA', 4:'Basal'}
classes = list(subtype_names.keys())
class_labels = [subtype_names[i] for i in classes]

DATASETS = ["Clinical","CNA","GSE","DNA","miRNA","Mutation","COEXP","All_sel","All_comb"]
MODES = ["EMB","RAWEMB"]
ALPHA_TOKENS = ["0.8","0_8"]  # file naming styles

def find_npys_for(dataset, mode):
    ds_safe = dataset.replace("+","_")
    if mode == "EMB":
        for a in ALPHA_TOKENS:
            yt = f"all_y_true_{ds_safe}_alpha_{a}.npy"
            yp = f"all_y_prob_{ds_safe}_alpha_{a}.npy"
            if os.path.exists(yt) and os.path.exists(yp):
                return yt, yp
    if mode == "RAWEMB":
        for a in ALPHA_TOKENS:
            yt = f"all_y_true_{ds_safe}_RAWEMB_CS-FFNN_alpha_{a}.npy"
            yp = f"all_y_prob_{ds_safe}_RAWEMB_CS-FFNN_alpha_{a}.npy"
            if os.path.exists(yt) and os.path.exists(yp):
                return yt, yp
    return None, None

def plot_confusion(y_true, y_pred, dataset, mode, save=True):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # print raw numbers
    print(f"\nConfusion Matrix — {dataset} ({mode}, FFNN):")
    print(cm)

    # plot heatmap
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix — {dataset} ({mode}, FFNN)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save:
        out_path = f"roc_auc_plots/CM_{dataset}_{mode}_FFNN.png"
        plt.savefig(out_path, dpi=200)
        print(f"✅ Saved heatmap: {out_path}")
    plt.show()

# ---- runner ----
for dataset in DATASETS:
    for mode in MODES:
        yt, yp = find_npys_for(dataset, mode)
        if not yt:
            continue
        y_true = np.load(yt).astype(int)
        y_prob = np.load(yp).astype(float)
        y_pred = y_prob.argmax(axis=1)

        plot_confusion(y_true, y_pred, dataset, mode)

-----------TCGA With 7 Modalities---------------

# ===============================================
# ONE FILE: Part 1 (your original) + Part 2 (all classifiers, EMB & RAWEMB)
# ===============================================

# ----------------------- Imports -----------------------
import os, random, math, re, glob
import numpy as np
import pandas as pd
from itertools import combinations

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Reproducibility ---------------------- #
SEED = 50
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_all()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- Helpers ---------------------- #
def to_numpy(arr):
    return arr.to_numpy() if hasattr(arr, "to_numpy") else np.asarray(arr)

def ensure_float32(x):
    x = to_numpy(x)
    return x.astype(np.float32, copy=False)

def mean_sd_str(values):
    vals = np.asarray(values, dtype=float)
    m = float(np.mean(vals)) if len(vals) else 0.0
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, sd, f"{m:.4f} ± {sd:.4f}"

# ---------------------- Subtype Mapping ---------------------- #
subtype_names = {0: 'LumB', 1: 'Normal', 2: 'Her2', 3: 'LumA', 4: 'Basal'}
ALL_CLASSES = list(subtype_names.keys())

# ---------------------- Evaluation ---------------------- #
def evaluate_subtypes(y_true, y_pred, y_prob, subtype_names):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_true_bin = label_binarize(y_true, classes=list(subtype_names.keys()))
    rows = []
    cm = confusion_matrix(y_true, y_pred, labels=list(subtype_names.keys()))

    for i, label in subtype_names.items():
        mask = (y_true == i)
        acc = accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else 0.0
        prec = precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        rec  = recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        f1   = f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        auc  = roc_auc_score(y_true_bin[:, i], y_prob[:, i]) if np.sum(y_true_bin[:, i]) > 0 else 0.0

        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - TP - FP - FN
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        rows.append({
            "Subtype": label,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "AUC": round(auc, 4),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4)
        })

    cm_df = pd.DataFrame(cm, index=subtype_names.values(), columns=subtype_names.values())
    return pd.DataFrame(rows), cm_df

def save_confusion_heatmap(cm_df, title, path_png):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

# ------------------------------ Model ------------------------------ #
class FFNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.4):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.cls = nn.Linear(hidden_dim // 4, num_classes)

    def forward(self, x):
        z = self.feat(x)
        logits = self.cls(z)
        return logits

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - 1e-8:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------------------ λ-Choquet & Sugeno ------------------------------ #
def solve_lambda(weights, tol=1e-7, max_iter=100):
    w = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    def f(lmbd):
        return float(np.prod(1.0 + lmbd * w) - (1.0 + lmbd))
    lo = -0.999999
    hi = 1.0
    f_lo, f_hi = f(lo), f(hi)
    while f_lo * f_hi > 0 and hi < 1e6:
        hi *= 2.0
        f_hi = f(hi)
    if f_lo * f_hi > 0:
        return 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))

def choquet_integral(values, weights):
    v = np.asarray(values, dtype=float)
    g = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    order = np.argsort(-v)
    v_sorted = v[order]
    g_sorted = g[order]
    lam = solve_lambda(g_sorted)
    mu_prev = 0.0
    res = 0.0
    for k in range(len(v_sorted)):
        mu_curr = mu_prev + g_sorted[k] + lam * mu_prev * g_sorted[k]
        res += v_sorted[k] * (mu_curr - mu_prev)
        mu_prev = mu_curr
    return float(res)

def sugeno_integral(values, weights):
    v = np.asarray(values, dtype=float)
    g = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    order = np.argsort(-v)
    v_sorted = v[order]
    g_sorted = g[order]
    lam = solve_lambda(g_sorted)
    mu_prev = 0.0
    best = 0.0
    for k in range(len(v_sorted)):
        mu_curr = mu_prev + g_sorted[k] + lam * mu_prev * g_sorted[k]
        best = max(best, min(v_sorted[k], mu_curr))
        mu_prev = mu_curr
    return float(best)

def fuse_probs(list_of_prob_arrays, weights, alpha):
    m = len(list_of_prob_arrays)
    if m == 1:
        return list_of_prob_arrays[0]
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    n, c = list_of_prob_arrays[0].shape
    fused = np.zeros((n, c), dtype=float)
    for i in range(n):
        for cls in range(c):
            vals = [arr[i, cls] for arr in list_of_prob_arrays]
            ch = choquet_integral(vals, w)
            su = sugeno_integral(vals, w)
            fused[i, cls] = alpha * ch + (1 - alpha) * su
    fused = np.maximum(fused, 1e-9)
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused

# ------------------------------ Training utils ------------------------------ #
def train_one(
    X_train, y_train, X_val, y_val,
    hidden_dim=256, max_epochs=200, lr=1e-2, wd=1e-3, patience=10
):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)

    smote = SMOTE(random_state=SEED)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr_sm), y=y_tr_sm)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    model = FFNNClassifier(X_tr_sm.shape[1], hidden_dim, len(ALL_CLASSES)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    early = EarlyStopping(patience=patience)

    X_tr_t = torch.tensor(X_tr_sm, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_sm, dtype=torch.long, device=device)
    X_v_t  = torch.tensor(X_v, dtype=torch.float32, device=device)
    y_v_t  = torch.tensor(y_val, dtype=torch.long, device=device)

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr_t)
        loss = crit(logits, y_tr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_v_t)
            v_loss = crit(v_logits, y_v_t).item()
        sched.step(v_loss)
        early(v_loss, model)
        if early.early_stop:
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    return model, scaler

def predict_proba(model, scaler, X):
    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xt), dim=1).cpu().numpy()
    return probs

def inner_cv_weight(X_train, y_train, n_splits=5):
    skf_in = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr_idx, va_idx in skf_in.split(X_train, y_train):
        model, scaler = train_one(X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx])
        probs = predict_proba(model, scaler, X_train[va_idx])
        y_pred = probs.argmax(axis=1)
        accs.append(accuracy_score(y_train[va_idx], y_pred))
    return float(np.mean(accs))

def retrain_on_outer_train(X_outer_train, y_outer_train, val_size=0.15):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_outer_train, y_outer_train))
    model, scaler = train_one(X_outer_train[tr_idx], y_outer_train[tr_idx],
                              X_outer_train[va_idx], y_outer_train[va_idx])
    return model, scaler

# ---- Data presence/loader helpers ----
def _as_float32(x): return np.asarray(x, dtype=np.float32, copy=False)
def _as_int(x):     return np.asarray(x, dtype=int, copy=False)

def get_var_or_load(name, fname, dtype="float"):
    """Use variable if already in memory; otherwise load from .npy."""
    g = globals()
    if name in g:
        arr = g[name]
    else:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing variable '{name}' and file '{fname}' not found.")
        arr = np.load(fname, allow_pickle=False)
        g[name] = arr  # cache in globals
    return _as_float32(arr) if dtype == "float" else _as_int(arr)

# ---- labels (int) ----
labels = get_var_or_load("labels", "labels.npy", dtype="int")

# ---- 7 latent modalities (float32) ----
latent_clinical  = get_var_or_load("latent_clinical",  "latent_clinical.npy",  dtype="float")
latent_cna       = get_var_or_load("latent_cna",       "latent_cna.npy",       dtype="float")
latent_gse       = get_var_or_load("latent_gse",       "latent_gse.npy",       dtype="float")
latent_dna       = get_var_or_load("latent_dna",       "latent_dna.npy",       dtype="float")
latent_mirna     = get_var_or_load("latent_mirna",     "latent_mirna.npy",     dtype="float")
latent_mutation  = get_var_or_load("latent_mutation",  "latent_mutation.npy",  dtype="float")
latent_coexp     = get_var_or_load("latent_coexp",     "latent_coexp.npy",     dtype="float")

# ---- sanity: all rows must match labels ----
n = labels.shape[0]
for nm, arr in [
    ("latent_clinical",  latent_clinical),
    ("latent_cna",       latent_cna),
    ("latent_gse",       latent_gse),
    ("latent_dna",       latent_dna),
    ("latent_mirna",     latent_mirna),
    ("latent_mutation",  latent_mutation),
    ("latent_COEXP",     latent_coexp),
]:
    if arr.shape[0] != n:
        raise ValueError(f"Row mismatch: {nm} has {arr.shape[0]} rows, but labels has {n} rows.")

# ---- Your modality combos (unchanged concept) ----
modality_combinations = {
    "Clinical": [latent_clinical],
    #"SElected_clinical": [selected_clinical_np],
    "CNA":      [latent_cna],
    "GSE":      [latent_gse],
    "DNA":      [latent_dna],
    "miRNA":    [latent_mirna],
    "Mutation": [latent_mutation],
    "COEXP":    [latent_coexp],
    "All_sel":      [selected_clinical_np, latent_cna, latent_gse, latent_dna, latent_mirna, latent_mutation, latent_coexp],
    #"All_sel_sel":  [selected_clinical_np, latent_cna, latent_gse, latent_dna, latent_mirna, latent_mutation, selected_coexp_np]
}

print("\nConfigured single-modality models:")
for nm, lst in modality_combinations.items():
    print(f" - {nm}: {[a.shape for a in lst]}")

# =========================================================
# (Optional) RUN your original Part-1 main loop
# =========================================================
RUN_BASELINE_PART1 = True   # set True if you want to run your original FFNN-only loop

ALPHAS = [ 0.8]
N_OUTER = 10
N_INNER = 5

if RUN_BASELINE_PART1:
    all_results = []
    os.makedirs("fold_metrics", exist_ok=True)
    for alpha in ALPHAS:
        print(f"\n\n{'#'*30} Fusion α = {alpha} {'#'*30}")
        per_alpha_rows = []
        for name, modality_list in modality_combinations.items():
            print(f"\n{'='*60}\nCombo: {name} | α={alpha}\n{'='*60}")
            skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42)
            fold_acc, fold_macro, fold_weighted = [], [], []
            y_true_all, y_pred_all, y_prob_all = [], [], []
            for fold, (tr_idx, te_idx) in enumerate(skf.split(labels, labels), start=1):
                y_tr, y_te = labels[tr_idx], labels[te_idx]
                weights = []
                per_mod_test_probs = []
                for mod_arr in modality_list:
                    X = ensure_float32(mod_arr)
                    X_tr, X_te = X[tr_idx], X[te_idx]
                    w = inner_cv_weight(X_tr, y_tr, n_splits=N_INNER)
                    weights.append(w)
                    model, scaler = retrain_on_outer_train(X_tr, y_tr, val_size=0.15)
                    probs_te = predict_proba(model, scaler, X_te)
                    per_mod_test_probs.append(probs_te)
                fused = fuse_probs(per_mod_test_probs, weights, alpha)
                y_pred = fused.argmax(axis=1)
                y_true_all.extend(y_te)
                y_pred_all.extend(y_pred)
                y_prob_all.extend(fused)
                acc = accuracy_score(y_te, y_pred)
                mf1 = f1_score(y_te, y_pred, average='macro')
                wf1 = f1_score(y_te, y_pred, average='weighted')
                fold_acc.append(acc); fold_macro.append(mf1); fold_weighted.append(wf1)
                print(f"Fold {fold:02d}  Acc={acc:.4f}  MacroF1={mf1:.4f}  WeightedF1={wf1:.4f}")
                subtype_df_fold, _ = evaluate_subtypes(y_te, y_pred, fused, subtype_names)
                print(subtype_df_fold.to_string(index=False))

            y_true_all = np.asarray(y_true_all); y_pred_all = np.asarray(y_pred_all); y_prob_all = np.asarray(y_prob_all)
            subtype_df, cm_df = evaluate_subtypes(y_true_all, y_pred_all, y_prob_all, subtype_names)
            base = f"{name.replace('+','_')}_alpha_{alpha}"
            np.save(f"all_y_true_{base}.npy", y_true_all)
            np.save(f"all_y_prob_{base}.npy", y_prob_all)
            subtype_df.to_csv(f"summary_{base}_metrics.csv", index=False)
            cm_df.to_csv(f"summary_{base}_confusion_matrix.csv", index=False)
            os.makedirs("confusion_matrix_plots", exist_ok=True)
            save_confusion_heatmap(cm_df, f"Confusion Matrix: {name} (α={alpha})",
                                   f"confusion_matrix_plots/summary_{base}_confusion_matrix.png")

            acc_mean, acc_sd, acc_ms = mean_sd_str(fold_acc)
            mf1_mean, mf1_sd, mf1_ms = mean_sd_str(fold_macro)
            wf1_mean, wf1_sd, wf1_ms = mean_sd_str(fold_weighted)

            fold_df = pd.DataFrame({"fold": list(range(1, N_OUTER + 1)), "Accuracy": fold_acc, "MacroF1": fold_macro, "WeightedF1": fold_weighted})
            fold_df.to_csv(f"fold_metrics/{base}.csv", index=False)

            print(f"\n== {name} | α={alpha} ==")
            print(f"Acc     mean ± SD: {acc_ms}")
            print(f"MacroF1 mean ± SD: {mf1_ms}")
            print(f"WeightF1 mean ± SD: {wf1_ms}")

            per_alpha_rows.append({
                "Classifier": f"Choquet+Sugeno-FFNN (α={alpha})",
                "Dataset": name,
                "Accuracy": round(float(np.mean(fold_acc)), 4),
                "Macro F1": round(float(np.mean(fold_macro)), 4),
                "Weighted F1": round(float(np.mean(fold_weighted)), 4),
                "Acc_mean": round(acc_mean, 4), "Acc_sd": round(acc_sd, 4), "Acc_mean±SD": acc_ms,
                "MacroF1_mean": round(mf1_mean, 4), "MacroF1_sd": round(mf1_sd, 4), "MacroF1_mean±SD": mf1_ms,
                "WeightedF1_mean": round(wf1_mean, 4), "WeightedF1_sd": round(wf1_sd, 4), "WeightedF1_mean±SD": wf1_ms
            })
            print(f"Saved metrics & CM for {name} (α={alpha})")

        df = pd.DataFrame(per_alpha_rows)
        df.to_csv(f"choquet_sugeno_fusion_alpha_{str(alpha).replace('.', '_')}.csv", index=False)
        all_results.extend(per_alpha_rows)

    grand = pd.DataFrame(all_results)
    grand.to_csv("choquet_sugeno_fusion_all_alphas_summary_GAE_7datasets.csv", index=False)
    print("\n✅ Done. Saved:")
    print("  - fold_metrics/<combo>_alpha_<α>.csv (per-fold metrics)")
    print("  - choquet_sugeno_fusion_all_alphas_summary_GAE_7datasets.csv (grand means + mean±SD)")

# =========================================================
# PART 2: α=0.8 for all classifiers & modes (EMB, RAWEMB)
# RAWEMB strictly uses your already-saved combined_<modality>.csv (or in-memory combined_<modality>)
# =========================================================
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def _exists(name: str) -> bool:
    return (name in globals()) and (globals()[name] is not None)

def _get_var(name: str):
    if not _exists(name):
        raise RuntimeError(f"In-memory variable '{name}' is required but not found.")
    return globals()[name]



# RAW+EMB combos (maps to combined_<modality>.csv)
RAWEMB_COMBOS = {
    "Clinical": [combined_clinical],
    #"SElected_clinical": [selected_clinical_np],
    "CNA":      [combined_cna],
    "GSE":      [combined_gse],
    "DNA":      [combined_dna],
    "miRNA":    [combined_mirna],
    "Mutation": [combined_mutation],
    "COEXP":    [combined_coexp],
    "All_comb":      [combined_clinical, combined_cna, combined_gse, combined_dna, combined_mirna, combined_mutation, combined_coexp],
    #"All_sel_sel":  [selected_clinical_np, latent_cna, latent_gse, latent_dna, latent_mirna, latent_mutation, selected_coexp_np]
}

# Classifiers
def make_svm():  return SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale", random_state=SEED)
def make_rf():   return RandomForestClassifier(n_estimators=400, max_depth=None, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
def make_mlp():  return MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu", solver="adam",
                                      alpha=1e-4, learning_rate="adaptive", max_iter=200, random_state=SEED)
def make_xgb():  return XGBClassifier(
    n_estimators=600, max_depth=6, subsample=0.9, colsample_bytree=0.8,
    learning_rate=0.03, reg_lambda=1.0, objective="multi:softprob",
    num_class=len(ALL_CLASSES), eval_metric="mlogloss", random_state=SEED, n_jobs=-1
)

CLASSIFIERS = {
    "FFNN": "FFNN",   # use your PyTorch FFNN from Part 1
    "SVM":  make_svm,
    "RF":   make_rf,
    "MLP":  make_mlp,
    "XGB":  make_xgb,
}

MODEL_PREFIX = "CS"     # short name instead of SUPREME
ALPHA = 0.8
MODES = ["EMB", "RAWEMB"]

def fit_predict_ffnn(X_train, y_train, X_val, y_val, X_test):
    model, scaler = retrain_on_outer_train(X_train, y_train, val_size=0.15)
    probs = predict_proba(model, scaler, X_test)
    y_pred = probs.argmax(axis=1)
    return probs, y_pred

def fit_predict_sklearn(clf, X_train, y_train, X_val, y_val, X_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    smote = SMOTE(random_state=SEED)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)

    clf.fit(X_tr_sm, y_tr_sm)
    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("Classifier lacks predict_proba")
    probs = clf.predict_proba(X_te)
    y_pred = probs.argmax(axis=1)
    return probs, y_pred

# -------- Main unified runner (EMB + RAWEMB, α=0.8, all classifiers) --------
os.makedirs("fold_metrics_all", exist_ok=True)

def run_all_classifiers_and_modes():
    s1_rows = []
    EMB_COMBOS = modality_combinations  # use your Part-1 combos directly

    for mode in MODES:
        print(f"\n\n{'#'*22} MODE = {mode} {'#'*22}")
        combos = EMB_COMBOS if mode == "EMB" else RAWEMB_COMBOS

        for combo_name, modality_list in combos.items():
            print(f"\n{'='*70}\nDataset Combo: {combo_name} | Mode: {mode} | α={ALPHA}\n{'='*70}")

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for clf_name, factory in CLASSIFIERS.items():
                model_code = f"{MODEL_PREFIX}-{clf_name}"
                print(f"\n--- Running {model_code} on {combo_name} ({mode}) ---")

                fold_acc, fold_macro, fold_weighted = [], [], []

                for fold, (tr_idx, te_idx) in enumerate(skf.split(labels, labels), start=1):
                    y_tr, y_te = labels[tr_idx], labels[te_idx]

                    weights = []
                    per_mod_test_probs = []

                    for mod_arr in modality_list:
                        X = ensure_float32(mod_arr)
                        X_tr, X_te = X[tr_idx], X[te_idx]

                        w = inner_cv_weight(X_tr, y_tr, n_splits=5)  # same concept as Part-1
                        weights.append(w)

                        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                        tr2_idx, va_idx = next(sss.split(X_tr, y_tr))

                        if clf_name == "FFNN":
                            probs_te, y_pred_te = fit_predict_ffnn(X_tr, y_tr, X_tr[va_idx], y_tr[va_idx], X_te)
                        else:
                            clf = factory() if callable(factory) else factory
                            probs_te, y_pred_te = fit_predict_sklearn(clf, X_tr, y_tr, X_tr[va_idx], y_tr[va_idx], X_te)

                        per_mod_test_probs.append(probs_te)

                    fused = fuse_probs(per_mod_test_probs, weights, ALPHA)
                    y_pred = fused.argmax(axis=1)

                    acc = accuracy_score(y_te, y_pred)
                    mf1 = f1_score(y_te, y_pred, average='macro')
                    wf1 = f1_score(y_te, y_pred, average='weighted')

                    fold_acc.append(acc); fold_macro.append(mf1); fold_weighted.append(wf1)
                    print(f"Fold {fold:02d}  Acc={acc:.4f}  MacroF1={mf1:.4f}  WeightedF1={wf1:.4f}")

                    s1_rows.append({
                        "Dataset": combo_name,
                        "Mode": mode,                        # EMB vs RAWEMB
                        "Model": model_code,                 # CS-FFNN / CS-SVM / ...
                        "Alpha": ALPHA,
                        "Fold": fold,
                        "Accuracy": acc,
                        "MacroF1": mf1,
                        "WeightedF1": wf1,
                    })

                base = f"{combo_name.replace('+','_')}_{mode}_{model_code}_alpha_{str(ALPHA).replace('.', '_')}"
                pd.DataFrame({
                    "fold": list(range(1, len(fold_acc) + 1)),
                    "Accuracy": fold_acc,
                    "MacroF1": fold_macro,
                    "WeightedF1": fold_weighted
                }).to_csv(f"fold_metrics_all/{base}.csv", index=False)

    s1_df = pd.DataFrame(s1_rows)
    s1_df.to_csv("ALL_folds_models_modes_alpha0p8.csv", index=False)
    print("\n✅ Saved master per-fold file: ALL_folds_models_modes_alpha0p8.csv")

    grp = s1_df.groupby(["Dataset", "Mode"])["MacroF1"]
    s1_like = pd.DataFrame({
        "Min_MacroF1": grp.min(),
        "Median_MacroF1": grp.median(),
        "Max_MacroF1": grp.max(),
    }).reset_index()
    s1_like.to_csv("Table_S1_like_MacroF1_by_Dataset_Mode_alpha0p8.csv", index=False)
    print("✅ Saved S1-like summary: Table_S1_like_MacroF1_by_Dataset_Mode_alpha0p8.csv")

    grp2 = s1_df.groupby(["Dataset", "Mode", "Model"])["MacroF1"]
    per_model = pd.DataFrame({
        "Mean_MacroF1": grp2.mean(),
        "SD_MacroF1": grp2.std(ddof=1),
        "Min_MacroF1": grp2.min(),
        "Median_MacroF1": grp2.median(),
        "Max_MacroF1": grp2.max(),
    }).reset_index()
    per_model.to_csv("PerModel_MacroF1_by_Dataset_Mode_Model_alpha0p8.csv", index=False)
    print("✅ Saved per-model details: PerModel_MacroF1_by_Dataset_Mode_Model_alpha0p8.csv")

# ------------------ RUN unified Part 2 now ------------------
run_all_classifiers_and_modes()

------------COMBINED DATASET With 3 Modalities(TCGA-BRCA + METABRIC)
# ===============================================
# ONE FILE: Part 1 (your original) + Part 2 (all classifiers, EMB & RAWEMB)
# ===============================================

# ----------------------- Imports -----------------------
import os, random, math, re, glob
import numpy as np
import pandas as pd
from itertools import combinations

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Reproducibility ---------------------- #
SEED = 50
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_all()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- Helpers ---------------------- #
def to_numpy(arr):
    return arr.to_numpy() if hasattr(arr, "to_numpy") else np.asarray(arr)

def ensure_float32(x):
    x = to_numpy(x)
    return x.astype(np.float32, copy=False)

def mean_sd_str(values):
    vals = np.asarray(values, dtype=float)
    m = float(np.mean(vals)) if len(vals) else 0.0
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, sd, f"{m:.4f} ± {sd:.4f}"

# ---------------------- Subtype Mapping ---------------------- #
subtype_names = {0: 'LumB', 1: 'Normal', 2: 'Her2', 3: 'LumA', 4: 'Basal'}
ALL_CLASSES = list(subtype_names.keys())

# ---------------------- Evaluation ---------------------- #
def evaluate_subtypes(y_true, y_pred, y_prob, subtype_names):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_true_bin = label_binarize(y_true, classes=list(subtype_names.keys()))
    rows = []
    cm = confusion_matrix(y_true, y_pred, labels=list(subtype_names.keys()))

    for i, label in subtype_names.items():
        mask = (y_true == i)
        acc = accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else 0.0
        prec = precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        rec  = recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        f1   = f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        auc  = roc_auc_score(y_true_bin[:, i], y_prob[:, i]) if np.sum(y_true_bin[:, i]) > 0 else 0.0

        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - TP - FP - FN
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        rows.append({
            "Subtype": label,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "AUC": round(auc, 4),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4)
        })

    cm_df = pd.DataFrame(cm, index=subtype_names.values(), columns=subtype_names.values())
    return pd.DataFrame(rows), cm_df

def save_confusion_heatmap(cm_df, title, path_png):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

# ------------------------------ Model ------------------------------ #
class FFNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.4):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.cls = nn.Linear(hidden_dim // 4, num_classes)

    def forward(self, x):
        z = self.feat(x)
        logits = self.cls(z)
        return logits

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - 1e-8:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------------------ λ-Choquet & Sugeno ------------------------------ #
def solve_lambda(weights, tol=1e-7, max_iter=100):
    w = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    def f(lmbd):
        return float(np.prod(1.0 + lmbd * w) - (1.0 + lmbd))
    lo = -0.999999
    hi = 1.0
    f_lo, f_hi = f(lo), f(hi)
    while f_lo * f_hi > 0 and hi < 1e6:
        hi *= 2.0
        f_hi = f(hi)
    if f_lo * f_hi > 0:
        return 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))

def choquet_integral(values, weights):
    v = np.asarray(values, dtype=float)
    g = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    order = np.argsort(-v)
    v_sorted = v[order]
    g_sorted = g[order]
    lam = solve_lambda(g_sorted)
    mu_prev = 0.0
    res = 0.0
    for k in range(len(v_sorted)):
        mu_curr = mu_prev + g_sorted[k] + lam * mu_prev * g_sorted[k]
        res += v_sorted[k] * (mu_curr - mu_prev)
        mu_prev = mu_curr
    return float(res)

def sugeno_integral(values, weights):
    v = np.asarray(values, dtype=float)
    g = np.clip(np.asarray(weights, dtype=float), 1e-9, 1 - 1e-9)
    order = np.argsort(-v)
    v_sorted = v[order]
    g_sorted = g[order]
    lam = solve_lambda(g_sorted)
    mu_prev = 0.0
    best = 0.0
    for k in range(len(v_sorted)):
        mu_curr = mu_prev + g_sorted[k] + lam * mu_prev * g_sorted[k]
        best = max(best, min(v_sorted[k], mu_curr))
        mu_prev = mu_curr
    return float(best)

def fuse_probs(list_of_prob_arrays, weights, alpha):
    m = len(list_of_prob_arrays)
    if m == 1:
        return list_of_prob_arrays[0]
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    n, c = list_of_prob_arrays[0].shape
    fused = np.zeros((n, c), dtype=float)
    for i in range(n):
        for cls in range(c):
            vals = [arr[i, cls] for arr in list_of_prob_arrays]
            ch = choquet_integral(vals, w)
            su = sugeno_integral(vals, w)
            fused[i, cls] = alpha * ch + (1 - alpha) * su
    fused = np.maximum(fused, 1e-9)
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused

# ------------------------------ Training utils ------------------------------ #
def train_one(
    X_train, y_train, X_val, y_val,
    hidden_dim=256, max_epochs=200, lr=1e-2, wd=1e-3, patience=10
):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)

    smote = SMOTE(random_state=SEED)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr_sm), y=y_tr_sm)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    model = FFNNClassifier(X_tr_sm.shape[1], hidden_dim, len(ALL_CLASSES)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
    early = EarlyStopping(patience=patience)

    X_tr_t = torch.tensor(X_tr_sm, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_sm, dtype=torch.long, device=device)
    X_v_t  = torch.tensor(X_v, dtype=torch.float32, device=device)
    y_v_t  = torch.tensor(y_val, dtype=torch.long, device=device)

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr_t)
        loss = crit(logits, y_tr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_v_t)
            v_loss = crit(v_logits, y_v_t).item()
        sched.step(v_loss)
        early(v_loss, model)
        if early.early_stop:
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    return model, scaler

def predict_proba(model, scaler, X):
    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xt), dim=1).cpu().numpy()
    return probs

def inner_cv_weight(X_train, y_train, n_splits=5):
    skf_in = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr_idx, va_idx in skf_in.split(X_train, y_train):
        model, scaler = train_one(X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx])
        probs = predict_proba(model, scaler, X_train[va_idx])
        y_pred = probs.argmax(axis=1)
        accs.append(accuracy_score(y_train[va_idx], y_pred))
    return float(np.mean(accs))

def retrain_on_outer_train(X_outer_train, y_outer_train, val_size=0.15):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_outer_train, y_outer_train))
    model, scaler = train_one(X_outer_train[tr_idx], y_outer_train[tr_idx],
                              X_outer_train[va_idx], y_outer_train[va_idx])
    return model, scaler

# ---- Data presence/loader helpers ----
def _as_float32(x): return np.asarray(x, dtype=np.float32, copy=False)
def _as_int(x):     return np.asarray(x, dtype=int, copy=False)

def get_var_or_load(name, fname, dtype="float"):
    """Use variable if already in memory; otherwise load from .npy."""
    g = globals()
    if name in g:
        arr = g[name]
    else:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing variable '{name}' and file '{fname}' not found.")
        arr = np.load(fname, allow_pickle=False)
        g[name] = arr  # cache in globals
    return _as_float32(arr) if dtype == "float" else _as_int(arr)

# ---- labels (int) ----
labels = get_var_or_load("labels", "labels.npy", dtype="int")

# ---- 7 latent modalities (float32) ----
latent_clinical  = get_var_or_load("latent_clinical",  "latent_clinical.npy",  dtype="float")
latent_cna       = get_var_or_load("latent_cna",       "latent_cna.npy",       dtype="float")
latent_gse       = get_var_or_load("latent_gse",       "latent_gse.npy",       dtype="float")
latent_dna       = get_var_or_load("latent_dna",       "latent_dna.npy",       dtype="float")
latent_mirna     = get_var_or_load("latent_mirna",     "latent_mirna.npy",     dtype="float")
latent_mutation  = get_var_or_load("latent_mutation",  "latent_mutation.npy",  dtype="float")
latent_coexp     = get_var_or_load("latent_coexp",     "latent_coexp.npy",     dtype="float")

# ---- sanity: all rows must match labels ----
n = labels.shape[0]
for nm, arr in [
    ("latent_clinical",  latent_clinical),
    ("latent_cna",       latent_cna),
    ("latent_gse",       latent_gse),
    ("latent_dna",       latent_dna),
    ("latent_mirna",     latent_mirna),
    ("latent_mutation",  latent_mutation),
    ("latent_COEXP",     latent_coexp),
]:
    if arr.shape[0] != n:
        raise ValueError(f"Row mismatch: {nm} has {arr.shape[0]} rows, but labels has {n} rows.")

# ---- Your modality combos (unchanged concept) ----
modality_combinations = {
    "Clinical": [latent_clinical],
    #"SElected_clinical": [selected_clinical_np],
   # "CNA":      [latent_cna],
    "GSE":      [latent_gse],
    #"DNA":      [latent_dna],
   # "miRNA":    [latent_mirna],
    "Mutation": [latent_mutation],
   # "COEXP":    [latent_coexp],
    "All_sel":      [selected_clinical_np, latent_gse, latent_mutation],
    #"All_sel_sel":  [selected_clinical_np, latent_cna, latent_gse, latent_dna, latent_mirna, latent_mutation, selected_coexp_np]
}

print("\nConfigured single-modality models:")
for nm, lst in modality_combinations.items():
    print(f" - {nm}: {[a.shape for a in lst]}")

# =========================================================
# (Optional) RUN your original Part-1 main loop
# =========================================================
RUN_BASELINE_PART1 = True   # set True if you want to run your original FFNN-only loop

ALPHAS = [ 0.8]
N_OUTER = 10
N_INNER = 5

if RUN_BASELINE_PART1:
    all_results = []
    os.makedirs("fold_metrics", exist_ok=True)
    for alpha in ALPHAS:
        print(f"\n\n{'#'*30} Fusion α = {alpha} {'#'*30}")
        per_alpha_rows = []
        for name, modality_list in modality_combinations.items():
            print(f"\n{'='*60}\nCombo: {name} | α={alpha}\n{'='*60}")
            skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42)
            fold_acc, fold_macro, fold_weighted = [], [], []
            y_true_all, y_pred_all, y_prob_all = [], [], []
            for fold, (tr_idx, te_idx) in enumerate(skf.split(labels, labels), start=1):
                y_tr, y_te = labels[tr_idx], labels[te_idx]
                weights = []
                per_mod_test_probs = []
                for mod_arr in modality_list:
                    X = ensure_float32(mod_arr)
                    X_tr, X_te = X[tr_idx], X[te_idx]
                    w = inner_cv_weight(X_tr, y_tr, n_splits=N_INNER)
                    weights.append(w)
                    model, scaler = retrain_on_outer_train(X_tr, y_tr, val_size=0.15)
                    probs_te = predict_proba(model, scaler, X_te)
                    per_mod_test_probs.append(probs_te)
                fused = fuse_probs(per_mod_test_probs, weights, alpha)
                y_pred = fused.argmax(axis=1)
                y_true_all.extend(y_te)
                y_pred_all.extend(y_pred)
                y_prob_all.extend(fused)
                acc = accuracy_score(y_te, y_pred)
                mf1 = f1_score(y_te, y_pred, average='macro')
                wf1 = f1_score(y_te, y_pred, average='weighted')
                fold_acc.append(acc); fold_macro.append(mf1); fold_weighted.append(wf1)
                print(f"Fold {fold:02d}  Acc={acc:.4f}  MacroF1={mf1:.4f}  WeightedF1={wf1:.4f}")
                subtype_df_fold, _ = evaluate_subtypes(y_te, y_pred, fused, subtype_names)
                print(subtype_df_fold.to_string(index=False))

            y_true_all = np.asarray(y_true_all); y_pred_all = np.asarray(y_pred_all); y_prob_all = np.asarray(y_prob_all)
            subtype_df, cm_df = evaluate_subtypes(y_true_all, y_pred_all, y_prob_all, subtype_names)
            base = f"{name.replace('+','_')}_alpha_{alpha}"
            np.save(f"all_y_true_{base}.npy", y_true_all)
            np.save(f"all_y_prob_{base}.npy", y_prob_all)
            subtype_df.to_csv(f"summary_{base}_metrics.csv", index=False)
            cm_df.to_csv(f"summary_{base}_confusion_matrix.csv", index=False)
            os.makedirs("confusion_matrix_plots", exist_ok=True)
            save_confusion_heatmap(cm_df, f"Confusion Matrix: {name} (α={alpha})",
                                   f"confusion_matrix_plots/summary_{base}_confusion_matrix.png")

            acc_mean, acc_sd, acc_ms = mean_sd_str(fold_acc)
            mf1_mean, mf1_sd, mf1_ms = mean_sd_str(fold_macro)
            wf1_mean, wf1_sd, wf1_ms = mean_sd_str(fold_weighted)

            fold_df = pd.DataFrame({"fold": list(range(1, N_OUTER + 1)), "Accuracy": fold_acc, "MacroF1": fold_macro, "WeightedF1": fold_weighted})
            fold_df.to_csv(f"fold_metrics/{base}.csv", index=False)

            print(f"\n== {name} | α={alpha} ==")
            print(f"Acc     mean ± SD: {acc_ms}")
            print(f"MacroF1 mean ± SD: {mf1_ms}")
            print(f"WeightF1 mean ± SD: {wf1_ms}")

            per_alpha_rows.append({
                "Classifier": f"Choquet+Sugeno-FFNN (α={alpha})",
                "Dataset": name,
                "Accuracy": round(float(np.mean(fold_acc)), 4),
                "Macro F1": round(float(np.mean(fold_macro)), 4),
                "Weighted F1": round(float(np.mean(fold_weighted)), 4),
                "Acc_mean": round(acc_mean, 4), "Acc_sd": round(acc_sd, 4), "Acc_mean±SD": acc_ms,
                "MacroF1_mean": round(mf1_mean, 4), "MacroF1_sd": round(mf1_sd, 4), "MacroF1_mean±SD": mf1_ms,
                "WeightedF1_mean": round(wf1_mean, 4), "WeightedF1_sd": round(wf1_sd, 4), "WeightedF1_mean±SD": wf1_ms
            })
            print(f"Saved metrics & CM for {name} (α={alpha})")

        df = pd.DataFrame(per_alpha_rows)
        df.to_csv(f"choquet_sugeno_fusion_alpha_{str(alpha).replace('.', '_')}.csv", index=False)
        all_results.extend(per_alpha_rows)

    grand = pd.DataFrame(all_results)
    grand.to_csv("choquet_sugeno_fusion_all_alphas_summary_GAE_7datasets.csv", index=False)
    print("\n✅ Done. Saved:")
    print("  - fold_metrics/<combo>_alpha_<α>.csv (per-fold metrics)")
    print("  - choquet_sugeno_fusion_all_alphas_summary_GAE_7datasets.csv (grand means + mean±SD)")

# =========================================================
# PART 2: α=0.8 for all classifiers & modes (EMB, RAWEMB)
# RAWEMB strictly uses your already-saved combined_<modality>.csv (or in-memory combined_<modality>)
# =========================================================
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def _exists(name: str) -> bool:
    return (name in globals()) and (globals()[name] is not None)

def _get_var(name: str):
    if not _exists(name):
        raise RuntimeError(f"In-memory variable '{name}' is required but not found.")
    return globals()[name]



# RAW+EMB combos (maps to combined_<modality>.csv)
RAWEMB_COMBOS = {
    "Clinical": [combined_clinical],
    #"SElected_clinical": [selected_clinical_np],
   # "CNA":      [combined_cna],
    "GSE":      [combined_gse],
    #"DNA":      [combined_dna],
    #"miRNA":    [combined_mirna],
    "Mutation": [combined_mutation],
   # "COEXP":    [combined_coexp],
    "All_comb":      [combined_clinical, combined_gse, combined_mutation],
    "All_comb_with_sel":      [selected_clinical_np, combined_gse, combined_mutation],

    #"All_sel_sel":  [selected_clinical_np, latent_cna, latent_gse, latent_dna, latent_mirna, latent_mutation, selected_coexp_np]
}

# Classifiers
def make_svm():  return SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale", random_state=SEED)
def make_rf():   return RandomForestClassifier(n_estimators=400, max_depth=None, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
def make_mlp():  return MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu", solver="adam",
                                      alpha=1e-4, learning_rate="adaptive", max_iter=200, random_state=SEED)
def make_xgb():  return XGBClassifier(
    n_estimators=600, max_depth=6, subsample=0.9, colsample_bytree=0.8,
    learning_rate=0.03, reg_lambda=1.0, objective="multi:softprob",
    num_class=len(ALL_CLASSES), eval_metric="mlogloss", random_state=SEED, n_jobs=-1
)

CLASSIFIERS = {
    "FFNN": "FFNN",   # use your PyTorch FFNN from Part 1
    "SVM":  make_svm,
    "RF":   make_rf,
    "MLP":  make_mlp,
    "XGB":  make_xgb,
}

MODEL_PREFIX = "CS"     # short name instead of SUPREME
ALPHA = 0.8
MODES = ["EMB", "RAWEMB"]

def fit_predict_ffnn(X_train, y_train, X_val, y_val, X_test):
    model, scaler = retrain_on_outer_train(X_train, y_train, val_size=0.15)
    probs = predict_proba(model, scaler, X_test)
    y_pred = probs.argmax(axis=1)
    return probs, y_pred

def fit_predict_sklearn(clf, X_train, y_train, X_val, y_val, X_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    smote = SMOTE(random_state=SEED)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)

    clf.fit(X_tr_sm, y_tr_sm)
    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("Classifier lacks predict_proba")
    probs = clf.predict_proba(X_te)
    y_pred = probs.argmax(axis=1)
    return probs, y_pred

# -------- Main unified runner (EMB + RAWEMB, α=0.8, all classifiers) --------
os.makedirs("fold_metrics_all", exist_ok=True)

def run_all_classifiers_and_modes():
    s1_rows = []
    EMB_COMBOS = modality_combinations  # use your Part-1 combos directly

    for mode in MODES:
        print(f"\n\n{'#'*22} MODE = {mode} {'#'*22}")
        combos = EMB_COMBOS if mode == "EMB" else RAWEMB_COMBOS

        for combo_name, modality_list in combos.items():
            print(f"\n{'='*70}\nDataset Combo: {combo_name} | Mode: {mode} | α={ALPHA}\n{'='*70}")

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for clf_name, factory in CLASSIFIERS.items():
                model_code = f"{MODEL_PREFIX}-{clf_name}"
                print(f"\n--- Running {model_code} on {combo_name} ({mode}) ---")

                fold_acc, fold_macro, fold_weighted = [], [], []

                for fold, (tr_idx, te_idx) in enumerate(skf.split(labels, labels), start=1):
                    y_tr, y_te = labels[tr_idx], labels[te_idx]

                    weights = []
                    per_mod_test_probs = []

                    for mod_arr in modality_list:
                        X = ensure_float32(mod_arr)
                        X_tr, X_te = X[tr_idx], X[te_idx]

                        w = inner_cv_weight(X_tr, y_tr, n_splits=5)  # same concept as Part-1
                        weights.append(w)

                        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                        tr2_idx, va_idx = next(sss.split(X_tr, y_tr))

                        if clf_name == "FFNN":
                            probs_te, y_pred_te = fit_predict_ffnn(X_tr, y_tr, X_tr[va_idx], y_tr[va_idx], X_te)
                        else:
                            clf = factory() if callable(factory) else factory
                            probs_te, y_pred_te = fit_predict_sklearn(clf, X_tr, y_tr, X_tr[va_idx], y_tr[va_idx], X_te)

                        per_mod_test_probs.append(probs_te)

                    fused = fuse_probs(per_mod_test_probs, weights, ALPHA)
                    y_pred = fused.argmax(axis=1)

                    acc = accuracy_score(y_te, y_pred)
                    mf1 = f1_score(y_te, y_pred, average='macro')
                    wf1 = f1_score(y_te, y_pred, average='weighted')

                    fold_acc.append(acc); fold_macro.append(mf1); fold_weighted.append(wf1)
                    print(f"Fold {fold:02d}  Acc={acc:.4f}  MacroF1={mf1:.4f}  WeightedF1={wf1:.4f}")

                    s1_rows.append({
                        "Dataset": combo_name,
                        "Mode": mode,                        # EMB vs RAWEMB
                        "Model": model_code,                 # CS-FFNN / CS-SVM / ...
                        "Alpha": ALPHA,
                        "Fold": fold,
                        "Accuracy": acc,
                        "MacroF1": mf1,
                        "WeightedF1": wf1,
                    })

                base = f"{combo_name.replace('+','_')}_{mode}_{model_code}_alpha_{str(ALPHA).replace('.', '_')}"
                pd.DataFrame({
                    "fold": list(range(1, len(fold_acc) + 1)),
                    "Accuracy": fold_acc,
                    "MacroF1": fold_macro,
                    "WeightedF1": fold_weighted
                }).to_csv(f"fold_metrics_all/{base}.csv", index=False)

    s1_df = pd.DataFrame(s1_rows)
    s1_df.to_csv("ALL_folds_models_modes_alpha0p8_both.csv", index=False)
    print("\n✅ Saved master per-fold file: ALL_folds_models_modes_alpha0p8_both.csv")

    grp = s1_df.groupby(["Dataset", "Mode"])["MacroF1"]
    s1_like = pd.DataFrame({
        "Min_MacroF1": grp.min(),
        "Median_MacroF1": grp.median(),
        "Max_MacroF1": grp.max(),
    }).reset_index()
    s1_like.to_csv("Table_S1_like_MacroF1_by_Dataset_Mode_alpha0p8_both.csv", index=False)
    print("✅ Saved S1-like summary: Table_S1_like_MacroF1_by_Dataset_Mode_alpha0p8_both.csv")

    grp2 = s1_df.groupby(["Dataset", "Mode", "Model"])["MacroF1"]
    per_model = pd.DataFrame({
        "Mean_MacroF1": grp2.mean(),
        "SD_MacroF1": grp2.std(ddof=1),
        "Min_MacroF1": grp2.min(),
        "Median_MacroF1": grp2.median(),
        "Max_MacroF1": grp2.max(),
    }).reset_index()
    per_model.to_csv("PerModel_MacroF1_by_Dataset_Mode_Model_alpha0p8_both.csv", index=False)
    print("✅ Saved per-model details: PerModel_MacroF1_by_Dataset_Mode_Model_alpha0p8_both.csv")

# ------------------ RUN unified Part 2 now ------------------
run_all_classifiers_and_modes()


