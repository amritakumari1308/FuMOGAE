## -*- coding: -*-
## FuMOGAE , GAE is applied to six omic modalities (CNA, GSE, DNA-Methylation, miRNA, Mutation, and
## Co-expression) after hybrid Boruta + mRMR feature selection. Latent embeddings extracted from these
## six GAEs are concatenated with the Clinical (CLN) features to form a unified representation.
## The final breast cancer subtype classification is performed using a hybrid Choquet–Sugeno fuzzy integral.
------------METABRIC DATASET(CLINICAL,CNA,GSE,MUTATION,COEXPRESSION) Modalities classification
# ============================================================
#  s-20: FINAL CLASSIFICATION + CHOQUET + SUGENO + HYBRID (ADD WEIGHTED F1)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# =========================
# 🔥 ADD THIS (INITIALIZE LISTS)
# =========================
#all_acc_sug, all_f1m_sug, all_f1w_sug, all_auc_sug = [], [], [], []
#all_acc_chq, all_f1m_chq, all_f1w_chq, all_auc_chq = [], [], [], []
#all_acc_hyb, all_f1m_hyb, all_f1w_hyb, all_auc_hyb = [], [], [], []


# =========================
# SAFE TENSOR → NUMPY
# =========================
z_train_np = z_train.detach().cpu().numpy().copy()
z_test_np  = z_test.detach().cpu().numpy().copy()

y_train_np = labels[train_idx].copy()
y_test_np  = labels[test_idx].copy()

assert z_train_np.shape[0] == len(y_train_np)
assert z_test_np.shape[0]  == len(y_test_np)


# =========================
# SMOTE
# =========================
try:
    z_train_np, y_train_np = SMOTE().fit_resample(z_train_np, y_train_np)
except:
    pass


# =========================
# VALIDATION SPLIT
# =========================
X_tr, X_val, y_tr, y_val = train_test_split(
    z_train_np, y_train_np,
    test_size=0.2,
    stratify=y_train_np,
    random_state=42
)


# =========================
# MODELS
# =========================
class FFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# =========================
# TRAIN NN
# =========================
def train_nn(model, X_train, y_train, X_test):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test) / 1.3
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


# =========================
# BUILD μ
# =========================
def build_mu(probs_list, y_true):

    accs = np.array([
        accuracy_score(y_true, np.argmax(p, axis=1))
        for p in probs_list
    ])

    weights = accs / (np.sum(accs) + 1e-8)

    mu = {(): 0}

    for i in range(len(probs_list)):
        mu[(i,)] = weights[i]

    for i in range(len(probs_list)):
        for j in range(i + 1, len(probs_list)):
            mu[(i, j)] = min(1.0, weights[i] + weights[j] - 0.4 * weights[i] * weights[j])

    mu[tuple(range(len(probs_list)))] = 1

    return mu


# =========================
# CHOQUET
# =========================
def choquet(probs, mu):

    n, N, C = probs.shape
    out = np.zeros((N, C))

    for i in range(N):
        for c in range(C):

            f = probs[:, i, c]
            idx = np.argsort(f)
            f = f[idx]

            prev = 0
            total = 0

            for k in range(n):
                subset = tuple(sorted(idx[k:]))
                total += (f[k] - prev) * mu.get(subset, 0)
                prev = f[k]

            out[i, c] = total

    return out


# =========================
# SUGENO
# =========================
def sugeno(probs, mu):

    n, N, C = probs.shape
    out = np.zeros((N, C))

    for i in range(N):
        for c in range(C):

            f = probs[:, i, c]
            idx = np.argsort(f)
            f = f[idx]

            vals = []
            for k in range(n):
                subset = tuple(sorted(idx[k:]))
                vals.append(min(f[k], mu.get(subset, 0)))

            out[i, c] = max(vals)

    return out


# =========================
# METRICS
# =========================
def compute_metrics(y_true, probs):

    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average='macro')
    f1_weighted = f1_score(y_true, preds, average='weighted')

    try:
        auc = roc_auc_score(
            label_binarize(y_true, classes=np.unique(y_true)),
            probs,
            average='macro',
            multi_class='ovr'
        )
    except:
        auc = 0.0

    return acc, f1_macro, f1_weighted, auc


# =========================
# VALIDATION μ
# =========================
probs_list = []

probs_list.append(
    RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    .fit(X_tr, y_tr)
    .predict_proba(X_val)
)

probs_list.append(
    train_nn(FFNN(X_tr.shape[1], len(np.unique(y_tr))), X_tr, y_tr, X_val)
)

probs_list.append(
    train_nn(MLP(X_tr.shape[1], len(np.unique(y_tr))), X_tr, y_tr, X_val)
)

mu = build_mu(probs_list, y_val)


# =========================
# TEST
# =========================
test_probs = []

test_probs.append(
    RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    .fit(z_train_np, y_train_np)
    .predict_proba(z_test_np)
)

test_probs.append(
    train_nn(FFNN(z_train_np.shape[1], len(np.unique(y_train_np))),
             z_train_np, y_train_np, z_test_np)
)

test_probs.append(
    train_nn(MLP(z_train_np.shape[1], len(np.unique(y_train_np))),
             z_train_np, y_train_np, z_test_np)
)

test_probs = np.array(test_probs)


# =========================
# FUSION (ADAPTIVE ADDED)
# =========================
choq_probs = choquet(test_probs, mu)
sug_probs  = sugeno(test_probs, mu)

conf_choq = np.mean(np.max(choq_probs, axis=1))
conf_sug  = np.mean(np.max(sug_probs, axis=1))
alpha = conf_choq / (conf_choq + conf_sug + 1e-8)

hybrid_probs = alpha * choq_probs + (1 - alpha) * sug_probs


# =========================
# METRICS + PRINT
# =========================
acc_sug, f1m_sug, f1w_sug, auc_sug = compute_metrics(y_test_np, sug_probs)
acc_chq, f1m_chq, f1w_chq, auc_chq = compute_metrics(y_test_np, choq_probs)
acc_hyb, f1m_hyb, f1w_hyb, auc_hyb = compute_metrics(y_test_np, hybrid_probs)

all_acc_sug.append(acc_sug)
all_f1m_sug.append(f1m_sug)
all_f1w_sug.append(f1w_sug)
all_auc_sug.append(auc_sug)

all_acc_chq.append(acc_chq)
all_f1m_chq.append(f1m_chq)
all_f1w_chq.append(f1w_chq)
all_auc_chq.append(auc_chq)

all_acc_hyb.append(acc_hyb)
all_f1m_hyb.append(f1m_hyb)
all_f1w_hyb.append(f1w_hyb)
all_auc_hyb.append(auc_hyb)

print(f"Fold {fold+1} → SUGENO  Acc:{acc_sug:.4f} | F1_macro:{f1m_sug:.4f} | F1_weighted:{f1w_sug:.4f} | AUC:{auc_sug:.4f}")
print(f"Fold {fold+1} → CHOQUET Acc:{acc_chq:.4f} | F1_macro:{f1m_chq:.4f} | F1_weighted:{f1w_chq:.4f} | AUC:{auc_chq:.4f}")
print(f"Fold {fold+1} → HYBRID  Acc:{acc_hyb:.4f} | F1_macro:{f1m_hyb:.4f} | F1_weighted:{f1w_hyb:.4f} | AUC:{auc_hyb:.4f}")
print("\n ===== FINAL RESULTS (10-FOLD CV) =====")

# =========================
# SUGENO
# =========================
print("\n--- SUGENO ---")
print(f"Accuracy     : {np.mean(all_acc_sug):.4f} ± {np.std(all_acc_sug):.4f}")
print(f"F1 Macro     : {np.mean(all_f1m_sug):.4f} ± {np.std(all_f1m_sug):.4f}")
print(f"F1 Weighted  : {np.mean(all_f1w_sug):.4f} ± {np.std(all_f1w_sug):.4f}")
print(f"AUC          : {np.mean(all_auc_sug):.4f} ± {np.std(all_auc_sug):.4f}")

# =========================
# CHOQUET
# =========================
print("\n--- CHOQUET ---")
print(f"Accuracy     : {np.mean(all_acc_chq):.4f} ± {np.std(all_acc_chq):.4f}")
print(f"F1 Macro     : {np.mean(all_f1m_chq):.4f} ± {np.std(all_f1m_chq):.4f}")
print(f"F1 Weighted  : {np.mean(all_f1w_chq):.4f} ± {np.std(all_f1w_chq):.4f}")
print(f"AUC          : {np.mean(all_auc_chq):.4f} ± {np.std(all_auc_chq):.4f}")

# =========================
# HYBRID
# =========================
print("\n--- HYBRID (CHOQUET + SUGENO) ---")
print(f"Accuracy     : {np.mean(all_acc_hyb):.4f} ± {np.std(all_acc_hyb):.4f}")
print(f"F1 Macro     : {np.mean(all_f1m_hyb):.4f} ± {np.std(all_f1m_hyb):.4f}")
print(f"F1 Weighted  : {np.mean(all_f1w_hyb):.4f} ± {np.std(all_f1w_hyb):.4f}")
print(f"AUC          : {np.mean(all_auc_hyb):.4f} ± {np.std(all_auc_hyb):.4f}")
