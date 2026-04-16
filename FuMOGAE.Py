## -*- coding: -*-
## FuMOGAE , GAE is applied to six omic modalities (CNA, GSE, DNA-Methylation, miRNA, Mutation, and
## Co-expression) after hybrid Boruta + mRMR feature selection. Latent embeddings extracted from these
## six GAEs are concatenated with the Clinical (CLN) features to form a unified representation.
## The final breast cancer subtype classification is performed using a hybrid Choquet–Sugeno fuzzy integral.
------------METABRIC DATASET(CLINICAL,CNA,GSE,MUTATION,COEXPRESSION) Modalities classification
# ============================================================
# 🔥 FINAL 10-FOLD CV (FULL METRICS + SEPARATE MODELS)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.combine import SMOTEENN

# ============================================================
# MODELS
# ============================================================
class FFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
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


# ============================================================
# TRAIN NN
# ============================================================
def train_nn(model, X_train, y_train, X_test, seed):
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    weights = torch.tensor([1.5,1.2,1.5,1.0,1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)

    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()

    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        return torch.softmax(model(X_test), dim=1).cpu().numpy()


# ============================================================
# FUZZY FUNCTIONS
# ============================================================
def build_mu(probs_list, y_true):
    n = len(probs_list)
    mu = {}

    accs = [accuracy_score(y_true, np.argmax(p,1)) for p in probs_list]
    weights = np.array(accs) / (np.sum(accs) + 1e-8)

    mu[()] = 0
    for i in range(n):
        mu[(i,)] = weights[i]

    for i in range(n):
        for j in range(i+1,n):
            comb = (probs_list[i]+probs_list[j])/2
            acc = accuracy_score(y_true, np.argmax(comb,1))
            mu[(i,j)] = max(mu[(i,)], mu[(j,)], acc)

    mu[tuple(range(n))] = 1
    return mu


def choquet(probs, mu):
    n,N,C = probs.shape
    out = np.zeros((N,C))
    for c in range(C):
        for i in range(N):
            f = probs[:,i,c]
            idx = np.argsort(f)
            f = f[idx]
            prev=0; total=0
            for k in range(n):
                subset = tuple(sorted(idx[k:]))
                total += (f[k]-prev)*mu.get(subset,0)
                prev = f[k]
            out[i,c]=total
    return out


def sugeno(probs, mu):
    n,N,C = probs.shape
    out = np.zeros((N,C))
    for c in range(C):
        for i in range(N):
            f = probs[:,i,c]
            idx = np.argsort(f)
            f = f[idx]
            vals=[]
            for k in range(n):
                subset = tuple(sorted(idx[k:]))
                vals.append(min(f[k], mu.get(subset,0)))
            out[i,c]=max(vals)
    return out


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, probs):

    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average='macro')
    auc = roc_auc_score(label_binarize(y_true, classes=np.unique(y_true)),
                        probs, average='macro', multi_class='ovr')

    return {"acc": acc, "f1_macro": f1_macro, "auc": auc}


# ============================================================
# PRINT FUNCTION (🔥 FIX)
# ============================================================
def print_metrics(m, name):
    print(f"{name} → Acc: {m['acc']:.4f}, F1_macro: {m['f1_macro']:.4f}, AUC: {m['auc']:.4f}")


# ============================================================
# 10-FOLD CV
# ============================================================
X = z_concat_train_np
y = labels[train_mask]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

rf_results, ffnn_results, mlp_results = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X,y)):

    print(f"\n🔥 Fold {fold+1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True)+1e-8)
    X_test  = X_test  / (np.linalg.norm(X_test, axis=1, keepdims=True)+1e-8)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = SMOTEENN().fit_resample(X_train, y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    rf_probs = []
    for seed in [0,1,2]:
        rf = RandomForestClassifier(n_estimators=600, random_state=seed)
        rf.fit(X_tr, y_tr)
        rf_probs.append(rf.predict_proba(X_val))

    ffnn_probs = [train_nn(FFNN(X_tr.shape[1],5), X_tr, y_tr, X_val, s) for s in [0,1,2]]
    mlp_probs  = [train_nn(MLP(X_tr.shape[1],5), X_tr, y_tr, X_val, s) for s in [0,1,2]]

    mu = build_mu(np.array(rf_probs + ffnn_probs + mlp_probs), y_val)

    def evaluate(probs):
        final = 0.6*choquet(probs, mu) + 0.4*sugeno(probs, mu)
        return compute_metrics(y_test, final)

    # RF
    rf_metrics = evaluate(np.array([
        RandomForestClassifier(n_estimators=600, random_state=s).fit(X_train,y_train).predict_proba(X_test)
        for s in [0,1,2]
    ]))

    # FFNN
    ffnn_metrics = evaluate(np.array([
        train_nn(FFNN(X_train.shape[1],5), X_train, y_train, X_test, s)
        for s in [0,1,2]
    ]))

    # MLP
    mlp_metrics = evaluate(np.array([
        train_nn(MLP(X_train.shape[1],5), X_train, y_train, X_test, s)
        for s in [0,1,2]
    ]))

    # 🔥 PRINT PER FOLD
    print_metrics(rf_metrics, "RF")
    print_metrics(ffnn_metrics, "FFNN")
    print_metrics(mlp_metrics, "MLP")

    rf_results.append(rf_metrics)
    ffnn_results.append(ffnn_metrics)
    mlp_results.append(mlp_metrics)


# ============================================================
# FINAL SUMMARY
# ============================================================
def summarize(results, name):
    print(f"\n🚀 {name} FINAL RESULTS")
    for key in results[0]:
        print(f"{key}: {np.mean([r[key] for r in results]):.4f}")

summarize(rf_results, "RF")
summarize(ffnn_results, "FFNN")
summarize(mlp_results, "MLP")
