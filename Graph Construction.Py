## -*- coding: -*-
## ChoqSugFuzGAE , GAE is applied to six omic modalities (CNA, GSE, DNA-Methylation, miRNA, Mutation, and
## Co-expression) after hybrid Boruta + mRMR feature selection. Latent embeddings extracted from these
## six GAEs are concatenated with the Clinical (CLN) features to form a unified representation.
## The final breast cancer subtype classification is performed using a hybrid Choquet–Sugeno fuzzy integral.
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
import numpy as np

def build_knn_gaussian_graph(X, k=10):
    X = np.asarray(X, dtype=np.float32)

    # Step 1: distance
    dist = euclidean_distances(X)
    dist = dist / (dist.max() + 1e-8)

    # Step 2: Gaussian similarity
    sigma = np.mean(dist)
    W = np.exp(-dist**2 / (2 * sigma**2))

    # Step 3: KNN mask
    knn = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    knn = knn.toarray()

    # Step 4: apply mask
    W = W * knn

    # 🔥 FIX (VERY IMPORTANT)
    W = np.maximum(W, W.T)   # make symmetric

    # remove self-loops
    np.fill_diagonal(W, 0)
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
W_dict = {}

for name, X in modalities.items():
    print(f"Building similarity knn gaussian for {name}")
    W_dict[name] = build_knn_gaussian_graph(X)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(W, title="Graph Visualization", num_nodes=200):
    """
    W: adjacency matrix
    num_nodes: visualize subset (important for large graphs)
    """

    # 🔥 Step 1: reduce size (for clarity)
    W_small = W[:num_nodes, :num_nodes]

    # 🔥 Step 2: create graph
    G = nx.from_numpy_array(W_small)

    # 🔥 Step 3: remove very weak edges (optional but important)
    threshold = np.percentile(W_small, 90)  # keep strong edges
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    G.remove_edges_from(edges_to_remove)

    # 🔥 Step 4: layout
    pos = nx.spring_layout(G, seed=42)

    # 🔥 Step 5: plot
    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos,
        node_size=30,
        edge_color="gray",
        alpha=0.7,
        with_labels=False
    )

    plt.title(title)
    plt.show()
visualize_graph(W_dict["clinical"], "Clinical Graph")
visualize_graph(W_dict["cna"], "CNA Graph")
visualize_graph(W_dict["gse"], "GSE Graph")
visualize_graph(W_dict["dna"], "DNA Graph")
visualize_graph(W_dict["mirna"], "miRNA Graph")
visualize_graph(W_dict["mutation"], "Mutation Graph")
visualize_graph(W_dict["coexp"], "Coexpression Graph")
# ============================================================
# 🔥 TRAIN-TEST SPLIT (CORRECT POSITION)
# ============================================================

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

y = labels  # 🔥 use your label variable

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(sss.split(y, y))

train_mask = np.zeros(len(y), dtype=bool)
test_mask = np.zeros(len(y), dtype=bool)

train_mask[train_idx] = True
test_mask[test_idx] = True

print("\n===== DATA SPLIT =====")
print("Train samples:", train_mask.sum())
print("Test samples :", test_mask.sum())

print("\n===== CLASS DISTRIBUTION =====")
print("Train:", np.bincount(y[train_mask]))
print("Test :", np.bincount(y[test_mask]))
W_train = {}
W_test = {}

for name, W in W_dict.items():
    W_train[name] = W[np.ix_(train_mask, train_mask)]
    W_test[name]  = W[np.ix_(test_mask, test_mask)]
# ============================================================
# 🔥 STEP 2: SPARSIFY GRAPH (VERY IMPORTANT)
# ============================================================

for k in W_train:
    thresh = np.percentile(W_train[k], 90)
    W_train[k] = (W_train[k] > thresh).astype(float)

for k in W_test:
    thresh = np.percentile(W_test[k], 90)
    W_test[k] = (W_test[k] > thresh).astype(float)
from torch_geometric.utils import dense_to_sparse
import torch

edge_index_train = {}
edge_index_test = {}

# 🔥 TRAIN GRAPH
for key in W_train:
    A = torch.tensor(W_train[key], dtype=torch.float32)
    edge_index, _ = dense_to_sparse(A)
    edge_index_train[key] = edge_index

# 🔥 TEST GRAPH
for key in W_test:
    A = torch.tensor(W_test[key], dtype=torch.float32)
    edge_index, _ = dense_to_sparse(A)
    edge_index_test[key] = edge_index
modalities_train = {}
modalities_test = {}

for name, X in modalities.items():
    modalities_train[name] = X[train_mask]
    modalities_test[name]  = X[test_mask]
import torch

X_train_dict = {
    name: torch.tensor(data, dtype=torch.float32)
    for name, data in modalities_train.items()
}

X_test_dict = {
    name: torch.tensor(data, dtype=torch.float32)
    for name, data in modalities_test.items()
}
print("\n📊 TRAIN SHAPES:")
for k, v in X_train_dict.items():
    print(f"{k}: {v.shape}")

print("\n📊 TEST SHAPES:")
for k, v in X_test_dict.items():
    print(f"{k}: {v.shape}")
import torch
import numpy as np

def comp(g):
    # Convert to numpy (if tensor)
    if isinstance(g, torch.Tensor):
        g = g.cpu().numpy()

    # Add self-loop
    g = g + np.eye(g.shape[0])

    # Degree matrix
    d = np.diag(g.sum(axis=1))

    # D^(-1/2)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

    # Convert to torch
    d_inv_sqrt = torch.tensor(d_inv_sqrt, dtype=torch.float32)
    g = torch.tensor(g, dtype=torch.float32)

    # Normalization
    s = d_inv_sqrt @ g @ d_inv_sqrt

    return s
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, 512)
        self.enc2 = nn.Linear(512, feature_dim)

        self.weight = nn.Parameter(torch.Tensor(feature_dim, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, s):
        # Graph propagation
        x = torch.matmul(s, x)
        x = F.relu(self.enc1(x))   # 🔥 changed from tanh → ReLU

        x = torch.matmul(s, x)
        z = F.relu(self.enc2(x))   # 🔥 changed from tanh → ReLU

        # Graph reconstruction
        a = torch.sigmoid(z @ self.weight @ z.T)

        return z, a


# =========================
# Decoder
# =========================
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.dec1 = nn.Linear(feature_dim, 512)
        self.dec2 = nn.Linear(512, input_dim)

    def forward(self, z, s):
        x = torch.matmul(s, z)
        x = F.relu(self.dec1(x))   # 🔥 changed to ReLU

        x = torch.matmul(s, x)
        x_recon = self.dec2(x)     # 🔥 NO activation (important)

        return x_recon


# =========================
# Multi-View Network
# =========================
class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super().__init__()

        # Multi-view encoders & decoders
        self.encoders = nn.ModuleList([
            Encoder(input_size[v], feature_dim).to(device)
            for v in range(view)
        ])

        self.decoders = nn.ModuleList([
            Decoder(input_size[v], feature_dim).to(device)
            for v in range(view)
        ])

        # Feature head (for contrastive learning)
        self.feature_head = nn.Linear(feature_dim, high_feature_dim)

        # 🔥 FIXED CLASSIFIER (NO SOFTMAX)
        self.classifier = nn.Linear(feature_dim, class_num)

        self.view = view

    def forward(self, xs, S_dict):
        hs, qs, xrs, zs, ars = [], [], [], [], []

        for v in range(self.view):
            # Encode
            z, a = self.encoders[v](xs[v], S_dict[v])

            # High-level features (contrastive)
            h = F.normalize(self.feature_head(z), dim=1)

            # Classification logits (NO softmax)
            q = self.classifier(z)

            # Decode
            xr = self.decoders[v](z, S_dict[v])

            # Store
            hs.append(h)
            qs.append(q)
            xrs.append(xr)
            zs.append(z)
            ars.append(a)

        return hs, qs, xrs, zs, ars
import torch

# 🔥 ADD THIS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = list(X_train_dict.keys())

# TRAIN
xs_train = [X_train_dict[name].to(device) for name in names]
S_train  = [comp(W_train[name]).to(device) for name in names]

# TEST
xs_test = [X_test_dict[name].to(device) for name in names]
S_test  = [comp(W_test[name]).to(device) for name in names]

# LABELS
y_train = torch.tensor(labels[train_mask], dtype=torch.long).to(device)
y_test  = torch.tensor(labels[test_mask], dtype=torch.long).to(device)
print("\n✅ TRAIN CHECK")
for i, name in enumerate(names):
    print(f"{name}: X {xs_train[i].shape} | S {S_train[i].shape}")

print("\n✅ TEST CHECK")
for i, name in enumerate(names):
    print(f"{name}: X {xs_test[i].shape} | S {S_test[i].shape}")
import torch.nn.functional as F
import torch

# ======================================
# 1. Reconstruction Loss
# ======================================
def reconstruction_loss(xs, xrs):
    loss = 0
    for x, xr in zip(xs, xrs):
        loss += F.mse_loss(xr, x)
    return loss


# ======================================
# 2. Contrastive Loss (FIXED)
# ======================================
def contrastive_loss(hs, temperature=0.5):
    loss = 0
    count = 0

    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):

            z_i = F.normalize(hs[i], dim=1)
            z_j = F.normalize(hs[j], dim=1)

            N = z_i.size(0)

            representations = torch.cat([z_i, z_j], dim=0)  # (2N, d)

            similarity_matrix = torch.matmul(representations, representations.T) / temperature

            # remove self similarity
            mask = torch.eye(2*N, dtype=torch.bool).to(z_i.device)
            similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

            # 🔥 CORRECT POSITIVE PAIRS
            labels = torch.cat([
                torch.arange(N, 2*N),
                torch.arange(0, N)
            ]).to(z_i.device)

            loss_i = F.cross_entropy(similarity_matrix, labels)

            loss += loss_i
            count += 1

    return loss / count


# ======================================
# 3. Classification Loss
# ======================================
def classification_loss(qs, labels):
    loss = 0
    for q in qs:
        loss += F.cross_entropy(q, labels)
    return loss


# ======================================
# 4. Graph Reconstruction Loss
# ======================================
def graph_loss(ars, W_dict, names, device):
    loss = 0

    for i, name in enumerate(names):
        g = torch.tensor(W_dict[name], dtype=torch.float32).to(device)

        # normalize target graph
        g = g / (g.max() + 1e-8)

        loss += F.mse_loss(ars[i], g)

    return loss
# ============================================================
# 🔥 TRAIN + TEST (FINAL CORRECT VERSION)
# ============================================================

import torch
import torch.optim as optim

# ============================================================
# ✅ ENSURE LABELS ARE CORRECT (DO THIS ONCE)
# ============================================================
y_train = torch.tensor(labels[train_mask], dtype=torch.long).to(device)
y_test  = torch.tensor(labels[test_mask], dtype=torch.long).to(device)

# ============================================================
# MODEL INIT
# ============================================================
model = Network(
    view=len(xs_train),
    input_size=[x.shape[1] for x in xs_train],
    feature_dim=128,
    high_feature_dim=64,
    class_num=5,
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

names = list(X_train_dict.keys())

# ============================================================
# 🔥 EARLY STOPPING SETUP
# ============================================================
best_loss = float("inf")
patience = 10
counter = 0

# ============================================================
# 🔍 DEBUG CHECK (IMPORTANT)
# ============================================================
print("X size:", xs_train[0].shape[0])
print("y size:", y_train.shape[0])

# ============================================================
# TRAINING
# ============================================================
for epoch in range(80):

    model.train()
    optimizer.zero_grad()

    hs, qs, xrs, zs, ars = model(xs_train, S_train)

    recon_loss = reconstruction_loss(xs_train, xrs)
    cont_loss = contrastive_loss(hs)
    cls_loss = classification_loss(qs, y_train)   # ✅ FIXED
    graph_loss_val = graph_loss(ars, W_train, names, device)

    loss = recon_loss + 0.1 * cont_loss + cls_loss + graph_loss_val

    loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

    optimizer.step()

    # ========================================================
    # EARLY STOPPING
    # ========================================================
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1

    if counter > patience:
        print(f"⛔ Early stopping at epoch {epoch}")
        break

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ============================================================
# TESTING
# ============================================================
model.eval()

with torch.no_grad():
    hs, qs, xrs, zs, ars = model(xs_test, S_test)

    q_avg = sum(qs) / len(qs)
    preds = torch.argmax(q_avg, dim=1)

    acc = (preds == y_test).float().mean()

print("✅ Test Accuracy:", acc.item())
model.eval()

with torch.no_grad():
    hs_te, qs_te, xrs_te, zs_te, ars_te = model(xs_test, S_test)
for i, name in enumerate(names):
    print(f"\n{name} TEST embedding shape:", zs_te[i].shape)
model.eval()

with torch.no_grad():
    hs_tr, qs_tr, xrs_tr, zs_tr, ars_tr = model(xs_train, S_train)
# ============================================================
# 🔥 TRAIN EMBEDDING (CORRECTED)
# ============================================================

z_concat_train = torch.cat(zs_tr, dim=1)
z_concat_train_np = z_concat_train.cpu().numpy()

# 🔥 STEP 1: ADD SMALL NOISE (ANTI-OVERFITTING)
z_concat_train_np = z_concat_train_np + np.random.normal(0, 0.01, z_concat_train_np.shape)

# 🔥 STEP 2: STANDARDIZATION (VERY IMPORTANT)
z_concat_train_np = (z_concat_train_np - z_concat_train_np.mean(axis=0)) / (z_concat_train_np.std(axis=0) + 1e-8)

# 🔥 STEP 3: L2 NORMALIZATION (STABILIZES MODELS)
z_concat_train_np = z_concat_train_np / (np.linalg.norm(z_concat_train_np, axis=1, keepdims=True) + 1e-8)

print("Train concat shape:", z_concat_train_np.shape)


# ============================================================
# 🔥 TEST EMBEDDING (MATCH TRAIN DISTRIBUTION)
# ============================================================

z_concat_test = torch.cat(zs_te, dim=1)
z_concat_test_np = z_concat_test.cpu().numpy()

# ⚠️ IMPORTANT: USE TRAIN STATS (NOT TEST!)
mean = z_concat_train_np.mean(axis=0)
std = z_concat_train_np.std(axis=0) + 1e-8

z_concat_test_np = (z_concat_test_np - mean) / std
z_concat_test_np = z_concat_test_np / (np.linalg.norm(z_concat_test_np, axis=1, keepdims=True) + 1e-8)

print("Test concat shape:", z_concat_test_np.shape)
    return W
