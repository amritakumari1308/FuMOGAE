## Hyperparameter Configuration
All hyperparameter settings used in theFuMOGAE experiments are stored in:
Graph construction:
distance metric = Euclidean
kernel = Gaussian
sigma = mean pairwise distance
kNN = 10
symmetrization = max(W, Wᵀ)
self-loops = removed (added during normalization)
sparsification = top 10% edges (90th percentile)

Multi-view graph autoencoder:
encoder = input → 512 → 128
latent dimension = 128
projection head = 128 → 64
activation = ReLU
graph normalization = D⁻¹/² A D⁻¹/²

Training:
optimizer = Adam
learning rate = 0.0005
weight decay = 1e-4
epochs = 80
early stopping patience = 10
gradient clipping = 5
contrastive temperature = 0.5

Classifier :

Random Forest:
n_estimators = 300
max_depth = 12
min_samples_split = 5
min_samples_leaf = 2

FFNN:
architecture = 128 → 64 → 5
dropout = 0.3, 0.2
learning rate = 0.001
epochs = 100

MLP:
architecture = 64 → 5

Fusion:
methods = Choquet, Sugeno
final output = 0.5 × Choquet + 0.5 × Sugen
