# --- TRAIN/TEST BLOCK ---

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# CONFIG
PARQUET_FILE = "flattened_by_seaons.parquet"
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
N_DISHES = 15
BATCH_SIZE = 256
LATENT_DIM = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

# --- DATA PREPARATION ---
df = pd.read_parquet(PARQUET_FILE)
samples = []
for _, row in df.iterrows():
    for m in MONTHS:
        dish_vec = [int(ch) for ch in str(row[m]).zfill(N_DISHES)]
        samples.append(dish_vec)
X = np.array(samples, dtype=np.float32)

# Torch Dataset & Loader
tensor_in = torch.tensor(X)
dataset = TensorDataset(tensor_in, tensor_in)   # autoencoder: input = target
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- MODEL DEFINITION ---
class DishAutoencoder(nn.Module):
    def __init__(self, n_dishes, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_dishes, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_dishes),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# --- TRAINING LOOP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DishAutoencoder(N_DISHES, LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1}: loss={avg_loss:.5f}")

# --- SAVE TRAINED MODEL ---
torch.save(model.state_dict(), "autoencoder_recommender.pt")
print("Model saved as autoencoder_recommender.pt")
