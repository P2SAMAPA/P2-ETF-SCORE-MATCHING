import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())  # Swish activation
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_score_network(data, input_dim, hidden_dim=128, n_layers=3, noise_scale=0.1,
                        lr=1e-3, batch_size=64, epochs=100):
    """
    Train a score network using denoising score matching.
    data: numpy array of shape (n_samples, input_dim)
    Returns trained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScoreNetwork(input_dim, hidden_dim, n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.tensor(data, dtype=torch.float32).to(device)
    n = len(dataset)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            x = dataset[batch_idx]
            # Denoising score matching: add Gaussian noise
            noise = torch.randn_like(x) * noise_scale
            x_noisy = x + noise
            score = model(x_noisy)
            # Loss = ||score + noise / noise_scale^2||^2
            target = -noise / (noise_scale**2)
            loss = torch.mean((score - target)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.6f}")
    return model

def langevin_sampling(model, initial_x, steps=10, step_size=0.1):
    """
    Run Langevin dynamics to sample from the distribution.
    Returns the final sample.
    """
    device = next(model.parameters()).device
    x = torch.tensor(initial_x, dtype=torch.float32).to(device)
    for _ in range(steps):
        score = model(x)
        x = x + step_size * score + np.sqrt(2 * step_size) * torch.randn_like(x)
    return x.cpu().detach().numpy()

def compute_score(model, x):
    """Compute score (gradient of log density) at point x."""
    device = next(model.parameters()).device
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        score = model(x_t).cpu().numpy()
    return score
