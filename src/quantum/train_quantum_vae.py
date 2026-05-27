import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class QuantumLayer(nn.Module):
    """Quantum-inspired layer with residual connections (fixed vanishing gradient issue)"""

    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Learnable parameters
        self.rotation_weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.entanglement_weights = nn.Parameter(torch.randn(n_layers, n_qubits, n_qubits) * 0.1)

        # Layer normalization to prevent exploding/vanishing values
        self.layer_norm = nn.LayerNorm(n_qubits)

        # Hadamard matrix (fixed)
        self.register_buffer('hadamard', self._hadamard_matrix(n_qubits))

    def _hadamard_matrix(self, n):
        H = torch.ones(n, n) / np.sqrt(n)
        for i in range(1, n):
            for j in range(1, n):
                H[i, j] = ((-1) ** bin(i & j).count('1')) / np.sqrt(n)
        return H

    def forward(self, x):
        batch_size = x.size(0)
        state = x.view(batch_size, self.n_qubits)

        for layer in range(self.n_layers):
            # Store residual
            residual = state

            # Apply Hadamard transform
            state = torch.matmul(state, self.hadamard)

            # Apply rotation (phase shift)
            phase = torch.sin(self.rotation_weights[layer] * np.pi)
            state = state * (1 + 0.5 * phase.unsqueeze(0))

            # Apply entanglement
            entanglement = torch.tanh(self.entanglement_weights[layer])
            entangled = torch.matmul(state, entanglement)

            # Combine with residual connection (CRITICAL FIX)
            state = 0.7 * state + 0.3 * entangled

            # Apply layer normalization instead of tanh
            state = self.layer_norm(state)

        return state


class QuantumEncoder(nn.Module):
    """Hybrid classical-quantum encoder"""

    def __init__(self, input_channels=2, latent_dim=16, n_qubits=8):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_to_quantum = nn.Linear(256, n_qubits)
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=2)  # Reduced layers
        self.fc_mu = nn.Linear(n_qubits, latent_dim)
        self.fc_logvar = nn.Linear(n_qubits, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        quantum_input = torch.tanh(self.fc_to_quantum(x))  # Keep bounded input
        quantum_state = self.quantum_layer(quantum_input)
        mu = self.fc_mu(quantum_state)
        logvar = self.fc_logvar(quantum_state)
        return mu, logvar


class QuantumDecoder(nn.Module):
    """Decoder for 64x64 output"""

    def __init__(self, latent_dim=16, output_channels=2):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.deconv_layers(x)


class QuantumVAE(nn.Module):
    """Complete Quantum-Inspired VAE"""

    def __init__(self, input_channels=2, latent_dim=16, n_qubits=8):
        super().__init__()
        self.encoder = QuantumEncoder(input_channels, latent_dim, n_qubits)
        self.decoder = QuantumDecoder(latent_dim, input_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)


def train_quantum_vae():
    """Main training function"""
    print("=" * 60)
    print("TRAINING QUANTUM-INSPIRED VAE FOR FLUID DYNAMICS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    # Parameters
    GRID_SIZE = 64
    LATENT_DIM = 16
    N_QUBITS = 8
    BATCH_SIZE = 32
    EPOCHS = 150
    BETA = 0.5  # KL weight (β-VAE)

    # Generate or load data
    print("\nLoading/generating fluid data...")
    data_path = 'quantum_fluid_data/quantum_fluid_data.npz'

    if os.path.exists(data_path):
        data = np.load(data_path)['data']
        print(f"Loaded existing data, shape: {data.shape}")
    else:
        from generate_data import generate_fluid_data
        _, data = generate_fluid_data(n_samples=2000, grid_size=GRID_SIZE)

    X_tensor = torch.FloatTensor(data)
    print(f"Data range: [{X_tensor.min():.3f}, {X_tensor.max():.3f}]")

    # Split data
    total = len(X_tensor)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_dataset = TensorDataset(X_tensor[:train_size])
    val_dataset = TensorDataset(X_tensor[train_size:train_size + val_size])
    test_dataset = TensorDataset(X_tensor[train_size + val_size:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}, Test: {total - train_size - val_size}")

    # Initialize model
    model = QuantumVAE(input_channels=2, latent_dim=LATENT_DIM, n_qubits=N_QUBITS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []

    print("\nTraining...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0

        for data in train_loader:
            data = data[0].to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)

            # Reconstruction loss (MSE)
            recon_loss = nn.MSELoss(reduction='sum')(recon, data)

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss with beta weighting
            loss = recon_loss + BETA * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data[0].to(device)
                recon, mu, logvar = model(data)
                recon_loss = nn.MSELoss(reduction='sum')(recon, data)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (recon_loss + BETA * kl_loss).item()

        # Average losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_train_recon = train_recon / len(train_loader.dataset)
        avg_train_kl = train_kl / len(train_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        recon_losses.append(avg_train_recon)
        kl_losses.append(avg_train_kl)

        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'quantum_vae_model_{timestamp}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_dim': LATENT_DIM,
                'n_qubits': N_QUBITS,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }, model_path)
            print(f"  ✓ Saved best model to {model_path} (val_loss: {best_val_loss:.4f})")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS}: Train Loss: {avg_train_loss:.4f} | Recon: {avg_train_recon:.4f} | KL: {avg_train_kl:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recon_losses, label='Reconstruction Loss', linewidth=2)
    axes[1].plot(kl_losses, label='KL Divergence', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantum_training_curves.png', dpi=150)
    plt.show()

    print(f"\n{'=' * 50}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 50}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final reconstruction loss: {recon_losses[-1]:.4f}")
    print(f"Final KL divergence: {kl_losses[-1]:.4f}")
    print(f"Model saved as: quantum_vae_model_*.pth")
    print(f"Training curves saved to: quantum_training_curves.png")

    return model_path


if __name__ == "__main__":
    train_quantum_vae()