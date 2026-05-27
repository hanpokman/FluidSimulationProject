import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class QuantumLayer(nn.Module):
    """Quantum-inspired layer"""

    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.rotation_weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.entanglement_weights = nn.Parameter(torch.randn(n_layers, n_qubits, n_qubits) * 0.1)

        # Hadamard matrix
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
            state = torch.matmul(state, self.hadamard)
            phase = torch.sin(self.rotation_weights[layer] * np.pi)
            state = state * (1 + 0.5 * phase.unsqueeze(0))
            entanglement = torch.tanh(self.entanglement_weights[layer])
            entangled = torch.matmul(state, entanglement)
            state = 0.7 * state + 0.3 * entangled
            state = torch.tanh(state)

        return state


class QuantumEncoder(nn.Module):
    """Hybrid classical-quantum encoder"""

    def __init__(self, input_channels=2, latent_dim=16, n_qubits=8, grid_size=32):
        super().__init__()

        # Calculate conv output size
        self.grid_size = grid_size

        # Classical feature extraction for 32x32 input
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to 1x1
        )

        # Project to quantum input
        self.fc_to_quantum = nn.Linear(256, n_qubits)

        # Quantum-inspired layer
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=3)

        # Project to latent space
        self.fc_mu = nn.Linear(n_qubits, latent_dim)
        self.fc_logvar = nn.Linear(n_qubits, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        quantum_input = self.fc_to_quantum(x)
        quantum_state = self.quantum_layer(quantum_input)
        mu = self.fc_mu(quantum_state)
        logvar = self.fc_logvar(quantum_state)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder for 32x32 output"""

    def __init__(self, latent_dim=16, output_channels=2):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)  # 2x2 feature map

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2 -> 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 4 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # 8 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 2, 2)
        return self.deconv_layers(x)


class QuantumVAE(nn.Module):
    """Complete Quantum-Inspired VAE"""

    def __init__(self, input_channels=2, latent_dim=16, n_qubits=8):
        super().__init__()
        self.encoder = QuantumEncoder(input_channels, latent_dim, n_qubits)
        self.decoder = Decoder(latent_dim, input_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def generate_fluid_data(n_samples=1000, grid_size=32):
    """Generate synthetic fluid flow data"""
    u_fields = []
    v_fields = []

    for i in range(n_samples):
        x = np.linspace(-2, 4, grid_size)
        y = np.linspace(-1.5, 1.5, grid_size)
        X, Y = np.meshgrid(x, y)

        flow_type = np.random.choice(['vortex_pair', 'shear_vortex', 'cylinder_wake'])

        if flow_type == 'vortex_pair':
            strength = np.random.uniform(0.8, 1.5)
            offset = np.random.uniform(-0.5, 0.5)

            r1 = np.sqrt((X + 1)**2 + (Y - offset)**2) + 0.1
            u1 = -strength * (Y - offset) / r1**2
            v1 = strength * (X + 1) / r1**2

            r2 = np.sqrt((X - 1)**2 + (Y + offset)**2) + 0.1
            u2 = strength * (Y + offset) / r2**2
            v2 = -strength * (X - 1) / r2**2

            u = u1 + u2
            v = v1 + v2

        elif flow_type == 'shear_vortex':
            shear_rate = np.random.uniform(0.01, 0.08)
            u = 1.0 + shear_rate * Y
            v = np.zeros_like(X)

            vortex_x = np.random.uniform(0, 2)
            vortex_y = np.random.uniform(-0.8, 0.8)
            strength = np.random.uniform(0.3, 0.8)

            r_vort = np.sqrt((X - vortex_x)**2 + (Y - vortex_y)**2) + 0.1
            u += -strength * (Y - vortex_y) / r_vort**2
            v += strength * (X - vortex_x) / r_vort**2

        else:  # cylinder_wake
            U_inf = 1.0
            radius = 0.3

            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            r_safe = np.maximum(r, radius)

            ur = U_inf * (1 - radius**2 / r_safe**2) * np.cos(theta)
            utheta = -U_inf * (1 + radius**2 / r_safe**2) * np.sin(theta)

            u = ur * np.cos(theta) - utheta * np.sin(theta)
            v = ur * np.sin(theta) + utheta * np.cos(theta)

            # Add vortex shedding
            strength = np.random.uniform(0.2, 0.5)
            for x_pos, y_sign in [(1.2, 1), (2.0, -1), (2.8, 1)]:
                y_pos = 0.4 * y_sign
                r_vort = np.sqrt((X - x_pos)**2 + (Y - y_pos)**2) + 0.1
                u += -strength * (Y - y_pos) / r_vort**2
                v += strength * (X - x_pos) / r_vort**2

            mask = r < radius
            u[mask] = 0
            v[mask] = 0

        u_fields.append(u)
        v_fields.append(v)

    u_fields = np.array(u_fields)
    v_fields = np.array(v_fields)

    # Normalize
    mean_u, std_u = u_fields.mean(), u_fields.std()
    mean_v, std_v = v_fields.mean(), v_fields.std()

    u_norm = (u_fields - mean_u) / (std_u + 1e-8)
    v_norm = (v_fields - mean_v) / (std_v + 1e-8)

    return np.stack([u_norm, v_norm], axis=1)


def main():
    print("=" * 60)
    print("QUANTUM-INSPIRED VAE FOR FLUID DYNAMICS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    GRID_SIZE = 32
    LATENT_DIM = 16
    N_QUBITS = 8
    BATCH_SIZE = 32
    EPOCHS = 50

    # Generate data
    print("\nGenerating fluid data...")
    X = generate_fluid_data(n_samples=1500, grid_size=GRID_SIZE)
    X_tensor = torch.FloatTensor(X)
    print(f"Data shape: {X_tensor.shape}")

    # Split data
    total = len(X_tensor)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_dataset = Subset(X_tensor, range(train_size))
    val_dataset = Subset(X_tensor, range(train_size, train_size + val_size))
    test_dataset = Subset(X_tensor, range(train_size + val_size, total))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}, Test: {total - train_size - val_size}")

    # Initialize model
    model = QuantumVAE(input_channels=2, latent_dim=LATENT_DIM, n_qubits=N_QUBITS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    train_losses = []
    val_losses = []

    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)

            # VAE loss
            recon_loss = nn.MSELoss(reduction='sum')(recon, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                recon_loss = nn.MSELoss(reduction='sum')(recon, data)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (recon_loss + kl_loss).item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Quantum-Inspired VAE Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

    # Test reconstruction
    model.eval()
    test_data = torch.stack([X_tensor[i] for i in range(min(8, len(test_dataset)))])

    with torch.no_grad():
        test_data = test_data.to(device)
        recon, _, _ = model(test_data)

    # Visualize results
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))

    for i in range(8):
        orig_mag = np.sqrt(test_data[i, 0].cpu().numpy()**2 + test_data[i, 1].cpu().numpy()**2)
        recon_mag = np.sqrt(recon[i, 0].cpu().numpy()**2 + recon[i, 1].cpu().numpy()**2)

        axes[0, i].imshow(orig_mag, cmap='viridis', aspect='auto')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_mag, cmap='viridis', aspect='auto')
        axes[1, i].axis('off')

    axes[0, 0].set_title('Original', fontsize=12)
    axes[1, 0].set_title('Quantum VAE\nReconstruction', fontsize=12)
    plt.suptitle('Fluid Flow Reconstruction using Quantum-Inspired VAE', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstruction_results.png', dpi=150)
    plt.show()

    # Calculate final metrics
    mse_total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon, _, _ = model(data)
            mse = nn.MSELoss()(recon, data)
            mse_total += mse.item()

    avg_mse = mse_total / len(test_loader)
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Test MSE: {avg_mse:.6f}")
    print(f"Compression: {GRID_SIZE*GRID_SIZE*2} → {LATENT_DIM} ({GRID_SIZE*GRID_SIZE*2/LATENT_DIM:.0f}x)")
    print(f"\nResults saved to:")
    print(f"  - training_curves.png")
    print(f"  - reconstruction_results.png")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()