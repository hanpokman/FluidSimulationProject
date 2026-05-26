import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.ndimage import zoom
from datetime import datetime

# Set device
#  The Corrected Logic
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


class Encoder(nn.Module):
    """Encoder network for VAE"""

    def __init__(self, input_channels=2, latent_dim=32, input_size=64):
        super(Encoder, self).__init__()

        # Convolutional encoder
        self.conv_layers = nn.Sequential(
            # Input: [batch, channels, 64, 64]
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [batch, 128, 8, 8]

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # [batch, 256, 4, 4]
        )

        # Calculate flattened size
        self.flattened_size = 256 * 4 * 4

        # Map to latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE"""

    def __init__(self, latent_dim=32, output_channels=2, output_size=64):
        super(Decoder, self).__init__()

        self.output_size = output_size

        # Map latent vector to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        # Transposed convolutional decoder
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [batch, 128, 8, 8]

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch, 64, 16, 16]

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch, 32, 32, 32]

            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # [batch, output_channels, 64, 64]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.deconv_layers(x)
        return x


class VAE(nn.Module):
    """Complete VAE model"""

    def __init__(self, input_channels=2, latent_dim=32, input_size=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim, input_size)
        self.decoder = Decoder(latent_dim, input_channels, input_size)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def load_fluid_data(data_path=None, grid_size=64, use_vorticity=False):
    """Load the generated fluid data"""
    if data_path is None:
        # Find the most recent data file
        data_dir = 'fluid_data'
        if not os.path.exists(data_dir):
            print(f"Error: {data_dir} directory not found. Please run generate_fluid_data.py first.")
            return None, None, None

        files = [f for f in os.listdir(data_dir) if f.startswith('fluid_data_') and f.endswith('.npz')]
        if not files:
            print("No data files found. Please run generate_fluid_data.py first.")
            return None, None, None
        latest_file = sorted(files)[-1]
        data_path = os.path.join(data_dir, latest_file)

    print(f"Loading data from {data_path}")
    data = np.load(data_path)

    if use_vorticity:
        # Use vorticity field instead of velocity
        X = data['vorticity']
        X = np.expand_dims(X, axis=1)  # Add channel dimension
        print("Using vorticity field")
    else:
        # Stack u and v velocity components
        X = np.stack([data['u'], data['v']], axis=1)
        print(f"Using velocity field (u, v)")

    # Normalize data to [-1, 1] range for Tanh output
    X_mean = X.mean()
    X_std = X.std()
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    # Resize if needed
    if X.shape[-1] != grid_size:
        print(f"Resizing data from {X.shape[-1]} to {grid_size}")
        resized_X = []
        for i in range(X.shape[0]):
            resized = zoom(X[i], (1, grid_size / X.shape[-2], grid_size / X.shape[-1]))
            resized_X.append(resized)
        X_normalized = np.array(resized_X)

    print(f"Data shape: {X_normalized.shape}")
    print(f"Data range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")

    return torch.FloatTensor(X_normalized), X_mean, X_std


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function (ELBO)"""
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(model, train_loader, val_loader, epochs=100, lr=1e-3, beta=1.0):
    """Train the VAE model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []

    print("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_data in train_loader:
            # DataLoader returns a tuple, we need to get the first element
            if isinstance(batch_data, (tuple, list)):
                data = batch_data[0]
            else:
                data = batch_data

            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        avg_kl_loss = epoch_kl_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (tuple, list)):
                    data = batch_data[0]
                else:
                    data = batch_data

                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss, _, _ = vae_loss(recon_batch, data, mu, logvar, beta)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}')

    return train_losses, val_losses, recon_losses, kl_losses


def visualize_results(model, test_data, X_mean, X_std, latent_dim, save_dir='vae_results'):
    """Visualize VAE reconstruction and latent space"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model.eval()
    with torch.no_grad():
        # Take first 8 samples
        test_samples = test_data[:min(8, len(test_data))].to(device)
        recon_data, _, _ = model(test_samples)

        # Denormalize
        test_data_np = test_samples.cpu().numpy() * X_std + X_mean
        recon_np = recon_data.cpu().numpy() * X_std + X_mean

        # Visualization 1: Original vs Reconstruction
        n_samples = min(4, len(test_samples))
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Original velocity magnitude
            orig_mag = np.sqrt(test_data_np[i, 0] ** 2 + test_data_np[i, 1] ** 2)
            im1 = axes[i, 0].imshow(orig_mag, cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'Original {i + 1} (Magnitude)')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0])

            # Reconstruction velocity magnitude
            recon_mag = np.sqrt(recon_np[i, 0] ** 2 + recon_np[i, 1] ** 2)
            im2 = axes[i, 1].imshow(recon_mag, cmap='viridis', aspect='auto')
            axes[i, 1].set_title(f'Reconstruction {i + 1}')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1])

            # Difference
            diff = np.abs(orig_mag - recon_mag)
            im3 = axes[i, 2].imshow(diff, cmap='hot', aspect='auto')
            axes[i, 2].set_title(f'Difference {i + 1}')
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2])

            # Velocity field quiver plot (subsample for clarity)
            step = 8
            y_coords, x_coords = np.mgrid[0:orig_mag.shape[0]:step, 0:orig_mag.shape[1]:step]
            axes[i, 3].quiver(x_coords, y_coords,
                              test_data_np[i, 0, ::step, ::step],
                              test_data_np[i, 1, ::step, ::step],
                              alpha=0.6)
            axes[i, 3].set_title(f'Velocity Field {i + 1}')
            axes[i, 3].axis('off')

        plt.suptitle('VAE Reconstruction Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'reconstruction_{timestamp}.png'), dpi=150)
        plt.show()

        # Visualization 2: Latent space interpolation
        print("\nGenerating latent space interpolations...")
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        alphas = np.linspace(0, 1, 7)
        fig, axes = plt.subplots(1, 7, figsize=(14, 3))

        for idx, alpha in enumerate(alphas):
            z_interp = (1 - alpha) * z1 + alpha * z2
            with torch.no_grad():
                interp_recon = model.decoder(z_interp).cpu().numpy()[0]
                interp_mag = np.sqrt(interp_recon[0] ** 2 + interp_recon[1] ** 2)

            axes[idx].imshow(interp_mag, cmap='viridis', aspect='auto')
            axes[idx].set_title(f'α={alpha:.2f}')
            axes[idx].axis('off')

        plt.suptitle('Latent Space Interpolation', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'interpolation_{timestamp}.png'), dpi=150)
        plt.show()


def create_flow_animation(model, test_data, X_mean, X_std, save_dir='vae_results'):
    """Create an animation comparing original and reconstructed flow"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model.eval()
    with torch.no_grad():
        # Take first 50 samples for animation
        n_frames = min(50, len(test_data))
        sequence = test_data[:n_frames].to(device)
        recon_sequence, _, _ = model(sequence)

        # Denormalize
        sequence_np = sequence.cpu().numpy() * X_std + X_mean
        recon_sequence_np = recon_sequence.cpu().numpy() * X_std + X_mean

    # Create animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def update_frame(frame):
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()

        # Original
        orig_mag = np.sqrt(sequence_np[frame, 0] ** 2 + sequence_np[frame, 1] ** 2)
        im1 = axes[0].imshow(orig_mag, cmap='viridis', aspect='auto', vmin=0, vmax=1.5)
        axes[0].set_title(f'Original (t={frame})')
        axes[0].axis('off')

        # Reconstruction
        recon_mag = np.sqrt(recon_sequence_np[frame, 0] ** 2 + recon_sequence_np[frame, 1] ** 2)
        im2 = axes[1].imshow(recon_mag, cmap='viridis', aspect='auto', vmin=0, vmax=1.5)
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')

        # Difference
        diff = np.abs(orig_mag - recon_mag)
        im3 = axes[2].imshow(diff, cmap='hot', aspect='auto')
        axes[2].set_title('Error')
        axes[2].axis('off')

        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        return im1, im2, im3

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=100, blit=False)
    anim.save(os.path.join(save_dir, f'flow_animation_{timestamp}.gif'), writer='pillow', fps=10)
    plt.close()
    print(f"Animation saved to {save_dir}/flow_animation_{timestamp}.gif")


def main():
    # Parameters
    LATENT_DIM = 16  # Bottleneck size (compressed representation)
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    GRID_SIZE = 64

    print(f"\n{'=' * 60}")
    print(f"FLUID DYNAMICS VAE TRAINING")
    print(f"{'=' * 60}")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Compression ratio: {GRID_SIZE * GRID_SIZE * 2} -> {LATENT_DIM}")
    print(f"Compression factor: {(GRID_SIZE * GRID_SIZE * 2) / LATENT_DIM:.1f}x")
    print(f"{'=' * 60}\n")

    # Load data
    data_tensor, X_mean, X_std = load_fluid_data(grid_size=GRID_SIZE, use_vorticity=False)
    if data_tensor is None:
        return

    # Split into train/val/test
    total_samples = len(data_tensor)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    # Create indices for splitting
    indices = list(range(total_samples))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subset datasets
    train_dataset = Subset(data_tensor, train_indices)
    val_dataset = Subset(data_tensor, val_indices)
    test_dataset = Subset(data_tensor, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    # Initialize model
    model = VAE(input_channels=2, latent_dim=LATENT_DIM, input_size=GRID_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Train model
    train_losses, val_losses, recon_losses, kl_losses = train_vae(
        model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, beta=1.0
    )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'vae_model_{timestamp}.pth')
    print(f"\nModel saved as vae_model_{timestamp}.pth")

    # Visualize results
    print("\nGenerating visualizations...")

    # Collect test data as tensor for visualization
    test_tensor_list = []
    for batch_data in test_loader:
        if isinstance(batch_data, (tuple, list)):
            test_tensor_list.append(batch_data[0])
        else:
            test_tensor_list.append(batch_data)
    test_tensor = torch.cat(test_tensor_list, dim=0)

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(recon_losses, label='Reconstruction Loss', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(kl_losses, label='KL Divergence', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('VAE Training Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'vae_training_curves_{timestamp}.png', dpi=150)
    plt.show()

    # Show reconstruction results
    visualize_results(model, test_tensor, X_mean, X_std, LATENT_DIM)

    # Create animation
    create_flow_animation(model, test_tensor, X_mean, X_std)

    print("\nTraining complete! All visualizations saved.")

    # Final evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, (tuple, list)):
                data = batch_data[0]
            else:
                data = batch_data
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, _, _ = vae_loss(recon, data, mu, logvar, beta=1.0)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"\nFinal test loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()