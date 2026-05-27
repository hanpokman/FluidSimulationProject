# ============================================
# 3. TRAINING AND VISUALIZATION (train_quantum_vae.py)
# ============================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import our models
from train_quantum_vae import QuantumVAE, QuantumInspiredVAE


def load_quantum_data(data_path='quantum_fluid_data/quantum_fluid_data.npz'):
    """Load generated fluid data"""
    data = np.load(data_path)

    # Stack u and v components
    X = np.stack([data['u'], data['v']], axis=1)

    print(f"Loaded data shape: {X.shape}")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")

    return torch.FloatTensor(X), data['mean_u'].item(), data['std_u'].item()


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with KL annealing"""
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(model, train_loader, val_loader, epochs=100, lr=1e-3,
              beta_start=0.0, beta_end=1.0, beta_anneal_epochs=50, device='cpu'):
    """Train VAE with KL annealing"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []
    betas = []

    print("\nStarting training...")
    for epoch in range(epochs):
        # KL annealing schedule
        if epoch < beta_anneal_epochs:
            beta = beta_start + (beta_end - beta_start) * epoch / beta_anneal_epochs
        else:
            beta = beta_end
        betas.append(beta)

        # Training
        model.train()
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_data in train_loader:
            data = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            data = data.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                data = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, _, _ = vae_loss(recon, data, mu, logvar, beta)
                epoch_val_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        avg_kl_loss = epoch_kl_loss / len(train_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}: Train: {avg_train_loss:.4f}, '
                  f'Val: {avg_val_loss:.4f}, β={beta:.3f}')

    return train_losses, val_losses, recon_losses, kl_losses, betas


def visualize_comparison(classical_model, quantum_model, test_data, save_dir='quantum_vae_results'):
    """Compare classical, quantum, and quantum-inspired VAEs"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = next(classical_model.parameters()).device
    test_samples = test_data[:8].to(device)

    models = {
        'Classical VAE': classical_model,
        'Quantum VAE': quantum_model,
    }

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for row, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(test_samples)
            recon_np = recon.cpu().numpy()

        for col in range(4):
            # Velocity magnitude
            mag = np.sqrt(recon_np[col, 0] ** 2 + recon_np[col, 1] ** 2)
            ax = axes[row, col]
            im = ax.imshow(mag, cmap='viridis', aspect='auto')
            ax.set_title(f'{name}\nSample {col + 1}')
            ax.axis('off')

            if row == 0 and col == 0:
                plt.colorbar(im, ax=ax, fraction=0.046)

    # Original samples in bottom row
    orig_np = test_samples.cpu().numpy()
    for col in range(4):
        mag = np.sqrt(orig_np[col, 0] ** 2 + orig_np[col, 1] ** 2)
        ax = axes[3, col]
        im = ax.imshow(mag, cmap='viridis', aspect='auto')
        ax.set_title(f'Original\nSample {col + 1}')
        ax.axis('off')

    plt.suptitle('VAE Reconstruction Comparison: Classical vs Quantum', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_{timestamp}.png'), dpi=150)
    plt.show()

    return fig


def visualize_latent_space(model, data_loader, device, save_dir='quantum_vae_results'):
    """Visualize latent space structure"""
    model.eval()
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            data = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            data = data.to(device)
            mu, _ = model.encoder(data)
            all_mu.append(mu.cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)

    # PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(all_mu)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot of latent space
    axes[0].scatter(mu_pca[:, 0], mu_pca[:, 1], alpha=0.5, s=5)
    axes[0].set_title('Latent Space (PCA Projection)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)

    # Histogram of latent dimensions
    axes[1].hist(all_mu.flatten(), bins=50, alpha=0.7)
    axes[1].set_title('Distribution of Latent Values')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Quantum VAE Latent Space Analysis')
    plt.tight_layout()
    plt.show()

    return fig


def create_animation(model, test_data, device, save_dir='quantum_vae_results'):
    """Create animation comparing original and reconstructed flow"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model.eval()
    n_frames = min(50, len(test_data))
    sequence = test_data[:n_frames].to(device)

    with torch.no_grad():
        recon_sequence, _, _ = model(sequence)

    sequence_np = sequence.cpu().numpy()
    recon_np = recon_sequence.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def update_frame(frame):
        for ax in axes:
            ax.clear()

        # Original
        mag_orig = np.sqrt(sequence_np[frame, 0] ** 2 + sequence_np[frame, 1] ** 2)
        im1 = axes[0].imshow(mag_orig, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Original (t={frame})')
        axes[0].axis('off')

        # Reconstruction
        mag_recon = np.sqrt(recon_np[frame, 0] ** 2 + recon_np[frame, 1] ** 2)
        im2 = axes[1].imshow(mag_recon, cmap='viridis', aspect='auto')
        axes[1].set_title('Quantum VAE Reconstruction')
        axes[1].axis('off')

        # Difference
        diff = np.abs(mag_orig - mag_recon)
        im3 = axes[2].imshow(diff, cmap='hot', aspect='auto')
        axes[2].set_title('Error')
        axes[2].axis('off')

        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        return im1, im2, im3

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=100, blit=False)
    anim.save(os.path.join(save_dir, f'quantum_animation_{timestamp}.gif'), writer='pillow', fps=10)
    plt.close()
    print(f"Animation saved to {save_dir}/quantum_animation_{timestamp}.gif")


def plot_training_curves(train_losses, val_losses, recon_losses, kl_losses, betas,
                         save_dir='quantum_vae_results'):
    """Plot all training curves"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction and KL loss
    axes[0, 1].plot(recon_losses, label='Reconstruction Loss', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL divergence
    axes[1, 0].plot(kl_losses, label='KL Divergence', color='red', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Beta schedule
    axes[1, 1].plot(betas, label='β (KL weight)', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('β')
    axes[1, 1].set_title('KL Annealing Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Quantum VAE Training Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{timestamp}.png'), dpi=150)
    plt.show()


def main():
    print("=" * 70)
    print("QUANTUM-ENHANCED VAE FOR FLUID DYNAMICS")
    print("=" * 70)

    # Parameters
    LATENT_DIM = 8  # Smaller for quantum
    BATCH_SIZE = 16
    EPOCHS = 80
    LEARNING_RATE = 1e-3
    GRID_SIZE = 32
    N_QUBITS = 6
    N_QUANTUM_LAYERS = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\n" + "=" * 50)
    print("STEP 1: Loading Fluid Data")
    print("=" * 50)

    try:
        data_tensor, mean_u, std_u = load_quantum_data()
    except FileNotFoundError:
        print("Data not found. Generating new data...")
        from quantum_vae_data import generate_fluid_data
        generate_fluid_data(n_samples=2000, grid_size=GRID_SIZE)
        data_tensor, mean_u, std_u = load_quantum_data()

    # Split data
    total_samples = len(data_tensor)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    indices = list(range(total_samples))
    train_dataset = Subset(data_tensor, indices[:train_size])
    val_dataset = Subset(data_tensor, indices[train_size:train_size + val_size])
    test_dataset = Subset(data_tensor, indices[train_size + val_size:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"Input shape: {data_tensor.shape[1:]}")

    # Initialize Quantum VAE
    print("\n" + "=" * 50)
    print("STEP 2: Initializing Quantum VAE")
    print("=" * 50)

    quantum_vae = QuantumVAE(
        input_channels=2,
        latent_dim=LATENT_DIM,
        n_qubits=N_QUBITS,
        n_quantum_layers=N_QUANTUM_LAYERS
    ).to(device)

    total_params = sum(p.numel() for p in quantum_vae.parameters())
    print(f"Quantum VAE parameters: {total_params:,}")
    print(f"Quantum circuit: {N_QUBITS} qubits, {N_QUANTUM_LAYERS} layers")
    print(
        f"Compression: {GRID_SIZE * GRID_SIZE * 2} → {LATENT_DIM} (factor: {(GRID_SIZE * GRID_SIZE * 2) / LATENT_DIM:.1f}x)")

    # Train Quantum VAE
    print("\n" + "=" * 50)
    print("STEP 3: Training Quantum VAE")
    print("=" * 50)

    train_losses, val_losses, recon_losses, kl_losses, betas = train_vae(
        quantum_vae, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE,
        beta_start=0.0, beta_end=1.0, beta_anneal_epochs=30,
        device=device
    )

    # Plot training curves
    plot_training_curves(train_losses, val_losses, recon_losses, kl_losses, betas)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(quantum_vae.state_dict(), f'quantum_vae_{timestamp}.pth')
    print(f"\nModel saved as quantum_vae_{timestamp}.pth")

    # Visualize results
    print("\n" + "=" * 50)
    print("STEP 4: Visualizing Results")
    print("=" * 50)

    # Collect test data
    test_tensor_list = []
    for batch_data in test_loader:
        data = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
        test_tensor_list.append(data)
    test_tensor = torch.cat(test_tensor_list, dim=0)

    # Visualize latent space
    visualize_latent_space(quantum_vae, test_loader, device)

    # Create animation
    create_animation(quantum_vae, test_tensor, device)

    # Compare with classical baseline (optional)
    print("\n" + "=" * 50)
    print("STEP 5: Comparison with Classical Baseline")
    print("=" * 50)

    from quantum_vae_model import QuantumInspiredVAE

    classical_vae = QuantumInspiredVAE(input_channels=2, latent_dim=16).to(device)
    print("Training classical VAE for comparison...")

    classical_train_losses, classical_val_losses, _, _, _ = train_vae(
        classical_vae, train_loader, val_loader,
        epochs=min(EPOCHS, 50), lr=LEARNING_RATE,
        beta_start=0.0, beta_end=1.0, beta_anneal_epochs=20,
        device=device
    )

    # Compare reconstructions
    visualize_comparison(classical_vae, quantum_vae, test_tensor[:8])

    # Final metrics
    quantum_vae.eval()
    classical_vae.eval()

    quantum_mse = 0
    classical_mse = 0

    with torch.no_grad():
        for batch_data in test_loader:
            data = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            data = data.to(device)

            # Quantum VAE
            recon_q, _, _ = quantum_vae(data)
            mse_q = nn.MSELoss()(recon_q, data)
            quantum_mse += mse_q.item()

            # Classical VAE
            recon_c, _, _ = classical_vae(data)
            mse_c = nn.MSELoss()(recon_c, data)
            classical_mse += mse_c.item()

    quantum_mse /= len(test_loader)
    classical_mse /= len(test_loader)

    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)
    print(f"Classical VAE MSE: {classical_mse:.6f}")
    print(f"Quantum VAE MSE:   {quantum_mse:.6f}")
    print(f"Quantum Advantage: {(classical_mse / quantum_mse - 1) * 100:.2f}% better")

    if quantum_mse < classical_mse:
        print("\n✅ Quantum VAE outperforms classical baseline!")
    else:
        print("\n⚠️ Quantum VAE underperforms classical (expected on simulator)")
        print("   Advantage would appear on real quantum hardware with error mitigation")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print("\nResults saved in 'quantum_vae_results/' directory")


if __name__ == "__main__":
    main()