import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os


# ============================================
# QUANTUM VAE ARCHITECTURE (Fixed)
# ============================================
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.entanglement_weights = nn.Parameter(torch.randn(n_layers, n_qubits, n_qubits) * 0.1)
        self.layer_norm = nn.LayerNorm(n_qubits)
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
            residual = state
            state = torch.matmul(state, self.hadamard)
            phase = torch.sin(self.rotation_weights[layer] * np.pi)
            state = state * (1 + 0.5 * phase.unsqueeze(0))
            entanglement = torch.tanh(self.entanglement_weights[layer])
            entangled = torch.matmul(state, entanglement)
            state = 0.7 * state + 0.3 * entangled
            state = self.layer_norm(state)
        return state


class QuantumEncoder(nn.Module):
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
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=2)
        self.fc_mu = nn.Linear(n_qubits, latent_dim)
        self.fc_logvar = nn.Linear(n_qubits, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        quantum_input = torch.tanh(self.fc_to_quantum(x))
        quantum_state = self.quantum_layer(quantum_input)
        mu = self.fc_mu(quantum_state)
        logvar = self.fc_logvar(quantum_state)
        return mu, logvar


class QuantumDecoder(nn.Module):
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
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.deconv_layers(x)


class QuantumVAE(nn.Module):
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


# ============================================
# CLASSIC VAE ARCHITECTURE
# ============================================
class ClassicEncoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=16):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class ClassicDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_channels=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.deconv_layers(x)


class ClassicVAE(nn.Module):
    def __init__(self, input_channels=2, latent_dim=16):
        super().__init__()
        self.encoder = ClassicEncoder(input_channels, latent_dim)
        self.decoder = ClassicDecoder(latent_dim, input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ============================================
# LOAD FUNCTIONS
# ============================================
@st.cache_resource
def load_classic_model(model_dir='.'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    model = ClassicVAE(input_channels=2, latent_dim=16).to(device)
    model_files = glob.glob(os.path.join(model_dir, "classic_vae_model_*.pth"))

    if model_files:
        latest_model = sorted(model_files)[-1]
        checkpoint = torch.load(latest_model, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, device
    else:
        st.warning("⚠️ No classic model found. Using untrained model.")
        return model, device


@st.cache_resource
def load_quantum_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    model_files = glob.glob("quantum_vae_model_*.pth")

    if model_files:
        latest_model = sorted(model_files)[-1]
        checkpoint = torch.load(latest_model, map_location=device)

        latent_dim = checkpoint.get('latent_dim', 16)
        n_qubits = checkpoint.get('n_qubits', 8)

        model = QuantumVAE(input_channels=2, latent_dim=latent_dim, n_qubits=n_qubits).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        st.success(f"✅ Loaded quantum model from {os.path.basename(latest_model)}")
        return model, device
    else:
        st.error("❌ No quantum model found. Please run train_quantum_vae.py first.")
        return None, None


def generate_flow(flow_type, grid_size=64, time_step=0):
    """Generate different flow patterns"""
    x = np.linspace(-2, 4, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)

    if flow_type == "Cylinder Wake":
        U_inf = 1.0
        radius = 0.3

        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        r_safe = np.maximum(r, radius)

        ur = U_inf * (1 - radius ** 2 / r_safe ** 2) * np.cos(theta)
        utheta = -U_inf * (1 + radius ** 2 / r_safe ** 2) * np.sin(theta)

        u = ur * np.cos(theta) - utheta * np.sin(theta)
        v = ur * np.sin(theta) + utheta * np.cos(theta)

        # Time-varying vortex shedding
        strength = 0.4 * np.sin(0.5 * time_step)
        vortex_x = np.array([1.2, 2.0, 2.8])
        vortex_y = np.array([0.5, -0.5, 0.5]) * np.sin(0.8 * time_step)

        for vx, vy in zip(vortex_x, vortex_y):
            r_vort = np.sqrt((X - vx) ** 2 + (Y - vy) ** 2) + 0.1
            u += -strength * (Y - vy) / r_vort ** 2
            v += strength * (X - vx) / r_vort ** 2

        mask = r < radius
        u[mask] = 0
        v[mask] = 0
        cylinder_outline = (r >= radius - 0.05) & (r <= radius + 0.05)

    elif flow_type == "Vortex Pair":
        strength = 1.0
        offset = 0.3 * np.sin(0.3 * time_step)

        r1 = np.sqrt((X + 1) ** 2 + (Y - offset) ** 2) + 0.1
        u1 = -strength * (Y - offset) / r1 ** 2
        v1 = strength * (X + 1) / r1 ** 2

        r2 = np.sqrt((X - 1) ** 2 + (Y + offset) ** 2) + 0.1
        u2 = strength * (Y + offset) / r2 ** 2
        v2 = -strength * (X - 1) / r2 ** 2

        u = u1 + u2
        v = v1 + v2
        cylinder_outline = np.zeros_like(u, dtype=bool)

    else:  # Shear Flow
        shear_rate = 0.03
        u = 1.0 + shear_rate * Y
        v = 0.05 * np.sin(0.5 * X) * np.cos(time_step * 0.5)  # Small perturbation
        cylinder_outline = np.zeros_like(u, dtype=bool)

    # Normalize to [-1, 1]
    max_val = max(np.abs(u).max(), np.abs(v).max())
    if max_val > 0:
        u = u / max_val
        v = v / max_val

    return np.stack([u, v], axis=0), cylinder_outline, X, Y


def main():
    st.set_page_config(page_title="Classic vs Quantum VAE - Fluid Dynamics", layout="wide")

    st.title("🌊 Classic vs Quantum-Inspired VAE for Fluid Dynamics")
    st.markdown("""
    Compare **Classic VAE** and **Quantum-Inspired VAE** reconstructions of fluid flow fields.
    The quantum VAE uses quantum-inspired layers with entanglement to capture complex flow patterns.
    """)

    # Load models
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Classic VAE")
        classic_model, classic_device = load_classic_model()
    with col2:
        st.subheader("⚛️ Quantum VAE")
        quantum_model, quantum_device = load_quantum_model()

    if quantum_model is None:
        st.stop()

    # Sidebar controls
    st.sidebar.header("🎮 Controls")

    flow_type = st.sidebar.selectbox(
        "Flow Type",
        ["Cylinder Wake", "Vortex Pair", "Shear Flow"]
    )

    visualization_mode = st.sidebar.selectbox(
        "Visualization Mode",
        ["Streamlines", "Velocity Magnitude", "Quiver Plot"]
    )

    animation_speed = st.sidebar.slider("Animation Speed (fps)", 5, 30, 15)
    show_error = st.sidebar.checkbox("Show Reconstruction Error", value=True)

    # Main display
    plot_placeholder = st.empty()
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    # Animation loop
    frame = 0
    running = True

    while running:
        try:
            # Generate flow
            flow_data, cylinder_outline, X, Y = generate_flow(flow_type, grid_size=64, time_step=frame * 0.1)
            flow_tensor = torch.FloatTensor(flow_data).unsqueeze(0)

            # Classic VAE inference
            with torch.no_grad():
                if classic_model is not None:
                    classic_recon, _, _ = classic_model(flow_tensor.to(classic_device))
                    classic_recon_np = classic_recon.squeeze().cpu().numpy()
                    classic_mse = np.mean((flow_data - classic_recon_np) ** 2)
                else:
                    classic_recon_np = flow_data
                    classic_mse = 0

                # Quantum VAE inference
                quantum_recon, _, _ = quantum_model(flow_tensor.to(quantum_device))
                quantum_recon_np = quantum_recon.squeeze().cpu().numpy()
                quantum_mse = np.mean((flow_data - quantum_recon_np) ** 2)

            # Display metrics
            metrics_col1.metric("Classic VAE MSE", f"{classic_mse:.6f}")
            metrics_col2.metric("Quantum VAE MSE", f"{quantum_mse:.6f}")
            if classic_mse > 0:
                improvement = ((classic_mse - quantum_mse) / classic_mse) * 100
                metrics_col3.metric("Quantum Improvement", f"{improvement:.2f}%",
                                    delta="better" if improvement > 0 else "worse")

            # Create visualization
            if show_error:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            else:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes = [axes[0], axes[1], axes[2]]

            # Helper function for plotting
            def plot_flow(ax, u, v, X, Y, title, cylinder=None):
                if visualization_mode == "Streamlines":
                    mag = np.sqrt(u ** 2 + v ** 2)
                    stream = ax.streamplot(X, Y, u, v, color=mag, cmap='viridis',
                                           linewidth=1, density=1.5)
                    plt.colorbar(stream.lines, ax=ax, fraction=0.046)
                elif visualization_mode == "Velocity Magnitude":
                    mag = np.sqrt(u ** 2 + v ** 2)
                    im = ax.imshow(mag, extent=[-2, 4, -1.5, 1.5], origin='lower',
                                   cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                else:  # Quiver Plot
                    step = 5
                    X_sub = X[::step, ::step]
                    Y_sub = Y[::step, ::step]
                    u_sub = u[::step, ::step]
                    v_sub = v[::step, ::step]
                    mag = np.sqrt(u_sub ** 2 + v_sub ** 2)
                    q = ax.quiver(X_sub, Y_sub, u_sub, v_sub, mag, cmap='viridis',
                                  scale=20, width=0.005)
                    plt.colorbar(q, ax=ax, fraction=0.046)

                if cylinder is not None and np.any(cylinder):
                    ax.contour(X, Y, cylinder, levels=[0.5], colors='red', linewidths=2)

                ax.set_title(title)
                ax.set_aspect('equal')
                ax.axis('off')

            if show_error:
                # Plot original and reconstructions
                plot_flow(axes[0, 0], flow_data[0], flow_data[1], X, Y,
                          f"Original Flow", cylinder_outline)
                plot_flow(axes[0, 1], classic_recon_np[0], classic_recon_np[1], X, Y,
                          f"Classic VAE\nMSE: {classic_mse:.6f}", cylinder_outline)
                plot_flow(axes[1, 0], quantum_recon_np[0], quantum_recon_np[1], X, Y,
                          f"Quantum VAE\nMSE: {quantum_mse:.6f}", cylinder_outline)

                # Error plots
                classic_error = np.abs(np.sqrt(flow_data[0] ** 2 + flow_data[1] ** 2) -
                                       np.sqrt(classic_recon_np[0] ** 2 + classic_recon_np[1] ** 2))
                quantum_error = np.abs(np.sqrt(flow_data[0] ** 2 + flow_data[1] ** 2) -
                                       np.sqrt(quantum_recon_np[0] ** 2 + quantum_recon_np[1] ** 2))

                im1 = axes[1, 1].imshow(classic_error, extent=[-2, 4, -1.5, 1.5],
                                        origin='lower', cmap='hot', aspect='auto')
                axes[1, 1].set_title(f'Classic Error (mean: {classic_mse:.6f})')
                axes[1, 1].axis('off')
                plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
            else:
                plot_flow(axes[0], flow_data[0], flow_data[1], X, Y,
                          f"Original Flow", cylinder_outline)
                plot_flow(axes[1], classic_recon_np[0], classic_recon_np[1], X, Y,
                          f"Classic VAE\nMSE: {classic_mse:.6f}", cylinder_outline)
                plot_flow(axes[2], quantum_recon_np[0], quantum_recon_np[1], X, Y,
                          f"Quantum VAE\nMSE: {quantum_mse:.6f}", cylinder_outline)

            plt.tight_layout()
            plot_placeholder.pyplot(fig)
            plt.close(fig)

            time.sleep(1.0 / animation_speed)
            frame += 1
            if frame > 200:
                frame = 0

        except Exception as e:
            st.error(f"Error: {e}")
            break


if __name__ == "__main__":
    main()