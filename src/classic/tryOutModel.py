import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.streamplot import streamplot


# Define the VAE architecture (same as before)
class Encoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=16):
        super(Encoder, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, output_channels=2):
        super(Decoder, self).__init__()
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


class VAE(nn.Module):
    def __init__(self, input_channels=2, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
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


@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_channels=2, latent_dim=16).to(device)

    # Find the most recent model
    model_files = glob.glob("vae_model_*.pth")
    if model_files:
        latest_model = sorted(model_files)[-1]
        model.load_state_dict(torch.load(latest_model, map_location=device))
        model.eval()
        return model, device
    else:
        st.error("No trained model found. Please run train_fluid_vae.py first.")
        return None, None


def generate_flow(flow_type, grid_size=64, time_step=0):
    """Generate different flow patterns based on user selection"""

    x = np.linspace(-2, 4, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)

    if flow_type == "Cylinder Wake (von Kármán)":
        # Cylinder wake with time-varying vortex street
        U_inf = 1.0
        cylinder_radius = 0.3
        vortex_shedding_freq = 0.2

        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        r_safe = np.maximum(r, cylinder_radius)

        # Potential flow
        ur = U_inf * (1 - cylinder_radius ** 2 / r_safe ** 2) * np.cos(theta)
        utheta = -U_inf * (1 + cylinder_radius ** 2 / r_safe ** 2) * np.sin(theta)

        u_pot = ur * np.cos(theta) - utheta * np.sin(theta)
        v_pot = ur * np.sin(theta) + utheta * np.cos(theta)

        # Vortices
        vortex_strength = 0.5 * np.sin(2 * np.pi * vortex_shedding_freq * time_step)
        vortex_x = np.array([1.0, 2.0, 3.0])
        vortex_y = np.array([0.5, -0.5, 0.5]) * np.sin(2 * np.pi * vortex_shedding_freq * time_step)

        u_vort = np.zeros_like(X)
        v_vort = np.zeros_like(Y)

        for vx, vy in zip(vortex_x, vortex_y):
            r_vort = np.sqrt((X - vx) ** 2 + (Y - vy) ** 2) + 0.1
            u_vort += -vortex_strength * (Y - vy) / r_vort ** 2
            v_vort += vortex_strength * (X - vx) / r_vort ** 2

        u = u_pot + u_vort
        v = v_pot + v_vort

        # Mask cylinder
        mask = r < cylinder_radius
        u[mask] = 0
        v[mask] = 0

        # Add cylinder outline for visualization
        cylinder_outline = (r >= cylinder_radius - 0.05) & (r <= cylinder_radius + 0.05)

    elif flow_type == "Uniform Flow":
        u = np.ones((grid_size, grid_size))
        v = np.zeros((grid_size, grid_size))
        cylinder_outline = np.zeros_like(u, dtype=bool)

    elif flow_type == "Vortex Pair":
        strength = 1.0
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        X, Y = np.meshgrid(x, y)

        r1 = np.sqrt((X + 1) ** 2 + Y ** 2) + 0.1
        u1 = -strength * Y / r1 ** 2
        v1 = strength * (X + 1) / r1 ** 2

        r2 = np.sqrt((X - 1) ** 2 + Y ** 2) + 0.1
        u2 = strength * Y / r2 ** 2
        v2 = -strength * (X - 1) / r2 ** 2

        u = u1 + u2
        v = v1 + v2
        cylinder_outline = np.zeros_like(u, dtype=bool)

    elif flow_type == "Shear Flow":
        shear_rate = 0.02
        y_vals = np.linspace(-1, 1, grid_size)
        u = 1.0 + shear_rate * y_vals.reshape(-1, 1)
        u = np.broadcast_to(u, (grid_size, grid_size))
        v = np.zeros((grid_size, grid_size))
        cylinder_outline = np.zeros_like(u, dtype=bool)

    else:  # Backward Facing Step
        u = np.ones((grid_size, grid_size))
        v = np.zeros((grid_size, grid_size))
        step_x = grid_size // 2
        step_y = grid_size // 2
        u[step_y:, step_x:] = 0.5
        u[:step_y, step_x:] = 0.8
        cylinder_outline = np.zeros_like(u, dtype=bool)

    return np.stack([u, v], axis=0), cylinder_outline


def plot_streamlines(u, v, X, Y, cylinder_outline=None, title="Streamlines", figsize=(8, 6)):
    """Create a streamline plot using matplotlib"""
    fig, ax = plt.subplots(figsize=figsize)

    # Subsample for better performance
    step = 3
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]

    # Calculate velocity magnitude for coloring
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # Create streamplot
    stream = ax.streamplot(X, Y, u, v,
                           color=magnitude,
                           cmap='viridis',
                           linewidth=1,
                           density=1.5,
                           arrowstyle='->',
                           arrowsize=1)

    # Add colorbar
    plt.colorbar(stream.lines, ax=ax, label='Velocity Magnitude')

    # Add cylinder outline if provided
    if cylinder_outline is not None and np.any(cylinder_outline):
        ax.contour(X, Y, cylinder_outline, levels=[0.5], colors='red', linewidths=2)
        ax.fill_between([], [], [], color='red', label='Obstacle')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig


def plot_quiver(u, v, X, Y, cylinder_outline=None, title="Vector Field", figsize=(8, 6)):
    """Create a quiver (arrow) plot"""
    fig, ax = plt.subplots(figsize=figsize)

    # Subsample for better visibility (show every 4th point)
    step = 4
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]

    # Calculate magnitude for coloring
    magnitude = np.sqrt(u_sub ** 2 + v_sub ** 2)

    # Create quiver plot
    quiver = ax.quiver(X_sub, Y_sub, u_sub, v_sub, magnitude,
                       cmap='viridis',
                       scale=30,
                       width=0.005,
                       headwidth=3,
                       headlength=4)

    # Add colorbar
    plt.colorbar(quiver, ax=ax, label='Velocity Magnitude')

    # Add cylinder outline
    if cylinder_outline is not None and np.any(cylinder_outline):
        ax.contour(X, Y, cylinder_outline, levels=[0.5], colors='red', linewidths=2)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig


def plot_pathlines_animation(u_sequence, v_sequence, X, Y, num_particles=50):
    """Create an animation of particle paths (pathlines)"""
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 8))

    # Initialize particles at random positions
    np.random.seed(42)
    particles_x = np.random.uniform(X.min(), X.max(), num_particles)
    particles_y = np.random.uniform(Y.min(), Y.max(), num_particles)

    # Store particle trails
    trails = [[] for _ in range(num_particles)]
    trail_length = 20

    scatter = ax.scatter(particles_x, particles_y, c='blue', s=30, alpha=0.7)

    def update(frame):
        nonlocal particles_x, particles_y
        u = u_sequence[frame]
        v = v_sequence[frame]

        # Update particle positions using velocity interpolation
        for i in range(num_particles):
            # Find nearest grid point
            ix = np.argmin(np.abs(X[0, :] - particles_x[i]))
            iy = np.argmin(np.abs(Y[:, 0] - particles_y[i]))

            # Add velocity
            particles_x[i] += u[iy, ix] * 0.1
            particles_y[i] += v[iy, ix] * 0.1

            # Keep particles in bounds
            particles_x[i] = np.clip(particles_x[i], X.min(), X.max())
            particles_y[i] = np.clip(particles_y[i], Y.min(), Y.max())

            # Update trails
            trails[i].append((particles_x[i], particles_y[i]))
            if len(trails[i]) > trail_length:
                trails[i].pop(0)

        # Update scatter
        scatter.set_offsets(np.c_[particles_x, particles_y])

        # Update trails
        for i in range(num_particles):
            if len(trails[i]) > 1:
                trail = np.array(trails[i])
                ax.plot(trail[:, 0], trail[:, 1], 'b-', alpha=0.3, linewidth=1)

        ax.set_title(f'Particle Pathlines - Frame {frame}')
        return scatter,

    anim = FuncAnimation(fig, update, frames=len(u_sequence), interval=100, blit=False)
    return anim


def main():
    st.set_page_config(page_title="Fluid Dynamics VAE - Vector Field Visualization", layout="wide")

    st.title("🌊 Real-Time Fluid Flow Visualization with VAE")
    st.markdown("""
    This app demonstrates real-time compression and reconstruction of fluid flow fields using a Variational Autoencoder (VAE).
    **Now with vector field visualization!** See the actual flow patterns with streamlines and arrows.
    """)

    # Load model
    model, device = load_model()
    if model is None:
        return

    # Sidebar controls
    st.sidebar.header("🎮 Controls")

    flow_type = st.sidebar.selectbox(
        "Flow Type",
        ["Cylinder Wake (von Kármán)", "Uniform Flow", "Vortex Pair", "Shear Flow", "Backward Step"]
    )

    visualization_mode = st.sidebar.selectbox(
        "Visualization Mode",
        ["Streamlines (Flow Lines)", "Quiver (Arrow Field)", "Heatmap (Velocity Magnitude)", "Combined View"]
    )

    animation_speed = st.sidebar.slider("Animation Speed (fps)", 5, 30, 15)
    show_reconstruction = st.sidebar.checkbox("Show VAE Reconstruction", value=True)
    show_error = st.sidebar.checkbox("Show Reconstruction Error", value=False)

    # Advanced controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Latent Space Manipulation")

    latent_dims = st.sidebar.multiselect(
        "Select latent dimensions to manipulate",
        list(range(16)),
        default=[0, 1, 2]
    )

    latent_values = {}
    for dim in latent_dims:
        latent_values[dim] = st.sidebar.slider(f"Dimension {dim}", -3.0, 3.0, 0.0, 0.1)

    # Create coordinate grid for visualization
    grid_size = 64
    x = np.linspace(-2, 4, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)

    # Main display area
    if visualization_mode == "Combined View":
        col1, col2 = st.columns(2)
    else:
        col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Original Flow Field")
        original_plot = st.empty()

    with col2:
        if show_reconstruction:
            st.subheader("🔄 VAE Reconstruction")
            recon_plot = st.empty()

    if show_error:
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📊 Reconstruction Error")
            error_plot = st.empty()

    # Metrics
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Compression Ratio", "512:1", "64x64x2 → 16 dims")
    mse_display = metric_cols[1].empty()
    psnr_display = metric_cols[2].empty()
    active_dims_display = metric_cols[3].empty()

    # Real-time animation
    st.markdown("---")
    status_text = st.empty()

    # Animation loop
    frame = 0
    running = True
    stop_button = st.button("Stop Animation")

    while running:
        try:
            if stop_button:
                running = False
                break

            # Generate flow
            flow_data, cylinder_outline = generate_flow(flow_type, grid_size=64, time_step=frame * 0.1)

            # Prepare tensor
            flow_tensor = torch.FloatTensor(flow_data).unsqueeze(0).to(device)

            # Process through VAE
            with torch.no_grad():
                recon, mu, _ = model(flow_tensor)

                # Apply latent manipulations if any
                if latent_values and any(v != 0 for v in latent_values.values()):
                    mu_modified = mu.clone()
                    for dim, val in latent_values.items():
                        mu_modified[0, dim] += val
                    recon = model.decoder(mu_modified)
                    latent_vector = mu_modified.cpu().numpy()[0]
                else:
                    latent_vector = mu.cpu().numpy()[0]

            # Convert to numpy
            flow_np = flow_data
            recon_np = recon.squeeze().cpu().numpy()

            # Calculate metrics
            orig_mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)
            recon_mag = np.sqrt(recon_np[0] ** 2 + recon_np[1] ** 2)
            error = np.abs(orig_mag - recon_mag)
            mse = np.mean(error ** 2)
            psnr = 20 * np.log10(2.0 / np.sqrt(mse + 1e-10))

            # Update metrics
            mse_display.metric("Current MSE", f"{mse:.6f}")
            psnr_display.metric("Current PSNR", f"{psnr:.2f} dB")
            active_dims = np.sum(np.abs(latent_vector) > 0.1)
            active_dims_display.metric("Latent Dims Active", f"{active_dims}/16")

            # Create visualizations based on mode
            if visualization_mode == "Streamlines (Flow Lines)":
                fig_orig = plot_streamlines(flow_np[0], flow_np[1], X, Y, cylinder_outline,
                                            title=f"Original Flow - Frame {frame}")
                original_plot.pyplot(fig_orig)
                plt.close(fig_orig)

                if show_reconstruction:
                    fig_recon = plot_streamlines(recon_np[0], recon_np[1], X, Y, cylinder_outline,
                                                 title=f"VAE Reconstruction - MSE: {mse:.6f}")
                    recon_plot.pyplot(fig_recon)
                    plt.close(fig_recon)

                if show_error:
                    fig_error = plot_streamlines(error, np.zeros_like(error), X, Y, None,
                                                 title=f"Error Field")
                    error_plot.pyplot(fig_error)
                    plt.close(fig_error)

            elif visualization_mode == "Quiver (Arrow Field)":
                fig_orig = plot_quiver(flow_np[0], flow_np[1], X, Y, cylinder_outline,
                                       title=f"Original Flow - Frame {frame}")
                original_plot.pyplot(fig_orig)
                plt.close(fig_orig)

                if show_reconstruction:
                    fig_recon = plot_quiver(recon_np[0], recon_np[1], X, Y, cylinder_outline,
                                            title=f"VAE Reconstruction - MSE: {mse:.6f}")
                    recon_plot.pyplot(fig_recon)
                    plt.close(fig_recon)

            elif visualization_mode == "Heatmap (Velocity Magnitude)":
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(orig_mag, extent=[-2, 4, -1.5, 1.5], origin='lower', cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax, label='Velocity Magnitude')
                if np.any(cylinder_outline):
                    ax.contour(X, Y, cylinder_outline, levels=[0.5], colors='red', linewidths=2)
                ax.set_title(f"Original Flow - Frame {frame}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                original_plot.pyplot(fig)
                plt.close(fig)

                if show_reconstruction:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    im2 = ax2.imshow(recon_mag, extent=[-2, 4, -1.5, 1.5], origin='lower', cmap='viridis',
                                     aspect='auto')
                    plt.colorbar(im2, ax=ax2, label='Velocity Magnitude')
                    ax2.set_title(f"VAE Reconstruction - MSE: {mse:.6f}")
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    recon_plot.pyplot(fig2)
                    plt.close(fig2)

            else:  # Combined View
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Original
                stream1 = ax1.streamplot(X, Y, flow_np[0], flow_np[1],
                                         color=np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2),
                                         cmap='viridis', linewidth=1, density=1.5)
                if np.any(cylinder_outline):
                    ax1.contour(X, Y, cylinder_outline, levels=[0.5], colors='red', linewidths=2)
                ax1.set_title(f'Original Flow - Frame {frame}')
                ax1.set_aspect('equal')
                plt.colorbar(stream1.lines, ax=ax1, label='Velocity')

                # Reconstruction
                stream2 = ax2.streamplot(X, Y, recon_np[0], recon_np[1],
                                         color=np.sqrt(recon_np[0] ** 2 + recon_np[1] ** 2),
                                         cmap='viridis', linewidth=1, density=1.5)
                if np.any(cylinder_outline):
                    ax2.contour(X, Y, cylinder_outline, levels=[0.5], colors='red', linewidths=2)
                ax2.set_title(f'VAE Reconstruction - MSE: {mse:.6f}')
                ax2.set_aspect('equal')
                plt.colorbar(stream2.lines, ax=ax2, label='Velocity')

                original_plot.pyplot(fig)
                plt.close(fig)

            status_text.info(
                f"🔄 Streaming frame {frame} | Flow: {flow_type} | Mode: {visualization_mode} | MSE: {mse:.6f}")

            # Control animation speed
            time.sleep(1.0 / animation_speed)
            frame += 1
            if frame > 200:
                frame = 0

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            break


if __name__ == "__main__":
    main()