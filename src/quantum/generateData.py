import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from datetime import datetime


def generate_cylinder_wake_data(grid_size=64, num_snapshots=1000, save_dir='fluid_data'):
    """
    Generate 2D flow around cylinder (von Kármán vortex street) data
    Returns: velocity field snapshots (u, v) and vorticity
    """
    print(f"Generating {num_snapshots} fluid dynamics snapshots...")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Grid setup
    x = np.linspace(-2, 4, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)

    # Cylinder parameters (center at origin)
    cylinder_radius = 0.3

    # Flow parameters
    U_inf = 1.0  # Free stream velocity
    vortex_shedding_freq = 0.2  # Strouhal number effect

    # Storage arrays
    u_snapshots = []
    v_snapshots = []
    vorticity_snapshots = []

    for t_idx in range(num_snapshots):
        t = t_idx * 0.1  # Time step

        # Base potential flow + vortex shedding
        # Distance from cylinder
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)

        # Avoid division by zero inside cylinder
        r_safe = np.maximum(r, cylinder_radius)

        # Potential flow around cylinder
        ur_potential = U_inf * (1 - cylinder_radius ** 2 / r_safe ** 2) * np.cos(theta)
        utheta_potential = -U_inf * (1 + cylinder_radius ** 2 / r_safe ** 2) * np.sin(theta)

        # Convert to Cartesian
        u_potential = ur_potential * np.cos(theta) - utheta_potential * np.sin(theta)
        v_potential = ur_potential * np.sin(theta) + utheta_potential * np.cos(theta)

        # Add von Kármán vortex street (alternating vortices)
        vortex_strength = 0.5 * np.sin(2 * np.pi * vortex_shedding_freq * t)

        # Vortex positions (alternating)
        vortex_x = np.array([1.0, 2.0, 3.0])
        vortex_y = np.array([0.5, -0.5, 0.5]) * np.sin(2 * np.pi * vortex_shedding_freq * t)

        u_vortex = np.zeros_like(X)
        v_vortex = np.zeros_like(Y)

        for vx, vy in zip(vortex_x, vortex_y):
            # Vortex-induced velocity (Rankine vortex model)
            r_vortex = np.sqrt((X - vx) ** 2 + (Y - vy) ** 2) + 0.1
            u_vortex += -vortex_strength * (Y - vy) / r_vortex ** 2
            v_vortex += vortex_strength * (X - vx) / r_vortex ** 2

        # Combine flows
        u = u_potential + u_vortex
        v = v_potential + v_vortex

        # Apply mask for cylinder interior (zero velocity inside cylinder)
        inside_cylinder = r < cylinder_radius
        u[inside_cylinder] = 0
        v[inside_cylinder] = 0

        # Add small random perturbations for realism
        noise = np.random.randn(grid_size, grid_size) * 0.01
        u += noise
        v += noise

        # Apply slight smoothing for physical realism
        u = gaussian_filter(u, sigma=0.5)
        v = gaussian_filter(v, sigma=0.5)

        # Compute vorticity
        vorticity = np.gradient(v, axis=0) - np.gradient(u, axis=1)

        u_snapshots.append(u)
        v_snapshots.append(v)
        vorticity_snapshots.append(vorticity)

        # Progress indicator
        if (t_idx + 1) % 100 == 0:
            print(f"Generated {t_idx + 1}/{num_snapshots} snapshots")

    # Convert to numpy arrays
    u_snapshots = np.array(u_snapshots)
    v_snapshots = np.array(v_snapshots)
    vorticity_snapshots = np.array(vorticity_snapshots)

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(os.path.join(save_dir, f'fluid_data_{timestamp}.npz'),
             u=u_snapshots, v=v_snapshots, vorticity=vorticity_snapshots,
             grid_size=grid_size, num_snapshots=num_snapshots)

    # Save a few visualization examples (use 4 snapshots for 2x2 grid)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    time_indices = [0, num_snapshots // 4, num_snapshots // 2, 3 * num_snapshots // 4]

    for idx, t_idx in enumerate(time_indices):
        # Velocity magnitude
        mag = np.sqrt(u_snapshots[t_idx] ** 2 + v_snapshots[t_idx] ** 2)
        im1 = axes[0, idx].contourf(X, Y, mag, levels=20, cmap='viridis')
        axes[0, idx].set_title(f'Velocity at t={t_idx * 0.1:.1f}')
        axes[0, idx].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, idx])

        # Vorticity
        im2 = axes[1, idx].contourf(X, Y, vorticity_snapshots[t_idx], levels=20, cmap='RdBu_r')
        axes[1, idx].set_title(f'Vorticity at t={t_idx * 0.1:.1f}')
        axes[1, idx].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1, idx])

    plt.suptitle('Fluid Flow Snapshots (Cylinder Wake)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_visualizations_{timestamp}.png'), dpi=150)
    plt.show()

    print(f"\nData saved to {save_dir}/fluid_data_{timestamp}.npz")
    print(f"Data shape - u: {u_snapshots.shape}, v: {v_snapshots.shape}, vorticity: {vorticity_snapshots.shape}")

    return u_snapshots, v_snapshots, vorticity_snapshots


if __name__ == "__main__":
    # Generate dataset
    u, v, vorticity = generate_cylinder_wake_data(grid_size=64, num_snapshots=1000)

    print("\nDataset generation complete!")
    print(f"Data ranges - u: [{u.min():.3f}, {u.max():.3f}], v: [{v.min():.3f}, {v.max():.3f}]")