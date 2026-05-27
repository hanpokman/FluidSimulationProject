import numpy as np
import os


def generate_fluid_data(n_samples=2000, grid_size=64, save_dir='quantum_fluid_data'):
    """Generate synthetic fluid flow data for 64x64 grid"""
    os.makedirs(save_dir, exist_ok=True)

    u_fields = []
    v_fields = []

    for i in range(n_samples):
        # Create coordinate grid
        x = np.linspace(-2, 4, grid_size)
        y = np.linspace(-1.5, 1.5, grid_size)
        X, Y = np.meshgrid(x, y)

        # Random flow type
        flow_type = np.random.choice(['vortex_pair', 'shear_vortex', 'cylinder_wake'])

        if flow_type == 'vortex_pair':
            # Counter-rotating vortex pair
            strength = np.random.uniform(0.8, 1.5)
            offset = np.random.uniform(-0.5, 0.5)

            r1 = np.sqrt((X + 1) ** 2 + (Y - offset) ** 2) + 0.1
            u1 = -strength * (Y - offset) / r1 ** 2
            v1 = strength * (X + 1) / r1 ** 2

            r2 = np.sqrt((X - 1) ** 2 + (Y + offset) ** 2) + 0.1
            u2 = strength * (Y + offset) / r2 ** 2
            v2 = -strength * (X - 1) / r2 ** 2

            u = u1 + u2
            v = v1 + v2

        elif flow_type == 'shear_vortex':
            # Shear flow with embedded vortex
            shear_rate = np.random.uniform(0.01, 0.08)
            u = 1.0 + shear_rate * Y
            v = np.zeros_like(X)

            vortex_x = np.random.uniform(0, 2)
            vortex_y = np.random.uniform(-0.8, 0.8)
            strength = np.random.uniform(0.3, 0.8)

            r_vort = np.sqrt((X - vortex_x) ** 2 + (Y - vortex_y) ** 2) + 0.1
            u += -strength * (Y - vortex_y) / r_vort ** 2
            v += strength * (X - vortex_x) / r_vort ** 2

        else:  # cylinder_wake
            # Flow past cylinder with vortex shedding
            U_inf = 1.0
            radius = 0.3

            r = np.sqrt(X ** 2 + Y ** 2)
            theta = np.arctan2(Y, X)
            r_safe = np.maximum(r, radius)

            ur = U_inf * (1 - radius ** 2 / r_safe ** 2) * np.cos(theta)
            utheta = -U_inf * (1 + radius ** 2 / r_safe ** 2) * np.sin(theta)

            u = ur * np.cos(theta) - utheta * np.sin(theta)
            v = ur * np.sin(theta) + utheta * np.cos(theta)

            # Add shedding vortices
            strength = np.random.uniform(0.2, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)

            vortex_positions = [(1.2, 0.5), (2.0, -0.5), (2.8, 0.5)]
            for x_pos, y_pos in vortex_positions:
                r_vort = np.sqrt((X - x_pos) ** 2 + (Y - y_pos) ** 2) + 0.1
                u += -strength * np.sin(phase) * (Y - y_pos) / r_vort ** 2
                v += strength * np.cos(phase) * (X - x_pos) / r_vort ** 2

            # Mask cylinder interior
            mask = r < radius
            u[mask] = 0
            v[mask] = 0

        u_fields.append(u)
        v_fields.append(v)

    # Convert to numpy arrays
    u_fields = np.array(u_fields)
    v_fields = np.array(v_fields)

    # Normalize to [-1, 1] range
    global_min_u = np.min(u_fields)
    global_max_u = np.max(u_fields)
    global_min_v = np.min(v_fields)
    global_max_v = np.max(v_fields)

    u_normalized = 2 * (u_fields - global_min_u) / (global_max_u - global_min_u + 1e-8) - 1
    v_normalized = 2 * (v_fields - global_min_v) / (global_max_v - global_min_v + 1e-8) - 1

    # Stack channels
    data = np.stack([u_normalized, v_normalized], axis=1)

    # Save
    save_path = os.path.join(save_dir, 'quantum_fluid_data.npz')
    np.savez(save_path,
             data=data,
             u_min=global_min_u, u_max=global_max_u,
             v_min=global_min_v, v_max=global_max_v)

    print(f"Generated {n_samples} samples, shape: {data.shape}")
    print(f"U range: [{global_min_u:.3f}, {global_max_u:.3f}]")
    print(f"V range: [{global_min_v:.3f}, {global_max_v:.3f}]")
    print(f"Saved to {save_path}")

    return save_path, data


if __name__ == "__main__":
    generate_fluid_data(n_samples=2000, grid_size=64)