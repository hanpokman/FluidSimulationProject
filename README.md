
# Quantum-Inspired Variational Autoencoder for Fluid Dynamics

## Overview

This project implements a **hybrid quantum-classical Variational Autoencoder (VAE)** for compressible representation and reconstruction of 2D fluid flow fields. Unlike classical VAEs that use purely neural network encoders, this model incorporates a **quantum-inspired layer** that simulates quantum phenomena (superposition, entanglement, and measurement) to create richer latent representations.

### Why Quantum-Inspired?

| Property | Classical VAE | Quantum-Inspired VAE |
|----------|---------------|----------------------|
| **Representation** | Deterministic latent space | Entangled, superposition-based |
| **Feature Correlation** | Learned through weights | Explicit entanglement simulation |
| **Latent Manifold** | Euclidean | Hilbert space-like |
| **Expressiveness** | Limited by network depth | Enhanced by quantum operations |

## Model Architecture

### Encoder: Classical → Quantum → Latent

```
Input Flow Field (64×64×2)
    ↓
Classical Convolutional Layers (4 layers, 32→256 channels)
    ↓
Feature Vector (256 dimensions)
    ↓
Quantum Encoding (8 qubits)
    ├── Hadamard Transform (superposition)
    ├── Phase Rotations (interference)
    ├── Entanglement Gates (qubit correlations)
    └── Measurement (collapse to classical)
    ↓
Latent Vector (16 dimensions: μ and log σ²)
```

### Decoder: Latent → Flow Field

```
Latent Vector (16 dimensions)
    ↓
Fully Connected Layer (→ 256×4×4)
    ↓
Transposed Convolutional Layers (4 layers, 256→2 channels)
    ↓
Reconstructed Flow Field (64×64×2)
```


### Training the Quantum VAE

```bash
cd src/quantum
python train_quantum_vae.py
```

**Expected output:**
```
============================================================
TRAINING QUANTUM-INSPIRED VAE FOR FLUID DYNAMICS
============================================================
Using device: mps
Data shape: torch.Size([2000, 2, 64, 64])
Train: 1400, Val: 300, Test: 300
Model parameters: 941,890

Training...
Epoch 10/100: Train Loss: 0.0234, Val Loss: 0.0241
Epoch 20/100: Train Loss: 0.0187, Val Loss: 0.0192
...
✓ Saved best model to quantum_vae_model_20241201_143022.pth (val_loss: 0.0158)
```

### Running the Comparison App

```bash
streamlit run visualize_comparison.py
```

This opens a browser window showing:
- Left panel: Original flow field
- Middle panel: Classic VAE reconstruction  
- Right panel: Quantum VAE reconstruction
- Sidebar: Controls for flow type, visualization mode, animation speed
