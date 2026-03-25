import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from act_copy.cvae_lucas.nn_models_updated import TransformerActionChunkingCVAE
from act_copy.ChunkedEpisodicDataset import load_data_chunked

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Config
# ---------------------------
checkpoint_path = "act_copy/cvae_lucas/model3_kl_20.pth"
num_batches = 1000        # how much data to visualize
max_points = 10000        # cap for plotting
use_pca = True
MAX_BATCHES = 1000        # limit for speed

# Dataset parameters
dataset_dir = '/media/pdz/Elements1/two_cams.hdf5'
num_episodes = 50
camera_names = ['image', 'image2']
num_cameras = len(camera_names)

# Model parameters
chunk_size = 32
latent_dim = 64
image_feature_dim = 128
context_length = 1

# Robot parameters
qpos_dim = 8
action_dim = 8

# Training parameters
batch_size_train = 128
batch_size_val = 64
num_epochs = 200
learning_rate = 1e-4
beta_max = 1
warmup = 0
beta_steps = 2000

# ---------------------------
# Load data
# ---------------------------
train_dataloader, val_dataloader, norm_stats, is_sim = load_data_chunked(
    dataset_dir=dataset_dir,
    num_episodes=num_episodes,
    camera_names=camera_names,
    batch_size_train=batch_size_train,
    batch_size_val=batch_size_val,
    chunk_size=chunk_size,
    context_length=context_length,
    velocity_control=False
)

# ---------------------------
# Load model
# ---------------------------
model = TransformerActionChunkingCVAE(
    num_cameras=num_cameras,
    qpos_dim=qpos_dim,
    action_dim=action_dim,
    chunk_size=chunk_size,
    latent_dim=latent_dim,
    context_length=context_length,
    image_feature_dim=image_feature_dim,
).to(DEVICE)

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# ---------------------------
# Collect latents
# ---------------------------
mus = []
logvars = []

with torch.no_grad():
    for i, batch in enumerate(train_dataloader):
        if i >= MAX_BATCHES:
            break
        if len(batch) == 5:
            images, qpos, actions, is_pad, _ = batch
        else:
            images, qpos, actions, is_pad = batch
        
        images = images.to(DEVICE)
        qpos = qpos.to(DEVICE)
        actions = actions.to(DEVICE)
        
        mu, logvar = model.encode(images, qpos, actions)
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())

mus = np.concatenate(mus, axis=0)
logvars = np.concatenate(logvars, axis=0)
print("Collected latents:", mus.shape)

# -------------------------
# PCA to 2D if needed
# -------------------------
if latent_dim > 2:
    print("Applying PCA to 2D...")
    pca = PCA(n_components=2)
    z = pca.fit_transform(mus)
else:
    z = mus

# -------------------------
# Create aggregated posterior density
# -------------------------
print("Computing aggregated posterior density...")

# Define grid
x_min, x_max = z[:, 0].min() - 1, z[:, 0].max() + 1
y_min, y_max = z[:, 1].min() - 1, z[:, 1].max() + 1
grid_size = 200
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_points = np.vstack([xx.ravel(), yy.ravel()])

# Compute aggregated posterior: E_{x~X}[q_φ(z|x)]
# This is the average of all individual posteriors
density = np.zeros(grid_points.shape[1])

# For computational efficiency, we'll use KDE as an approximation
# The true aggregated posterior would require evaluating each Gaussian and averaging
print("Using KDE approximation for aggregated posterior...")
kde = gaussian_kde(z.T)
density = kde(grid_points)
density = density.reshape(xx.shape)

# -------------------------
# Visualization
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Aggregated posterior density (heatmap)
im = axes[0].imshow(
    density,
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    cmap='hot',
    aspect='auto'
)
axes[0].set_xlabel('z₁', fontsize=12)
axes[0].set_ylabel('z₂', fontsize=12)
axes[0].set_title('Aggregated Posterior Density\nE_{x~X}[q_φ(z|x)]', fontsize=14)
plt.colorbar(im, ax=axes[0])

# Right: Scatter plot of posterior means
axes[1].scatter(z[:, 0], z[:, 1], s=5, alpha=0.5, c='#1f77b4')
axes[1].set_xlabel('z₁', fontsize=12)
axes[1].set_ylabel('z₂', fontsize=12)
axes[1].set_title('Posterior Centers\n{E_{q_φ(z|x_n)}[z]}ᴺₙ₌₁', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

# Match axis limits
for ax in axes:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('cvae_latents/latent_visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to latent_visualization.png")
plt.show()

# -------------------------
# Optional: More detailed aggregated posterior
# -------------------------
print("\nCreating detailed aggregated posterior by averaging individual Gaussians...")

# Sample a subset for computational efficiency
n_samples = min(5000, len(z))
indices = np.random.choice(len(z), n_samples, replace=False)
z_subset = z[indices]
logvars_subset = logvars[indices]

if latent_dim > 2:
    # Transform logvars using PCA
    # This is approximate - true covariance transformation would be more complex
    stds_subset = np.exp(0.5 * logvars_subset[:, :2])  # Just use first 2 dims as approximation
else:
    stds_subset = np.exp(0.5 * logvars_subset)

# Compute true aggregated posterior
density_true = np.zeros(grid_points.shape[1])
for i in range(n_samples):
    mu_i = z_subset[i]
    std_i = stds_subset[i]
    
    # Evaluate 2D Gaussian
    diff = grid_points.T - mu_i
    cov_inv = np.diag(1.0 / (std_i ** 2))
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    normalizer = 1.0 / (2 * np.pi * np.prod(std_i))
    density_true += normalizer * np.exp(exponent)

density_true /= n_samples
density_true = density_true.reshape(xx.shape)

# Create single heatmap matching paper style
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
im = ax.imshow(
    density_true,
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    cmap='hot',
    aspect='auto'
)
ax.set_xlabel('z₁', fontsize=12)
ax.set_ylabel('z₂', fontsize=12)
ax.set_title('Aggregated Posterior (True)\nE_{x~X}[q_φ(z|x)]', fontsize=14)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('cvae_latents/latent_density_true.png', dpi=150, bbox_inches='tight')
print("Saved true density visualization to latent_density_true.png")
plt.show()