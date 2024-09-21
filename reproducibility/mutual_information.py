from seismic_purifier import RepresentationLearningAutoencoder
from directory import get_checkpoint_path
from config import BATCH_SIZE, N_CHANNELS
from kfold_environment import KFoldEnvironment
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# Experiment name.
EXP_NAME = "exp_test"

REPRESENTATION_LEARNING_MODEL_CLASS = RepresentationLearningAutoencoder

# Should be stead or instance.
TRAIN_DATASET = "stead"
TEST_DATASET = "stead"

EPOCH = 6

# Split.
SPLIT = 0

kenv = KFoldEnvironment(TEST_DATASET)

train_metadata, validation_metadata, test_metadata = kenv.get_split_metadata(SPLIT)
train_gen, validation_gen, test_gen, __ = kenv.get_generators(SPLIT)

model = RepresentationLearningAutoencoder()
model.compile()
model(np.random.normal(size=[BATCH_SIZE, 3000, N_CHANNELS]))

cp_path = get_checkpoint_path(EXP_NAME, 
                                   RepresentationLearningAutoencoder().name,
                                   TRAIN_DATASET,
                                   SPLIT,
                                   EPOCH)

model.load_weights(cp_path)

X = []
Y = []

for i in range(12):
    x_batch, y_batch = test_gen.__getitem__(i)
    X.append(x_batch)
    Y.append(y_batch)

X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)

# Create boolean masks
mask_eq = Y > 0.5
mask_no = Y <= 0.5

# Separate the datasets
X_eq = X[mask_eq]
X_no = X[mask_no]

X_eq = X_eq[0:256]
X_no = X_no[0:256]

F_eq, __ = model(X_eq)
F_no, __ = model(X_no)

def estimate_mutual_information(x, y):
    x = np.expand_dims(x, axis=-1)
    mi = mutual_info_regression(x, y)
    mi = np.squeeze(mi)
    return mi

def estimate_pairwise_mutual_informations(y, channels, timesteps):
    mi = np.zeros(shape=[channels, timesteps, timesteps])

    for c in range(channels):
        for i in range(timesteps):
            for j in range(timesteps):
                if i < j:
                    mi[c, i, j] = estimate_mutual_information(y[:, i, c], y[:, j, c])

    mi = mi + np.transpose(mi, axes=[0, 2, 1])
    return mi

def calculate_pairwise_cross_correlations(y):
    y = y - np.mean(y, axis=0, keepdims=True)
    y = y / np.std(y, axis=0)
    cov = np.einsum("bni, bmi->inm", y, y) / np.shape(y)[0]
    return cov

def calculate_pairwise_cross_covariances(y):
    y = y - np.mean(y, axis=0, keepdims=True)
    cov = np.einsum("bni, bmi->inm", y, y) / np.shape(y)[0]
    return cov

# eq_mutual_informations = estimate_pairwise_mutual_informations(F_eq, 64, 94)
# no_mutual_informations = estimate_pairwise_mutual_informations(F_no, 64, 94)
# np.save("mi_eq.npy", eq_mutual_informations)
# np.save("mi_no.npy", no_mutual_informations)
cov_eq = calculate_pairwise_cross_covariances(F_eq)
cov_no = calculate_pairwise_cross_covariances(F_no)

corr_eq = calculate_pairwise_cross_correlations(F_eq)
corr_no = calculate_pairwise_cross_correlations(F_no)

mi_eq = np.load("mi_eq.npy")
mi_no = np.load("mi_no.npy")

def plot_mutual_information_tensor(mi_tensor, num_rows, num_columns, path, range_min, range_max):
    """
    Plots a grid of mutual information heatmaps with a single colorbar.

    Parameters:
    - mi_tensor (numpy.ndarray): Tensor of shape [64, 94, 94] containing mutual information data.
    - num_rows (int): Number of rows in the subplot grid.
    - num_columns (int): Number of columns in the subplot grid.
    - path (str): File path to save the resulting plot.

    Returns:
    - None
    """
    # Shuffle the cov_tensor to select random channels
    np.random.seed(0)
    np.random.shuffle(mi_tensor)
    
    # Determine the number of channels to plot based on grid size
    num_channels = num_rows * num_columns
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 8, num_rows * 8))
    fig.suptitle('Mutual Information Heatmaps for Randomly Selected Channels', fontsize=32)
    
    # Iterate over each channel and plot its heatmap
    for i in range(num_channels):
        row = i // num_columns
        col = i % num_columns  # Corrected from num_rows to num_columns
        ax = axes[row, col]
        
        if i < len(mi_tensor):
            sns.heatmap(
                mi_tensor[i],
                ax=ax,
                cmap='magma',
                vmin=range_min,
                vmax=range_max,
                cbar=False,  # Disable individual colorbars
                square=True,
                linewidths=.5,
                linecolor='gray'
            )
            ax.set_title(f'Channel {i+1}', fontsize=20)
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            ax.axis('off')  # Hide unused subplots
    
    # Create a ScalarMappable for the colorbar using the same cmap and norm
    norm = mpl.colors.Normalize(vmin=range_min, vmax=range_max)
    sm = mpl.cm.ScalarMappable(cmap='magma', norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    
    # Position the colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coordinates
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Mutual Information Value', fontsize=28)
    
    # Adjust layout to make room for the colorbar
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # Leave space on the right for colorbar
    
    # Save the figure
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
def plot_cross_covariance_tensor(cov_tensor, num_rows, num_columns, path, range_min, range_max):
    """
    Plots a grid of cross-covariance heatmaps with a single colorbar.

    Parameters:
    - cov_tensor (numpy.ndarray): Tensor of shape [64, 94, 94] containing cross-covariance data.
    - num_rows (int): Number of rows in the subplot grid.
    - num_columns (int): Number of columns in the subplot grid.
    - path (str): File path to save the resulting plot.

    Returns:
    - None
    """
    # Shuffle the cov_tensor to select random channels
    np.random.seed(0)
    np.random.shuffle(cov_tensor)
    
    # Determine the number of channels to plot based on grid size
    num_channels = num_rows * num_columns
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 8, num_rows * 8))
    fig.suptitle('Cross-Covariance Heatmaps for Randomly Selected Channels', fontsize=32)
    
    # Iterate over each channel and plot its heatmap
    for i in range(num_channels):
        row = i // num_columns
        col = i % num_columns  # Corrected from num_rows to num_columns
        ax = axes[row, col]
        
        if i < len(cov_tensor):
            sns.heatmap(
                cov_tensor[i],
                ax=ax,
                cmap='magma',
                vmin=range_min,
                vmax=range_max,
                cbar=False,  # Disable individual colorbars
                square=True,
                linewidths=.5,
                linecolor='gray'
            )
            ax.set_title(f'Channel {i+1}', fontsize=20)
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            ax.axis('off')  # Hide unused subplots
    
    # Create a ScalarMappable for the colorbar using the same cmap and norm
    norm = mpl.colors.Normalize(vmin=range_max, vmax=range_max)
    sm = mpl.cm.ScalarMappable(cmap='magma', norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    
    # Position the colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coordinates
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Cross-Covariance Value', fontsize=28)
    
    # Adjust layout to make room for the colorbar
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # Leave space on the right for colorbar
    
    # Save the figure
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
def plot_cross_correlation_tensor(r_tensor, num_rows, num_columns, path, range_min, range_max):
    """
    Plots a grid of cross-correlation heatmaps with a single colorbar.

    Parameters:
    - r_tensor (numpy.ndarray): Tensor of shape [64, 94, 94] containing cross-correlation data.
    - num_rows (int): Number of rows in the subplot grid.
    - num_columns (int): Number of columns in the subplot grid.
    - path (str): File path to save the resulting plot.

    Returns:
    - None
    """
    # Shuffle the r_tensor to select random channels
    np.random.seed(0)
    np.random.shuffle(r_tensor)
    
    # Determine the number of channels to plot based on grid size
    num_channels = num_rows * num_columns
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 8, num_rows * 8))
    fig.suptitle('Cross-Correlation Heatmaps for Randomly Selected Channels', fontsize=32)
    
    # Iterate over each channel and plot its heatmap
    for i in range(num_channels):
        row = i // num_columns
        col = i % num_columns  # Corrected from num_rows to num_columns
        ax = axes[row, col]
        
        if i < len(r_tensor):
            sns.heatmap(
                r_tensor[i],
                ax=ax,
                cmap='magma',
                vmin=range_min,
                vmax=range_max,
                cbar=False,  # Disable individual colorbars
                square=True,
                linewidths=.5,
                linecolor='gray'
            )
            ax.set_title(f'Channel {i+1}', fontsize=20)
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            ax.axis('off')  # Hide unused subplots
    
    # Create a ScalarMappable for the colorbar using the same cmap and norm
    norm = mpl.colors.Normalize(vmin=range_min, vmax=range_max)
    sm = mpl.cm.ScalarMappable(cmap='magma', norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    
    # Position the colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coordinates
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Cross-Correlation Value', fontsize=28)
    
    # Adjust layout to make room for the colorbar
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # Leave space on the right for colorbar
    
    # Save the figure
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

mi_range = 3 * np.std(np.concatenate([mi_eq, mi_no], axis=0))
cov_range = 3 * np.std(np.concatenate([cov_eq, cov_no], axis=0))

plot_mutual_information_tensor(mi_eq, num_rows=3, num_columns=3, path="eq_mutual_informations.png", range_min=0., range_max=mi_range)
plot_mutual_information_tensor(mi_no, num_rows=3, num_columns=3, path="no_mutual_informations.png", range_min=0., range_max=mi_range)
plot_cross_covariance_tensor(cov_eq, num_rows=3, num_columns=3, path="eq_covariances.png", range_min=-cov_range, range_max=cov_range)
plot_cross_covariance_tensor(cov_no, num_rows=3, num_columns=3, path="no_covariances.png", range_min=-cov_range, range_max=cov_range)
plot_cross_correlation_tensor(corr_eq, num_rows=3, num_columns=3, path="eq_correlations.png", range_min=-1., range_max=1.0)
plot_cross_correlation_tensor(corr_no, num_rows=3, num_columns=3, path="no_correlations.png", range_min=-1., range_max=1.0)