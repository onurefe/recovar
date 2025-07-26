import numpy as np
import matplotlib.pyplot as plt
from recovar import (RepresentationLearningSingleAutoencoder, 
                     RepresentationLearningDenoisingSingleAutoencoder, 
                     RepresentationLearningMultipleAutoencoder)

from directory import get_checkpoint_path
from config import BATCH_SIZE, N_CHANNELS
from kfold_environment import KFoldEnvironment
from matplotlib.gridspec import GridSpec
from itertools import combinations

# -------------------------------
# Configuration and Setup
# -------------------------------

# Experiment name identifier
EXP_NAME = "exp_test"

# Choose the representation learning model class
REPRESENTATION_LEARNING_MODEL_CLASS = RepresentationLearningMultipleAutoencoder

# Specify training and testing datasets ('stead' or 'instance')
TRAIN_DATASET = "instance"
TEST_DATASET = "custom"

# Number of training epochs
EPOCH = 6

# Data split identifier
SPLIT = 0

# Num of samples to plot.
NUM_SAMPLES = 25000

def compute_covariance(data_arrays):
    """
    Computes autocovariance or cross-covariance between signals.
    
    Args:
        data_arrays (np.ndarray): Variable number of 2D arrays, each with shape (timesteps, channels)
    
    Returns:
        lags (np.ndarray): Lag values
        avg_cov (np.ndarray): Averaged covariance
    """
    if (len(np.shape(data_arrays)) == 2):
        data_arrays = np.expand_dims(data_arrays,axis=0)
    
    num_signals = len(data_arrays)    
    num_timesteps, num_channels = data_arrays[0].shape

    covariances = []
    lags = np.arange(-num_timesteps + 1, num_timesteps)

    if num_signals == 1:
        # Autocovariance
        data = data_arrays[0]
        for c in range(num_channels):
            channel_data = data[:, c]
            channel_data = channel_data - np.mean(channel_data)  # Zero-mean
            cov = np.correlate(channel_data, channel_data, mode='full')
            covariances.append(cov)
    else:
        # Cross-covariance between all possible pairs
        pairs = list(combinations(range(num_signals), 2))
        for idx1, idx2 in pairs:
            data1 = data_arrays[idx1]
            data2 = data_arrays[idx2]
            for c in range(num_channels):
                channel_data1 = data1[:, c]
                channel_data2 = data2[:, c]
                channel_data1 = channel_data1 - np.mean(channel_data1)  # Zero-mean
                channel_data2 = channel_data2 - np.mean(channel_data2)  # Zero-mean
                cov = np.correlate(channel_data1, channel_data2, mode='full')
                covariances.append(cov)

    covariances = np.array(covariances)
    avg_cov = np.mean(covariances, axis=0)
    return lags, avg_cov

def plot_waveform_channel(ax, timesteps, waveform, channel_idx, ylim_min=None, ylim_max=None, color='blue', show_xticks=True):
    """
    Plots a single waveform channel on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        timesteps (np.ndarray): Array of timesteps
        waveform (np.ndarray): Waveform data for one channel
        channel_idx (int): Channel index (0-based)
        color (str): Color for the plot
        show_xticks (bool): Whether to show x-axis tick labels and label
    """
    channels =['E', 'N', 'Z']
    ax.plot(timesteps, waveform, color=color, linewidth=1)
    if channel_idx == 0:
        ax.set_title("Waveform", fontsize=14, pad=5, fontweight='bold')
        
    if ylim_min != None and ylim_max != None:
        ax.set_ylim(ymin=ylim_min, ymax=ylim_max)
    
    ax.tick_params(axis='y', labelsize=10)
    if show_xticks:
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax.grid(True)

def plot_heatmap(ax, heatmap, vmin=None, vmax=None, title=None):
    """
    Plots the heatmap on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        heatmap (np.ndarray): Shape (94, 64)
    """
    if vmin != None and vmax != None:
        cax = ax.imshow(heatmap, aspect='auto', cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
    else:
        cax = ax.imshow(heatmap, aspect='auto', cmap='magma', origin='lower')
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

def plot_autocovariance(ax, lags, autocov,  ylim_min=None, ylim_max=None, title=None, ylabel='Autocovariance'):
    """
    Plots the autocovariance function on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        lags (np.ndarray): Lag values
        autocov (np.ndarray): Autocovariance values
        title (str): Title of the plot
    """
    if ylim_min != None and ylim_max != None:
        ax.set_ylim(ymin=ylim_min, ymax=ylim_max)
        
    ax.plot(lags, autocov)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

def load_model():
    # Initialize the representation learning model
    model = REPRESENTATION_LEARNING_MODEL_CLASS()
    model.compile()

    # Perform a forward pass with random input to initialize model weights
    model(np.random.normal(size=[BATCH_SIZE, 3000, N_CHANNELS]))

    # Construct the checkpoint path for the model weights
    cp_path = get_checkpoint_path(
        EXP_NAME,
        REPRESENTATION_LEARNING_MODEL_CLASS().name,
        TRAIN_DATASET,
        SPLIT,
        EPOCH
    )

    # Load the pre-trained weights into the model
    model.load_weights(cp_path)
    return model

def load_sample_data():
    """
    Generates sample waveform and heatmap data.
    
    Returns:
        waveform (np.ndarray): Shape (NUM_SAMPLES, 3000, 3)
        labels (np.ndarray): Shape (NUM_SAMPLES)
        metadata (dataframe)
    """
    # Create a K-Fold environment for the specified test dataset
    kenv = KFoldEnvironment(TEST_DATASET)

    # Retrieve metadata for training, validation, and testing splits
    __, __, test_metadata = kenv.get_split_metadata(SPLIT)

    # Retrieve data generators for training, validation, and testing
    __, __, test_gen, __ = kenv.get_generators(SPLIT)
    
    # -------------------------------
    # Data Preparation
    # -------------------------------

    # Initialize lists to hold batches of data
    X = []
    Y = []

    num_batches = 1 + (NUM_SAMPLES // BATCH_SIZE)
    
    # Iterate over the test generator.
    for i in range(num_batches):
        x_batch, y_batch = test_gen.__getitem__(i)
        X.append(x_batch)
        Y.append(y_batch)

    # Concatenate all batches into single numpy arrays
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    
    return X, Y, test_metadata

# Generate sample data
waveforms, labels, metadata = load_sample_data()
model = load_model()

if REPRESENTATION_LEARNING_MODEL_CLASS == RepresentationLearningMultipleAutoencoder:
    model_out = model(waveforms)
    feature_maps = list(model_out)[0:5]
    feature_maps = np.array(feature_maps)
else:
    model_out = model(waveforms)
    feature_maps = list(model_out)[0:1]
    feature_maps = np.array(feature_maps)
    
feature_maps = np.transpose(feature_maps, axes=[1, 0, 2, 3])

WAVEFORM_COLORS = ['blue', 'green', 'red']  # Adjust based on actual channels

# Separate earthquake and noise indices
earthquake_indices = [i for i, label in enumerate(labels) if label > 0.5]
noise_indices = [i for i, label in enumerate(labels) if label <= 0.5]

# Ensure equal number of earthquake and noise samples
NUM_PLOTS = min(len(earthquake_indices), 
                len(noise_indices), 
                NUM_SAMPLES)

for plot_idx in range(NUM_PLOTS):
    eq_idx = earthquake_indices[plot_idx]
    noise_idx = noise_indices[plot_idx]
    
    # Extract earthquake data
    eq_waveform = waveforms[eq_idx]
    eq_feature_map = feature_maps[eq_idx]
    lags_waveform_eq, autocov_waveform_eq = compute_covariance(eq_waveform)  # Averaging over channels
    lags_heatmap_eq, autocov_heatmap_eq = compute_covariance(eq_feature_map)
    
    # Extract noise data
    noise_waveform = waveforms[noise_idx]
    noise_feature_map = feature_maps[noise_idx]
    lags_waveform_noise, autocov_waveform_noise = compute_covariance(noise_waveform)  # Averaging over channels
    lags_heatmap_noise, autocov_heatmap_noise = compute_covariance(noise_feature_map)
    
    # Create a figure with a 1x2 grid: left for earthquake, right for noise
    fig = plt.figure(figsize=(20, 10))  # Adjust size as needed
    main_gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # --- Earthquake Column ---
    eq_gs = main_gs[0, 0].subgridspec(2, 2, wspace=0.3, hspace=0.4)
    
    # Top-Left: Waveform Channels
    eq_waveform_gs = eq_gs[0, 0].subgridspec(eq_waveform.shape[1], 1, hspace=0.3)
    timesteps_eq = np.arange(eq_waveform.shape[0])
    
    for channel in range(eq_waveform.shape[1]):
        ax = fig.add_subplot(eq_waveform_gs[channel, 0])
        show_xticks = (channel == eq_waveform.shape[1] - 1)
        plot_waveform_channel(ax, timesteps_eq, eq_waveform[:, channel], channel, 
                              color=WAVEFORM_COLORS[channel % len(WAVEFORM_COLORS)], 
                              show_xticks=show_xticks)
    
    # Top-Right: Heatmap
    ax_heatmap_eq = fig.add_subplot(eq_gs[0, 1])
    plot_heatmap(ax_heatmap_eq, eq_feature_map[0].T, title="Latent Representation")
    
    # Bottom-Left: Autocovariance of Waveform
    ax_autocov_waveform_eq = fig.add_subplot(eq_gs[1, 0])
    plot_autocovariance(ax_autocov_waveform_eq, lags_waveform_eq, autocov_waveform_eq, 
                        title='Waveform\nAutocovariance function')
    
    if REPRESENTATION_LEARNING_MODEL_CLASS == RepresentationLearningMultipleAutoencoder:
        latent_covariance_title = 'Latent Representation\nCross-covariance function'
        latent_covariance_ylabel = 'Mean Cross-covariance'
    else:
        latent_covariance_title = 'Latent Representation\nAuto-covariance function'
        latent_covariance_ylabel = 'Auto-covariance'
        
    # Bottom-Right: Autocovariance of Heatmap
    ax_autocov_heatmap_eq = fig.add_subplot(eq_gs[1, 1])
    plot_autocovariance(ax_autocov_heatmap_eq, 
                        lags_heatmap_eq, 
                        autocov_heatmap_eq, 
                        title=latent_covariance_title,
                        ylabel=latent_covariance_ylabel)
    
    # --- Noise Column ---
    feature_map_max = np.max(eq_feature_map[0], axis=(0, 1))
    feature_map_min = np.min(eq_feature_map[0], axis=(0, 1))
    
    autocov_heatmap_max = np.max(autocov_heatmap_eq, axis=(0))
    autocov_heatmap_min = np.min(autocov_heatmap_eq, axis=(0))
    
    noise_gs = main_gs[0, 1].subgridspec(2, 2, wspace=0.3, hspace=0.4)
    
    # Top-Left: Waveform Channels
    noise_waveform_gs = noise_gs[0, 0].subgridspec(noise_waveform.shape[1], 1, hspace=0.3)
    timesteps_noise = np.arange(noise_waveform.shape[0])
    
    for channel in range(noise_waveform.shape[1]):
        ax = fig.add_subplot(noise_waveform_gs[channel, 0])
        show_xticks = (channel == noise_waveform.shape[1] - 1)
        plot_waveform_channel(ax, timesteps_noise, noise_waveform[:, channel], channel, 
                              color=WAVEFORM_COLORS[channel % len(WAVEFORM_COLORS)], 
                              show_xticks=show_xticks)
    
    # Top-Right: Heatmap
    ax_heatmap_noise = fig.add_subplot(noise_gs[0, 1])
    plot_heatmap(ax_heatmap_noise, noise_feature_map[0].T, feature_map_min, feature_map_max, "Latent Representation")
    
    # Bottom-Left: Autocovariance of Waveform
    ax_autocov_waveform_noise = fig.add_subplot(noise_gs[1, 0])
    plot_autocovariance(ax_autocov_waveform_noise, 
                        lags_waveform_noise, 
                        autocov_waveform_noise, 
                        title='Waveform\nAutocovariance function')
    
    # Bottom-Right: Autocovariance of Heatmap
    ax_autocov_heatmap_noise = fig.add_subplot(noise_gs[1, 1])
    plot_autocovariance(ax_autocov_heatmap_noise, 
                        lags_heatmap_noise, 
                        autocov_heatmap_noise,
                        autocov_heatmap_min,
                        autocov_heatmap_max,
                        latent_covariance_title,
                        latent_covariance_ylabel)
    
    # --- Add Column Titles ---
    # Positioning the titles above the respective columns
    # Adjust the y-coordinate (0.95) if necessary based on your figure's layout
    fig.text(0.30, 0.935, 'Earthquake sample', ha='center', va='center', fontsize=20, fontweight='bold')
    fig.text(0.725, 0.935, 'Noise sample', ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Adjust overall layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 0.03, 0.75])  # Adjust rect to accommodate the main title
    plt.savefig(f"latent_plot_pair_{plot_idx + 1}.png")
    plt.close(fig)  # Close the figure to free memory