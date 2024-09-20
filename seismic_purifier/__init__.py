from .representation_learning_models import (
    RepresentationLearningAutoencoder, 
    RepresentationLearningDenoisingAutoencoder, 
    RepresentationLearningAutoencoderEnsemble
)

from .classifier_models import (
    ClassifierAutocovariance, 
    ClassifierAugmentationCrossCovariances, 
    ClassifierRepresentationCrossCovariances
)

from .config import (
    BATCH_SIZE,
    SAMPLING_FREQ,
    INSTANCE_TIME_WINDOW,
    STEAD_TIME_WINDOW,
    SUBSAMPLING_FACTOR,
    TRAIN_VALIDATION_SPLIT,
    KFOLD_SPLITS,
    DATASET_CHUNKS,
    PHASE_PICK_ENSURED_CROP_RATIO,
    PHASE_ENSURING_MARGIN,
    FREQMIN,
    FREQMAX,
    N_CHANNELS
)

from .cubic_interpolation import diff, cubic_interp1d

from .utils import demean, l2_distance, l2_normalize

from .layers import (
    AddNoise,
    NormalizeStd,
    Padding,
    Conv,
    Upsample,
    UpsampleNoactivation,
    Downsample,
    ResIdentity,
    CrossCovarianceCircular
)

__all__ = [
    # Representation Learning Models
    'RepresentationLearningAutoencoder',
    'RepresentationLearningDenoisingAutoencoder',
    'RepresentationLearningAutoencoderEnsemble',
    
    # Classifier Models
    'ClassifierAutocovariance',
    'ClassifierAugmentationCrossCovariances',
    'ClassifierRepresentationCrossCovariances',
    
    # Configuration Constants
    'BATCH_SIZE',
    'SAMPLING_FREQ',
    'INSTANCE_TIME_WINDOW',
    'STEAD_TIME_WINDOW',
    'SUBSAMPLING_FACTOR',
    'TRAIN_VALIDATION_SPLIT',
    'KFOLD_SPLITS',
    'DATASET_CHUNKS',
    'PHASE_PICK_ENSURED_CROP_RATIO',
    'PHASE_ENSURING_MARGIN',
    'FREQMIN',
    'FREQMAX',
    'N_CHANNELS',
    
    # Version Information
    '__version__'
]

__version__ = '0.1.0'
