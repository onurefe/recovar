from .representation_learning_models import (
    RepresentationLearningSingleAutoencoder, 
    RepresentationLearningDenoisingSingleAutoencoder, 
    RepresentationLearningMultipleAutoencoder
)

from .classifier_models import (
    ClassifierAutocovariance, 
    ClassifierAugmentedAutoencoder, 
    ClassifierMultipleAutoencoder
)

from .config import (
    BATCH_SIZE,
    SAMPLING_FREQ
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
    'RepresentationLearningSingleAutoencoder',
    'RepresentationLearningDenoisingSingleAutoencoder',
    'RepresentationLearningMultipleAutoencoder',
    
    # Classifier Models
    'ClassifierAutocovariance',
    'ClassifierAugmentedAutoencoder',
    'ClassifierMultipleAutoencoder',
    
    # Configuration Constants
    'BATCH_SIZE',
    'SAMPLING_FREQ'
    
    # Version Information
    '__version__'
]

__version__ = '0.1.0'
