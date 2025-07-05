from recovar import (
    RepresentationLearningMultipleAutoencoder,
    ClassifierMultipleAutoencoder
)
from direct_tester import DirectTester
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

TEST_DATA_PATH=''
MODEL_PATH=''

tester = DirectTester()
model_class = RepresentationLearningMultipleAutoencoder

results = tester.test(
    representation_model_class=model_class,
    classifier_wrapper_class=ClassifierMultipleAutoencoder,
    test_dataset_path="/home/ege/recovar/reproducibility/preprocessed_data/stead_splits/FULL_DATASET_SUBSAMPLED_1_test.hdf5",
    model_weights_path="/home/ege/recovar/reproducibility/checkpoints/FULL_DATASET_SUBSAMPLED_1_train_best_model.h5",
    batch_size=256,
    use_hdf5_generator=True  
)

scores, labels = results
print(f"Got {len(scores)} predictions")

def plot_roc_curve(scores, labels, title="ROC Curve"):
    """
    Plot ROC curve and calculate AUC
    
    Args:
        scores: Predicted probabilities/scores
        labels: True binary labels (0/1)
        title: Plot title
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"AUC Score: {roc_auc:.4f}")
    return roc_auc

scores, labels = results 
plot_roc_curve(scores, labels, "Earthquake Detection ROC Curve")