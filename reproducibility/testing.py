from recovar import (RepresentationLearningSingleAutoencoder, 
                     RepresentationLearningDenoisingSingleAutoencoder, 
                     RepresentationLearningMultipleAutoencoder)
from recovar import ClassifierAutocovariance, ClassifierAugmentedAutoencoder, ClassifierMultipleAutoencoder
from kfold_tester import KFoldTester
from evaluator import Evaluator, CropOffsetFilter, LastEarthquakeFilter
from sklearn.metrics import auc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Should be RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder or RepresentationLearningMultipleAutoencoder
REPRESENTATION_LEARNING_MODEL_CLASS = RepresentationLearningMultipleAutoencoder
CLASSIFIER_MODEL_CLASS = ClassifierMultipleAutoencoder

# Split.
SPLIT = 0

rows = []

def _eval_resamplings(experiment_prefix, df_path):
    for train_set in ["instance"]:
        for test_set in ["instance"]:
            for resample_eq_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                if test_set == "custom": 
                    filters = [CropOffsetFilter(), LastEarthquakeFilter()]
                else:
                    filters = [CropOffsetFilter()]
                    
                evaluator = Evaluator(exp_name = f"{experiment_prefix}{resample_eq_ratio}", 
                                      representation_learning_model_class=REPRESENTATION_LEARNING_MODEL_CLASS, 
                                      classifier_model_class = CLASSIFIER_MODEL_CLASS, 
                                      train_dataset = train_set, 
                                      test_dataset = test_set, 
                                      filters = filters, 
                                      split = SPLIT,
                                      apply_resampling=True,
                                      resample_eq_ratio=resample_eq_ratio,
                                      resample_while_keeping_total_waveforms_fixed=True,
                                      report_best_val_score_epoch=True,
                                      method_params={})

                roc_vectors = evaluator.get_roc_vectors()
                roc_auc = auc(roc_vectors[0]["fpr"], roc_vectors[0]["tpr"])

                rows.append({"train_set": train_set, 
                             "test_set": test_set, 
                             "resample_eq_ratio": resample_eq_ratio,
                             "roc_auc": roc_auc})

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(df_path)
    
def _plot_roc(experiment_prefix, resample_eq_ratio):
    filters = [CropOffsetFilter(), LastEarthquakeFilter()]
    evaluator = Evaluator(exp_name = f"{experiment_prefix}{resample_eq_ratio}", 
                          representation_learning_model_class=REPRESENTATION_LEARNING_MODEL_CLASS, 
                          classifier_model_class = CLASSIFIER_MODEL_CLASS, 
                          train_dataset = "custom", 
                          test_dataset = "custom", 
                          filters = filters, 
                          split = 0,
                          apply_resampling=True,
                          resample_eq_ratio=resample_eq_ratio,
                          report_best_val_score_epoch=True,
                          method_params={})

    roc_vectors = evaluator.get_roc_vectors()
    roc_auc = auc(roc_vectors[0]["fpr"], roc_vectors[0]["tpr"])
    print(roc_auc)

    # Create a DataFrame for Seaborn
    roc_data = pd.DataFrame({
        'False Positive Rate': roc_vectors[0]["fpr"],
        'True Positive Rate': roc_vectors[0]["tpr"]
    })

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=roc_data, x='False Positive Rate', y='True Positive Rate', label='ROC Curve')

    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Seismic Detection Model')
    plt.legend()
    plt.grid(True)
    plt.savefig("tpr-fpr.png")
    
#_plot_roc("exp_resample_eq_ratio", 0.1)
_eval_resamplings("exp_instance_resample_eq_ratio", "/home/onur/Code/recovar/resampling_vs_score.csv")  