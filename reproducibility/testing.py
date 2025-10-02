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

def _eval_cross_testing(train_dataset, test_dataset, df_path):
    rows = []
    for resample_eq_ratio in [0.01,0.02, 0.03, 0.04, 0.05]:
        filters = [CropOffsetFilter()]

        evaluator = Evaluator(exp_name = f"exp_{train_dataset}_resample_eq_ratio_{resample_eq_ratio}",
                              representation_learning_model_class=REPRESENTATION_LEARNING_MODEL_CLASS,
                              classifier_model_class = CLASSIFIER_MODEL_CLASS,
                              train_dataset = train_dataset,
                              test_dataset = test_dataset,
                              filters = filters,
                              split = SPLIT,
                              apply_resampling=True,
                              resample_eq_ratio=resample_eq_ratio,
                              resample_while_keeping_total_waveforms_fixed=True,
                              report_best_val_score_epoch=True,
                              method_params={})

        roc_vectors = evaluator.get_roc_vectors()
        roc_auc = auc(roc_vectors[0]["fpr"], roc_vectors[0]["tpr"])

        rows.append({"train_dataset": train_dataset,
                     "test_dataset": test_dataset,
                     "resample_eq_ratio": resample_eq_ratio,
                     "roc_auc": roc_auc})

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(df_path)
    
def _plot_roc(train_dataset, test_dataset, resample_eq_ratio):
    filters = [CropOffsetFilter()]
    evaluator = Evaluator(exp_name = f"exp_{train_dataset}_resample_eq_ratio_{resample_eq_ratio}",
                            representation_learning_model_class=REPRESENTATION_LEARNING_MODEL_CLASS,
                            classifier_model_class = CLASSIFIER_MODEL_CLASS,
                            train_dataset = train_dataset,
                            test_dataset = test_dataset,
                            filters = filters,
                            split = SPLIT,
                            apply_resampling=True,
                            resample_eq_ratio=resample_eq_ratio,
                            resample_while_keeping_total_waveforms_fixed=True,
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
    plt.title(f'Train:{train_dataset} Test:{test_dataset} ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{train_dataset}_on_{test_dataset}_{resample_eq_ratio}_tpr-fpr.png")

_eval_cross_testing("MERGED_fixed", "MERGED_fixed", "/home/ege/recovar/MERGED_test_MERGED_1.csv")
