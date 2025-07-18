from recovar import (RepresentationLearningSingleAutoencoder, 
                     RepresentationLearningDenoisingSingleAutoencoder, 
                     RepresentationLearningMultipleAutoencoder,
                     RepresentationLearningMultipleAutoencoderL4)
from recovar import ClassifierAutocovariance, ClassifierAugmentedAutoencoder, ClassifierMultipleAutoencoder
from kfold_tester import KFoldTester
from evaluator import Evaluator, CropOffsetFilter
from sklearn.metrics import auc
import pandas as pd

# Should be RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder or RepresentationLearningMultipleAutoencoder
REPRESENTATION_LEARNING_MODEL_CLASS = RepresentationLearningMultipleAutoencoderL4
CLASSIFIER_MODEL_CLASS = ClassifierMultipleAutoencoder

# Split.
SPLIT = 0

rows = []

for train_set in ["stead", "instance"]:
    for test_set in ["stead", "instance"]:
        for resample_eq_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            filters = [CropOffsetFilter()]
            evaluator = Evaluator(exp_name = f"expl4_resample_eq_ratio{resample_eq_ratio}", 
                                  representation_learning_model_class=REPRESENTATION_LEARNING_MODEL_CLASS, 
                                  classifier_model_class = CLASSIFIER_MODEL_CLASS, 
                                  train_dataset = train_set, 
                                  test_dataset = test_set, 
                                  filters = filters, 
                                  split = SPLIT,
                                  apply_resampling=True,
                                  resample_eq_ratio=resample_eq_ratio,
                                  report_best_val_score_epoch=True,
                                  method_params={})

            roc_vectors = evaluator.get_roc_vectors()
            fpr = roc_vectors[0]["fpr"]
            tpr = roc_vectors[0]["tpr"]
            roc_auc = auc(roc_vectors[0]["fpr"], roc_vectors[0]["tpr"])

            rows.append({"train_set": train_set, 
                         "test_set": test_set, 
                         "resample_eq_ratio": resample_eq_ratio,
                         "roc_auc": roc_auc})

scores_df = pd.DataFrame(rows)
scores_df.to_csv("eq_no_ratio_auc_scores.csv")