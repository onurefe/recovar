from kfold_dynamic_trainer import KfoldDynamicTrainer
from recovar import RepresentationLearningMultipleAutoencoder

trainer = KfoldDynamicTrainer(
    exp_name="MERGED_dynamic",
    model_class=RepresentationLearningMultipleAutoencoder,
    dataset="MERGED_fixed",
    split=0,
    epochs=20,
    batch_multiplier=10,
    keep_top_batches=2, 
)

trainer.train()
