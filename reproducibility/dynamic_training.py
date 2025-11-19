from kfold_dynamic_trainer import KfoldDynamicTrainer
from recovar import RepresentationLearningMultipleAutoencoder

trainer = KfoldDynamicTrainer(
    exp_name="SLVT_dynamic_v1",
    model_class=RepresentationLearningMultipleAutoencoder,
    dataset="SLVT_fixed",
    split=0,
    epochs=10,
    batch_multiplier=10,
    keep_top_batches=1, 
)

trainer.train()
