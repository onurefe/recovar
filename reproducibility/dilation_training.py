from kfold_trainer_dilation import KfoldTrainerWithDilation
from recovar import RepresentationLearningMultipleAutoencoder

trainer = KfoldTrainerWithDilation(
    exp_name="BGKT_dilation_v2",
    model_class=RepresentationLearningMultipleAutoencoder,
    dataset="BGKT_fixed",
    split=0,
    epochs=10,
    final_dilation=0.1,  
    dilation_schedule='linear',
    score_every_n_epochs=1,  
    warmup_epochs=3, 
    apply_resampling=False,
    learning_rate=1e-4
)

trainer.train()