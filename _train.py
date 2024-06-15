from config import KFOLD_SPLITS
from training_models import Autoencoder, DenoisingAutoencoder, AutoencoderEnsemble
from kfold_trainer import KfoldTrainer

for split in range(4, 5):
    #kfold_trainer = KfoldTrainer(
    #    "exp_test", AutoencoderEnsemble, "instance", split, epochs=20
    #)
    #kfold_trainer.train()

    kfold_trainer = KfoldTrainer(
        "exp_test", AutoencoderEnsemble, "stead", split, epochs=20
    )
    kfold_trainer.train()
