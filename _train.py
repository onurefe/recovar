from config import KFOLD_SPLITS
from training_models import Autoencoder, DenoisingAutoencoder, AutoencoderEnsemble
from kfold_trainer import KfoldTrainer

for split in range(KFOLD_SPLITS):
    kfold_trainer = KfoldTrainer(
        "exp_test", DenoisingAutoencoder, "instance", split, epochs=20
    )
    kfold_trainer.train()

    kfold_trainer = KfoldTrainer(
        "exp_test", DenoisingAutoencoder, "stead", split, epochs=20
    )
    kfold_trainer.train()
