from config import KFOLD_SPLITS
from training_models import Autoencoder, DenoisingAutoencoder, AutoencoderEnsemble
from monitor_models import AugmentationCrossCovariances , StaLta, Autocovariance, AugmentationCrossCovariances, RepresentationCrossCovariances
from kfold_tester import KFoldTester

for split in range(4, 5):
    #kfold_tester = KfoldTester(
    #    "exp_test", AutoencoderEnsemble, "instance", split, epochs=20
    #)
    #kfold_tester.test()

    #Autoencodersensemble
    '''
    kfold_tester = KFoldTester(
        exp_name="exp_test", 
        training_ctor=AutoencoderEnsemble, 
        monitoring_ctor=AugmentationCrossCovariances, 
        train_dataset ='stead', 
        test_dataset="stead", 
        split=split, 
        epochs=[0,5], 
        monitored_params=['fcov'], 
        method_params={"augmentations": 5, "std": 0.15, "knots": 4}
    )
    '''
    kfold_tester = KFoldTester(
        exp_name="exp_test", 
        training_ctor=Autocovariance, 
        monitoring_ctor=StaLta, 
        train_dataset ='stead', 
        test_dataset="stead", 
        split=split, 
        epochs=[0,5], 
        monitored_params=["fcov"], 
        method_params={}
    )
    kfold_tester.test()