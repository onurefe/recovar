from training_models import AutoencoderEnsemble as RCCTraining
from monitor_models import RepresentationCrossCovariances as RCCMonitoring
from kfold_tester import KFoldTester

tester = KFoldTester(
    "exp_test",
    RCCTraining,
    RCCMonitoring,
    "stead",
    "stead",
    0,
    [8],
    ["x", "fcov", "f1", "f2"],
)
tester.test()
