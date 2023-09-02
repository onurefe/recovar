import os
import shutil

EPOCHS = range(19, 20)
BASE_DIRS = [
    "results/sta_lta_stw3_ltw6/training_instance/testing_instance/StaLtaMonitor/split0",
    "results/sta_lta_stw3_ltw6/training_instance/testing_instance/StaLtaMonitor/split1",
    "results/sta_lta_stw3_ltw6/training_instance/testing_instance/StaLtaMonitor/split2",
    "results/sta_lta_stw3_ltw6/training_instance/testing_instance/StaLtaMonitor/split3",
    "results/sta_lta_stw3_ltw6/training_instance/testing_instance/StaLtaMonitor/split4",
    "results/sta_lta_stw3_ltw6/training_instance/testing_stead/StaLtaMonitor/split0",
    "results/sta_lta_stw3_ltw6/training_instance/testing_stead/StaLtaMonitor/split1",
    "results/sta_lta_stw3_ltw6/training_instance/testing_stead/StaLtaMonitor/split2",
    "results/sta_lta_stw3_ltw6/training_instance/testing_stead/StaLtaMonitor/split3",
    "results/sta_lta_stw3_ltw6/training_instance/testing_stead/StaLtaMonitor/split4",
    "results/sta_lta_stw3_ltw6/training_stead/testing_instance/StaLtaMonitor/split0",
    "results/sta_lta_stw3_ltw6/training_stead/testing_instance/StaLtaMonitor/split1",
    "results/sta_lta_stw3_ltw6/training_stead/testing_instance/StaLtaMonitor/split2",
    "results/sta_lta_stw3_ltw6/training_stead/testing_instance/StaLtaMonitor/split3",
    "results/sta_lta_stw3_ltw6/training_stead/testing_instance/StaLtaMonitor/split4",
    "results/sta_lta_stw3_ltw6/training_stead/testing_stead/StaLtaMonitor/split0",
    "results/sta_lta_stw3_ltw6/training_stead/testing_stead/StaLtaMonitor/split1",
    "results/sta_lta_stw3_ltw6/training_stead/testing_stead/StaLtaMonitor/split2",
    "results/sta_lta_stw3_ltw6/training_stead/testing_stead/StaLtaMonitor/split3",
    "results/sta_lta_stw3_ltw6/training_stead/testing_stead/StaLtaMonitor/split4",
]


BASE_FILE_NAME = "epoch0_monitoredparams['score'].hdf5"

for base_dir in BASE_DIRS:
    src = os.path.join(base_dir, BASE_FILE_NAME)

    for epoch in EPOCHS:
        dst = os.path.join(
            base_dir, "epoch{}_monitoredparams['score'].hdf5".format(epoch)
        )

        shutil.copyfile(src, dst)
