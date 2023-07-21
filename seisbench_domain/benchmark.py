import seisbench.data as sbd
import seisbench.models as sbm
import seisbench.generate as sbg
from seisbench.util import worker_seeding
import pandas as pd
from torch.utils.data import DataLoader

from os.path import join
import torch
import numpy as np
from config import SAMPLING_FREQ, TIME_WINDOW, BATCH_SIZE, N_TIMESTEPS

BATCH_SIZE = 250
NUM_WORKERS = 4

EVENTS_DSET_DIR = "/home/onur/instance/events"
NOISE_DSET_DIR = "/home/onur/instance/noise"

EQT_COMPONENT_ORDER = "ZNE"
EQT_DIMENSION_ORDER = "NCW"
PHASENET_COMPONENT_ORDER = "ENZ"
PHASENET_DIMENSION_ORDER = "NCW"

TRACES_CSV_PATH = "traces/instance/split1/traces.csv"

PHASENET_PROB_CSV_PATH = "./benchmarks/phasenet_prob.csv"
EQT_PROB_CSV_PATH = "./benchmarks/eqt_prob.csv"


def filter_dataset(dataset, df_traces):
    mask = dataset.metadata["trace_name"].isin(df_traces["trace_name"])
    dataset.filter(mask)
    return dataset


def get_test_dataset(dset_dir, target_component_order, target_dimension_order):
    dataset = sbd.WaveformDataset(
        dset_dir,
        sampling_rate=SAMPLING_FREQ,
        component_order=target_component_order,
        dimension_order=target_dimension_order,
    )

    __, __, test_dataset = dataset.train_dev_test()
    return test_dataset


def get_data_loader(dataset, window_size):
    aug_crop_window = sbg.RandomWindow(0, window_size, window_size)
    aug_channelwise_normalize = sbg.Normalize(
        demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"
    )
    aug_change_dtype = sbg.ChangeDtype(dtype=np.float32)

    gen = sbg.GenericGenerator(dataset)
    gen.augmentation(aug_crop_window)
    gen.augmentation(aug_channelwise_normalize)
    gen.augmentation(aug_change_dtype)

    loader = DataLoader(
        gen,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_seeding,
    )

    return loader


def phasenet_stitch(d1, d2, d3):
    d1 = np.pad(d1, ((0, 0), (0, N_TIMESTEPS // 2 - 1)))
    d2 = np.pad(d2, ((0, 0), (N_TIMESTEPS // 4, N_TIMESTEPS // 4 - 1)))
    d3 = np.pad(d3, ((0, 0), (N_TIMESTEPS // 2 - 1, 0)))

    d1_placeholder = (np.arange(0, N_TIMESTEPS) <= N_TIMESTEPS // 2).astype(int)
    d2_placeholder = (
        (N_TIMESTEPS // 4 <= np.arange(0, N_TIMESTEPS))
        & (np.arange(0, N_TIMESTEPS) <= 3 * N_TIMESTEPS // 4)
    ).astype(int)
    d3_placeholder = (N_TIMESTEPS // 2 - 1 <= np.arange(0, N_TIMESTEPS)).astype(int)

    weight = 1.0 / (d1_placeholder + d2_placeholder + d3_placeholder)

    return weight * (d1 + d2 + d3)


def eqt_classify(model, test_dataset, df):
    test_dataset = filter_dataset(test_dataset, df)
    test_loader = get_data_loader(test_dataset, N_TIMESTEPS)

    detection_probs = []
    model.eval()
    for batchidx, batch in enumerate(test_loader):
        x = batch["X"].to(model.device)
        with torch.no_grad():
            __, __, d = model(x)
            detection_probs.append(np.max(np.array(d), axis=1))
            print("Batch: {}".format(batchidx))

    df = test_dataset.metadata
    df["detection_probability"] = np.concatenate(detection_probs, axis=0)
    return df


def phasenet_classify(model, test_dataset, df):
    test_dataset = filter_dataset(test_dataset, df)
    test_loader = get_data_loader(test_dataset, N_TIMESTEPS)

    detection_probs = []
    model.eval()
    for batchidx, batch in enumerate(test_loader):
        x = batch["X"].to(model.device)
        x1 = x[:, :, 0 : N_TIMESTEPS // 2 + 1]
        x2 = x[:, :, N_TIMESTEPS // 4 : 3 * N_TIMESTEPS // 4 + 1]
        x3 = x[:, :, N_TIMESTEPS // 2 - 1 : N_TIMESTEPS]

        with torch.no_grad():
            n1 = model(x1)[:, 0, :]
            n2 = model(x2)[:, 0, :]
            n3 = model(x3)[:, 0, :]

            n = phasenet_stitch(n1, n2, n3)

            detection_probs.append(np.max((1.0 - n), axis=1))
            print("Batch: {}".format(batchidx))

    df = test_dataset.metadata
    df["detection_probability"] = np.concatenate(detection_probs, axis=0)
    return df


def get_df_probabilities(df_model, df_traces):
    results = []
    for rowidx, row in df_traces.iterrows():
        row_df = df_traces.iloc[[rowidx]]
        row_df["detection_probability"] = df_model[
            df_model["trace_name"] == row_df["trace_name"].values[0]
        ]["detection_probability"].values[0]
        results.append(row_df)

    return pd.concat(results, ignore_index=True)


eqt = sbm.EQTransformer.from_pretrained("original", update=True)
phasenet = sbm.PhaseNet.from_pretrained("original")

df_traces = pd.read_csv(TRACES_CSV_PATH)
df_pick_traces = df_traces[df_traces.label == "pick"]
df_noise_traces = df_traces[df_traces.label == "noise"]

phasenet_events_test_dataset = get_test_dataset(
    EVENTS_DSET_DIR, PHASENET_COMPONENT_ORDER, PHASENET_DIMENSION_ORDER
)
phasenet_noise_test_dataset = get_test_dataset(
    NOISE_DSET_DIR, PHASENET_COMPONENT_ORDER, PHASENET_DIMENSION_ORDER
)

eqt_events_test_dataset = get_test_dataset(
    EVENTS_DSET_DIR, EQT_COMPONENT_ORDER, EQT_DIMENSION_ORDER
)
eqt_noise_test_dataset = get_test_dataset(
    NOISE_DSET_DIR, EQT_COMPONENT_ORDER, EQT_DIMENSION_ORDER
)

df_phasenet_events = phasenet_classify(
    phasenet, phasenet_events_test_dataset, df_pick_traces
)

df_phasenet_noise = phasenet_classify(
    phasenet, phasenet_noise_test_dataset, df_noise_traces
)

df_eqt_events = eqt_classify(eqt, eqt_events_test_dataset, df_pick_traces)
df_eqt_noise = eqt_classify(eqt, eqt_noise_test_dataset, df_noise_traces)

df_phasenet = pd.concat([df_phasenet_events, df_phasenet_noise], ignore_index=True)
df_eqt = pd.concat([df_eqt_events, df_eqt_noise], ignore_index=True)

df_phasenet = get_df_probabilities(df_phasenet, df_traces)
df_eqt = get_df_probabilities(df_eqt, df_traces)

df_phasenet.to_csv(PHASENET_PROB_CSV_PATH, index=False)
df_eqt.to_csv(EQT_PROB_CSV_PATH, index=False)
