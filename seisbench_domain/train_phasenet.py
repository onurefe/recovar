import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
import pandas as pd
from config import SAMPLING_FREQ, BATCH_SIZE
import numpy as np
from os.path import join
import torch
from torch.utils.data import DataLoader

NUM_WORKERS = 4
LEARNING_RATE = 0.01
STEAD_DIR = "/home/onur/stead"
PHASENET_COMPONENT_ORDER = "ENZ"
PHASENET_DIMENSION_ORDER = "NCW"
MODEL_DIR = "./trained_models/stead/phasenet"


def split_dataset(df, splits=[0.8, 0.05, 0.15]):
    df["split"] = "none"

    num_rows = len(df)
    rowrange = np.arange(num_rows)
    np.random.shuffle(rowrange)

    train_rows = rowrange[: int(splits[0] * num_rows)]
    val_rows = rowrange[
        int(splits[0] * num_rows) : int((splits[0] + splits[1]) * num_rows)
    ]
    test_rows = rowrange[int((splits[0] + splits[1]) * num_rows) :]

    train_df = df.iloc[train_rows]
    val_df = df.iloc[val_rows]
    test_df = df.iloc[test_rows]

    train_df["split"] = "train"
    val_df["split"] = "dev"
    test_df["split"] = "test"

    df = pd.concat([train_df, val_df, test_df])

    return df


def get_train_dataset(
    dset_dir, target_component_order, target_dimension_order, splits=[0.8, 0.05, 0.15]
):
    dataset = sbd.WaveformDataset(
        dset_dir,
        sampling_rate=SAMPLING_FREQ,
        component_order=target_component_order,
        dimension_order=target_dimension_order,
    )

    train, validation, test = dataset.train_dev_test()
    return train, validation, test


def loss_fn(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


def train_loop(dataloader):
    size = len(dataloader.dataset)
    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(batch["X"].to(model.device))
        loss = loss_fn(pred, batch["y"].to(model.device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * batch["X"].shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")


# Split dataset to training, validation and test sets.
df = pd.read_csv(join(STEAD_DIR, "metadata.csv"))

df = split_dataset(df)
df.to_csv(join(STEAD_DIR, "metadata.csv"), index=False)

model = sbm.PhaseNet(phases="PSN", norm="peak")

train, dev, test = get_train_dataset(
    STEAD_DIR, PHASENET_COMPONENT_ORDER, PHASENET_DIMENSION_ORDER
)

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}

train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)

augmentations = [
    sbg.WindowAroundSample(
        list(phase_dict.keys()),
        samples_before=3000,
        windowlen=6000,
        selection="random",
        strategy="variable",
    ),
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)

train_loader = DataLoader(
    train_generator,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=worker_seeding,
)

dev_loader = DataLoader(
    dev_generator,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    worker_init_fn=worker_seeding,
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for t in range(10):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader)
    test_loop(dev_loader)

    model = sbm.PhaseNet(phases="PSN", norm="peak")
    torch.save(model.state_dict(), join(MODEL_DIR, "model_ep{}.pickle".format(t)))

# model.load_state_dict(torch.load(join(MODEL_DIR, "model_ep{}.pickle".format(t))))
