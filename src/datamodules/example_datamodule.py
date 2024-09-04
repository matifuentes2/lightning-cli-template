import torch
import lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

DATASETS_PATH = "src/datasets"


class D1DataModule(L.LightningDataModule):
    def __init__(self,
                 dataset: str,
                 batch_size: int,
                 pin_mem: bool,
                 num_workers: int,
                 training: bool,
                 drop_last: bool,) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.collator = collate_fn
        self.pin_mem = pin_mem
        self.num_workers = num_workers
        self.training = training
        self.drop_last = drop_last

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.tl = make_loader(
                self.dataset,
                self.batch_size,
                self.collator,
                self.pin_mem,
                self.num_workers,
                self.drop_last,
                True,
                self.training,
            )
            self.vl = make_loader(
                self.dataset,
                self.batch_size,
                self.collator,
                self.pin_mem,
                self.num_workers,
                False,
                False,
                self.training,
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        return self.tl

    def val_dataloader(self):
        return self.vl

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass


class GraphTreeDataset(Dataset):
    def __init__(self, adj_matrices, labels, to_triangular=False):
        self.adj_matrices = adj_matrices
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.adj_matrices[idx], self.labels[idx]


def collate_fn(batch):
    adj_matrices, labels = zip(*batch)
    adj_tensor = torch.tensor(adj_matrices, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return adj_tensor, labels_tensor


def make_loader(
    dataset,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    drop_last=True,
    shuffle=True,
    training=True,
    to_triangular=False
):
    train_ds_fn = get_data_filename(dataset, "train_ds")
    test_ds_fn = get_data_filename(dataset, "test_ds")
    train_labels_fn = get_data_filename(
        dataset, "train_lab")
    test_labels_fn = get_data_filename(dataset, "test_lab")

    TRAIN_DATASET = np.load(
        f"{DATASETS_PATH}/{dataset}/{train_ds_fn}")
    TEST_DATASET = np.load(
        f"{DATASETS_PATH}/{dataset}/{test_ds_fn}")
    TRAIN_LABELS = np.load(
        f"{DATASETS_PATH}/{dataset}/{train_labels_fn}")
    TEST_LABELS = np.load(
        f"{DATASETS_PATH}/{dataset}/{test_labels_fn}")

    if training:
        dataset = GraphTreeDataset(TRAIN_DATASET, TRAIN_LABELS, to_triangular)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            pin_memory=pin_mem,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return loader
    else:
        dataset = GraphTreeDataset(TEST_DATASET, TEST_LABELS, to_triangular)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            pin_memory=pin_mem,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return loader


def get_data_filename(dataset_class, data_string):
    files = os.listdir(f"{DATASETS_PATH}/{dataset_class}")
    candidates = np.array(files)[[data_string in file for file in files]]
    if len(candidates) != 1:
        raise FileNotFoundError("Fix dataset naming")
    return candidates[0]
