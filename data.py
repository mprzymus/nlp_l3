import os
import typing as t
import warnings
from contextlib import redirect_stderr
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import PROBLEM_TEST, PROBLEM_TRAIN

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class TextDataset(Dataset):
    def __init__(self, csv_file: Path, embeddings_model: Path) -> None:
        super().__init__()
        df = pd.read_csv(csv_file)

        with open(os.devnull, "w") as null:
            with redirect_stderr(null):
                model = fasttext.load_model(str(embeddings_model))

        self.embeddings = torch.tensor(
            np.array([model.get_sentence_vector(text) for text in df["text"]])
        )
        self.labels = torch.tensor(df["label"].values)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class TransformerEmbeddingsDataset(Dataset):
    def __init__(self, csv_file: Path, text_to_embeddings: t.Callable) -> None:
        super().__init__()
        df = pd.read_csv(csv_file)
        ls = []
        for text in df["text"]:
            ls.append(text_to_embeddings(text).cpu())
        self.embeddings = torch.cat(ls)
        self.labels = torch.tensor(df["label"].values)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx].detach(), self.labels[idx]


def sentence_to_tensor(text, model):
    ls = []
    for word in fasttext.tokenize(text):
        ls.append(torch.from_numpy(model.get_word_vector(word)))
    ls = torch.stack(ls)
    return ls


class RnnDataset(Dataset):
    def __init__(self, csv_file: Path, embeddings_model: Path) -> None:
        super().__init__()
        df = pd.read_csv(csv_file)

        with open(os.devnull, "w") as null:
            with redirect_stderr(null):
                model = fasttext.load_model(str(embeddings_model))

        embeddings_list = [sentence_to_tensor(
            text, model) for text in df["text"]]

        self.embeddings = pad_sequence(embeddings_list, batch_first=True)
        self.labels = torch.tensor(df["label"].values)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class HatefulTweets(pl.LightningDataModule):
    def __init__(
        self,
        embeddings_model: Path,
        batch_size: int,
        dataset_cls: t.Type[TextDataset] = TextDataset,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.embeddings_model = embeddings_model
        self.batch_size = batch_size
        self.dataset_cls = dataset_cls
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(PROBLEM_TRAIN, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(PROBLEM_TEST, shuffle=False)

    def _dataloader(self, f: Path, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.dataset_cls(f, self.embeddings_model),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )
