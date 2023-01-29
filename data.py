import typing as t
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import fasttext
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import PROBLEM_TEST, PROBLEM_TRAIN
import torch.nn.functional as F


warnings.filterwarnings("ignore", ".*does not have many workers.*")


class TextDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        get_text_embedding: t.Callable[[str], torch.Tensor],
    ) -> None:
        super().__init__()
        df = pd.read_csv(csv_file)

        self.embeddings = get_text_embedding(df["text"].to_list())
        self.labels = torch.tensor(df["label"].values)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class WordDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        get_word_embedding: t.Callable[[str], torch.Tensor],
        max_length: int = 32,
    ) -> None:
        super().__init__()
        df = pd.read_csv(csv_file)

        unpadded = [
            torch.stack(
                [
                    torch.from_numpy(get_word_embedding(text))
                    for text in sentence.split(" ")
                ]
            )
            for sentence in df["text"]
        ]
        unpadded[0] = F.pad(unpadded[0], (0, 0, 0, max_length - len(unpadded[0])))

        self.embeddings = pad_sequence(unpadded, batch_first=True)
        self.labels = torch.tensor(df["label"].values)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


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
        train_dataset: TextDataset | WordDataset | RnnDataset,
        test_dataset: TextDataset | WordDataset | RnnDataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)

    def _dataloader(
        self,
        dataset: TextDataset | WordDataset,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )
