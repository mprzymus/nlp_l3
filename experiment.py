import logging
import os
import time
import typing
import warnings
from contextlib import redirect_stderr
from pathlib import Path
from statistics import mean, stdev

import fasttext
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import PROBLEM_TEST, PROBLEM_TRAIN
from data import HatefulTweets, WordDataset
from nn import CNNModel, train_model


def run_cnn_test(
    datamodule: pl.LightningDataModule,
    sentence_length: int = 32,
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    pl.seed_everything(seed, workers=True)

    model = CNNModel(
        conv_kernels=[3, 4, 5],
        conv_filter=100,
        head_dim=300,
        sentence_length=sentence_length,
        learning_rate=1e-5,
    )

    start = time.perf_counter()
    best_log = train_model(model, datamodule, name=name, epochs=200, verbose=verbose)
    end = time.perf_counter()

    best_log["train_time"] = end - start
    return best_log


def run_repeated(run_test: typing.Callable, seed_start=1, seed_end=11):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    logs: dict[str, list[float]] = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Checkpoint directory.*")

        for seed in range(seed_start, seed_end):
            log = run_test(seed)

            for k, v in log.items():
                logs.setdefault(k, []).append(v)

    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    return {
        key: f"{mean(value):.4f} ± {stdev(value):.4f}" for key, value in logs.items()
    }


def run_repeated_cnn(
    embeddings_model_path: Path,
    train_path: Path = PROBLEM_TRAIN,
    test_path: Path = PROBLEM_TEST,
    sentence_length: int = 32,
    name: str = "cnn",
    verbose: bool = False,
):
    with open(os.devnull, "w") as null:
        with redirect_stderr(null):
            model = fasttext.load_model(str(embeddings_model_path))

    train_dataset = WordDataset(train_path, model.get_word_vector, sentence_length)
    test_dataset = WordDataset(test_path, model.get_word_vector, sentence_length)

    datamodule = HatefulTweets(train_dataset, test_dataset, 128)

    return run_repeated(
        lambda seed: run_cnn_test(
            datamodule=datamodule,
            sentence_length=sentence_length,
            name=f"{name}_{seed}",
            seed=seed,
            verbose=verbose,
        )
    )


def test_inference_time(model: nn.Module, data_loader: DataLoader) -> str:
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(100):
            for x, _ in data_loader:
                x = x.cuda()

                start = time.perf_counter()
                model(x)
                end = time.perf_counter()

                times.append(end - start)

    return f"{mean(times):.4f} ± {stdev(times):.4f}"


def calculate_memory_usage(model: nn.Module) -> str:
    mem_params = sum(p.nelement() * p.element_size() for p in model.parameters())
    mem_buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    mem = mem_params + mem_buffers

    return f"{mem / (1024**2):.3f} MB"
