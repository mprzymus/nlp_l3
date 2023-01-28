import logging
import typing
import warnings
from pathlib import Path
from statistics import mean, stdev
from herbert import calc_text_embedding

import pytorch_lightning as pl

from data import HatefulTweets, TransformerEmbeddingsDataset
from nn import BinaryMLP, train_model


def run_transformer_test(
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    pl.seed_everything(seed, workers=True)

    datamodule = HatefulTweets(
        calc_text_embedding, 128, dataset_cls=TransformerEmbeddingsDataset
    )
    model = BinaryMLP(768, [512, 256, 128, 64], learning_rate=1e-4)

    best_log = train_model(model, datamodule, name=name, epochs=3, verbose=verbose)
    return best_log


def run_fasttext_test(
    model_file: Path,
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    pl.seed_everything(seed, workers=True)

    datamodule = HatefulTweets(model_file, 128)
    model = BinaryMLP(300, [512, 256, 128, 64], learning_rate=1e-4)

    best_log = train_model(model, datamodule, name=name, epochs=200, verbose=verbose)
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


def run_repeated_fasttext(
    model_file: Path,
    name: str = "model",
    verbose: bool = False,
) -> dict[str, str]:
    return run_repeated(
        lambda seed: run_fasttext_test(
            name=f"{name}_{seed}", seed=seed, verbose=verbose, model_file=model_file
        )
    )


def run_repeated_transformer(
    name: str = "model", verbose: bool = False, seed_start=1, seed_end=11
) -> dict[str, str]:
    return run_repeated(
        lambda seed: run_transformer_test(
            name=f"{name}_{seed}", seed=seed, verbose=verbose
        ),
        seed_start=seed_start,
        seed_end=seed_end,
    )
