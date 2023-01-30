import logging
import os
import time
import typing
import warnings
from contextlib import redirect_stderr
from functools import partial
from pathlib import Path
from statistics import mean, stdev

import fasttext
import pandas as pd
import pytorch_lightning as pl
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

from config import ENGLISH_TEST, ENGLISH_TRAIN, PROBLEM_TEST, PROBLEM_TRAIN
from data import HatefulTweets, RnnDataset, TextDataset, WordDataset
from nn import BinaryMLP, CNNModel, LSTMModel, RNNModel, train_model
from text_processing import get_fasttext_embeddings


def run_lstm_test(
    datamodule: pl.LightningDataModule,
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
    feature_size=300,
    hidden_size=150,
    num_layers=1,
) -> dict[str, float]:
    pl.seed_everything(seed, workers=True)
    model = LSTMModel(
        feature_size=feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=1e-4,
    )
    start = time.perf_counter()
    best_log = train_model(model, datamodule, name=name, epochs=200, verbose=verbose)
    end = time.perf_counter()
    best_log["train_time"] = end - start
    return best_log


def run_rnn_test(
    datamodule: pl.LightningDataModule,
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
    feature_size=300,
    hidden_size=150,
    num_layers=1,
) -> dict[str, float]:
    pl.seed_everything(seed, workers=True)

    model = RNNModel(
        feature_size=feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=1e-4,
    )
    start = time.perf_counter()
    best_log = train_model(model, datamodule, name=name, epochs=200, verbose=verbose)
    end = time.perf_counter()
    best_log["train_time"] = end - start

    return best_log


def run_mlp_test(
    datamodule: pl.LightningDataModule,
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    pl.seed_everything(seed, workers=True)

    model = BinaryMLP(300, [512, 256, 128, 64], learning_rate=1e-4)

    start = time.perf_counter()
    best_log = train_model(model, datamodule, name=name, epochs=200, verbose=verbose)
    end = time.perf_counter()

    best_log["train_time"] = end - start
    return best_log


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


def run_labse_test_single(
    datamodule: pl.LightningDataModule,
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
):
    pl.seed_everything(seed, workers=True)

    model = BinaryMLP(
        emb_dim=768,
        hidden_dims=[256, 128],
        learning_rate=1e-4,
    )

    start = time.perf_counter()
    best_log = train_model(
        model,
        datamodule,
        name=f"{name}",
        epochs=200,
        verbose=verbose,
    )
    end = time.perf_counter()

    best_log["train_time"] = end - start
    return best_log


def run_labse_test_multi(
    datamodule_english: pl.LightningDataModule,
    datamodule_polish: pl.LightningDataModule,
    name: str = "model",
    seed: int = 42,
    verbose: bool = True,
):
    pl.seed_everything(seed, workers=True)

    model = BinaryMLP(
        emb_dim=768,
        hidden_dims=[256, 128],
        learning_rate=1e-4,
    )

    english_logs = train_model(
        model,
        datamodule_english,
        name=f"{name}_english",
        epochs=200,
        verbose=verbose,
    )

    model.load_state_dict(torch.load(f"checkpoints/{name}_english.ckpt")["state_dict"])
    polish_pre_training_logs = model.trainer.validate(
        model,
        datamodule=datamodule_polish,
        verbose=verbose,
    )[0]

    polish_logs = train_model(
        model,
        datamodule_polish,
        name=f"{name}_polish",
        epochs=200,
        verbose=verbose,
    )

    return {
        **{f"english_{k}": v for k, v in english_logs.items()},
        **{f"polish_pre_training_{k}": v for k, v in polish_pre_training_logs.items()},
        **{f"polish_{k}": v for k, v in polish_logs.items()},
    }


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


def run_repeated_mlp(
    embeddings_model_path: Path,
    train_path: Path = PROBLEM_TRAIN,
    test_path: Path = PROBLEM_TEST,
    name: str = "model",
    verbose: bool = False,
) -> dict[str, str]:
    with open(os.devnull, "w") as null:
        with redirect_stderr(null):
            embeddings_model = fasttext.load_model(str(embeddings_model_path))

    get_embeddings = partial(get_fasttext_embeddings, embeddings_model)

    train_dataset = TextDataset(train_path, get_embeddings)
    test_dataset = TextDataset(test_path, get_embeddings)

    datamodule = HatefulTweets(train_dataset, test_dataset, 128)

    return run_repeated(
        lambda seed: run_mlp_test(
            datamodule,
            name=f"{name}_{seed}",
            seed=seed,
            verbose=verbose,
        )
    )


def run_repeated_lstm(
    model_file: Path,
    name: str = "lstm",
    verbose: bool = False,
    seed_start=1,
    seed_end=11,
    feature_size=300,
    hidden_size=150,
    num_layers=1,
    train_path: Path = PROBLEM_TRAIN,
    test_path: Path = PROBLEM_TEST,
) -> dict[str, str]:
    train_dataset = RnnDataset(train_path, model_file)
    test_dataset = RnnDataset(test_path, model_file)

    datamodule = HatefulTweets(train_dataset, test_dataset, 128)
    return run_repeated(
        lambda seed: run_lstm_test(
            datamodule=datamodule,
            name=f"{name}_{seed}",
            seed=seed,
            verbose=verbose,
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ),
        seed_start=seed_start,
        seed_end=seed_end,
    )


def run_repeated_rnn(
    model_file: Path,
    name: str = "rnn",
    verbose: bool = False,
    seed_start=1,
    seed_end=11,
    feature_size=300,
    hidden_size=150,
    num_layers=1,
    train_path: Path = PROBLEM_TRAIN,
    test_path: Path = PROBLEM_TEST,
) -> dict[str, str]:
    train_dataset = RnnDataset(train_path, model_file)
    test_dataset = RnnDataset(test_path, model_file)

    datamodule = HatefulTweets(train_dataset, test_dataset, 128)
    return run_repeated(
        lambda seed: run_rnn_test(
            datamodule=datamodule,
            name=f"{name}_{seed}",
            seed=seed,
            verbose=verbose,
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ),
        seed_start=seed_start,
        seed_end=seed_end,
    )


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


def run_repeated_labse_single(
    train_path: Path = PROBLEM_TRAIN,
    test_path: Path = PROBLEM_TEST,
    name: str = "model",
    verbose: bool = False,
):
    embeddings_model = SentenceTransformer("sentence-transformers/LaBSE")
    get_embeddings = lambda x: embeddings_model.encode(
        x,
        convert_to_numpy=False,
        convert_to_tensor=True,
        batch_size=128,
    ).cpu()

    train_dataset = TextDataset(train_path, get_embeddings)
    test_dataset = TextDataset(test_path, get_embeddings)
    datamodule = HatefulTweets(train_dataset, test_dataset, 128)

    return run_repeated(
        lambda seed: run_labse_test_single(
            datamodule=datamodule,
            name=f"{name}_{seed}",
            seed=seed,
            verbose=verbose,
        )
    )


def run_repeated_labse_multi(
    initial_train_path: Path = ENGLISH_TRAIN,
    initial_test_path: Path = ENGLISH_TEST,
    target_train_path: Path = PROBLEM_TRAIN,
    target_test_path: Path = PROBLEM_TEST,
    name: str = "model",
    verbose: bool = False,
):
    embeddings_model = SentenceTransformer("sentence-transformers/LaBSE")
    get_embeddings = lambda x: embeddings_model.encode(
        x,
        convert_to_numpy=False,
        convert_to_tensor=True,
        batch_size=128,
    ).cpu()

    train_dataset_english = TextDataset(initial_train_path, get_embeddings)
    test_dataset_english = TextDataset(initial_test_path, get_embeddings)
    datamodule_english = HatefulTweets(train_dataset_english, test_dataset_english, 128)

    train_dataset_polish = TextDataset(target_train_path, get_embeddings)
    test_dataset_polish = TextDataset(target_test_path, get_embeddings)
    datamodule_polish = HatefulTweets(train_dataset_polish, test_dataset_polish, 128)

    return run_repeated(
        lambda seed: run_labse_test_multi(
            datamodule_english=datamodule_english,
            datamodule_polish=datamodule_polish,
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


def check_errors(
    model: nn.Module,
    data_file: Path,
    data_loader: DataLoader,
    use_sigmoid: bool = True,
    example_num: int = 10,
) -> None:
    model.eval()

    df = pd.read_csv(data_file)
    true_list = []
    pred_list = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda()

            out = model(x).cpu()
            if use_sigmoid:
                out = torch.sigmoid(out)

            true_list.append(y)
            pred_list.append(out)

    trues = torch.cat(true_list)
    pred_probs = torch.cat(pred_list)
    preds = (pred_probs > 0.5).long()

    correct = trues == preds

    print("Predicted correctly:", correct.sum().item())
    print("Predicted incorrectly:", (~correct).sum().item())

    invalid_df = df[(~correct).numpy()]
    invalid_true = trues[~correct]
    invalid_pred_probs = pred_probs[~correct]

    print()
    print("Non-hate tweets predicted as hate:", (invalid_true == 0).sum().item())
    print("Most misclassified examples:")

    example_probs = invalid_pred_probs[invalid_true == 0]
    example_indices = example_probs.argsort(descending=True)[:example_num].numpy()
    example_df = invalid_df[invalid_df["label"] == 0].reset_index(drop=True)
    for idx in example_indices:
        print(
            "\tProb:",
            f"{example_probs[idx]:.3f}",
            "\tText:",
            f"'{example_df['text'].iloc[idx]}'",
        )

    print()
    print("Hate tweets predicted as non-hate:", (invalid_true == 1).sum().item())
    print("Most misclassified examples:")

    example_probs = invalid_pred_probs[invalid_true == 1]
    example_indices = example_probs.argsort(descending=False)[:example_num].numpy()
    example_df = invalid_df[invalid_df["label"] == 1].reset_index(drop=True)

    for idx in example_indices:
        print(
            "\tProb:",
            f"{example_probs[idx]:.3f}",
            "\tText:",
            f"'{example_df['text'].iloc[idx]}'",
        )
