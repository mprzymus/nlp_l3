import re
import typing as t
from itertools import chain
from pathlib import Path

from datasets import load_dataset
from pyarrow.lib import StringScalar
from tqdm.contrib import tzip

from config import CONTENT_LINK_REGEX, ENGLISH_TEST, ENGLISH_TRAIN
from text_processing import clean_to_sentences

NON_BREAKING_SPACE_REGEX = re.compile(r"\u00A0", re.UNICODE)
USELESS_UNICODE_REGEX = re.compile(r"(\ufe0f|\u2066|\u2069|\u00ad)", re.UNICODE)


def extract(
    text_iter: t.Iterable[StringScalar],
    label_iter: t.Iterable[StringScalar],
    out_file: Path,
    sep: str = ",",
) -> None:
    with out_file.open(mode="w", encoding="utf-8") as f:
        f.write(f"text{sep}label\n")

        for text, label in tzip(text_iter, label_iter):
            text = (
                text.as_py()
                .replace("\\n", " ")
                .replace(r"\"", '"')
                .replace("\n\n", " ")
                .replace("\\u0026amp;", "&")
                .replace("\\u0026gt;", ">")
                .replace("\\u0026lt;", "<")
                .replace("\\\\", "\\")
                # .replace("RT", "")
            )
            text = CONTENT_LINK_REGEX.sub("", text)
            text = NON_BREAKING_SPACE_REGEX.sub(" ", text)
            text = USELESS_UNICODE_REGEX.sub("", text)
            text = " ".join(
                clean_to_sentences(
                    text,
                    user_handle="@user",
                    language="english",
                )
            )

            if text:
                f.write(f"{text.strip()}{sep}{label.as_py()}\n")


def main() -> None:
    dataset = load_dataset("tweet_eval", "hate")

    extract(
        chain(dataset.data["train"]["text"], dataset.data["validation"]["text"]),
        chain(dataset.data["train"]["label"], dataset.data["validation"]["label"]),
        ENGLISH_TRAIN,
    )

    extract(
        dataset.data["test"]["text"],
        dataset.data["test"]["label"],
        ENGLISH_TEST,
    )

    # for subset in dataset:
    #     out = ENGLISH_TRAIN if subset == "train" else ENGLISH_TEST
    #     extract(dataset.data[subset], out)  # type: ignore


if __name__ == "__main__":
    main()
