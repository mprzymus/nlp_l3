from pathlib import Path

from datasets import load_dataset
from pyarrow.lib import StringScalar
from tqdm.contrib import tzip

from config import CONTENT_LINK_REGEX, PROBLEM_TEST, PROBLEM_TRAIN
from text_processing import clean_to_sentences


def extract(
    data: dict[str, list[StringScalar]],
    out_file: Path,
    sep: str = ",",
) -> None:
    with out_file.open(mode="w", encoding="utf-8") as f:
        f.write(f"text{sep}label\n")

        for text, label in tzip(data["text"], data["label"]):
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
            text = " ".join(clean_to_sentences(text, user_handle="@anonymized_account"))

            if text:
                f.write(f"{text.strip()}{sep}{label.as_py()}\n")


def main() -> None:
    dataset = load_dataset("poleval2019_cyberbullying", "task01")

    for subset in dataset:
        out = PROBLEM_TRAIN if subset == "train" else PROBLEM_TEST
        extract(dataset.data[subset], out)  # type: ignore


if __name__ == "__main__":
    main()
