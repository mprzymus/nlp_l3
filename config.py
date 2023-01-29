import re
from pathlib import Path

CHECKPOINTS_DIR = Path("checkpoints")
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("models")

OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SG_FULL = MODELS_DIR / "sg_full.bin"
SG_CORPUS = MODELS_DIR / "sg_corpus.bin"

PROBLEM_TRAIN = DATA_DIR / "problem_train.csv"
PROBLEM_TEST = DATA_DIR / "problem_test.csv"

ENGLISH_TRAIN = OUTPUTS_DIR / "english_train.csv"
ENGLISH_TEST = OUTPUTS_DIR / "english_test.csv"

CONTENT_LINK_REGEX = re.compile(r"https?://t.co/[a-zA-Z0-9]+")
