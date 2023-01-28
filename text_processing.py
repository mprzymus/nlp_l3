import string
import typing as t

import emoji
import nltk
from nltk.tokenize import casual_tokenize, sent_tokenize

nltk.download("punkt", quiet=True)
TRANSITION_TABLE = str.maketrans(
    "", "", string.punctuation + r"—ᴗ❛… ️³￣’”„𓆉•♡›«»–‘“¦â˜´ðÿ‡​¹¸žïž‹‚¬·"
)


def clean_to_sentences(
    text: str,
    user_handle: str = "@user",
    language="polish",
) -> t.Iterator[str]:
    text = text.replace(user_handle, "").lower()

    sentences = sent_tokenize(text, language=language)
    for sent in sentences:
        words = [
            word.translate(TRANSITION_TABLE)
            for word in casual_tokenize(sent)
            if not word.startswith("#")
        ]

        out_sent = " ".join(
            word for word in words if word and word not in emoji.EMOJI_DATA
        )
        out_sent = out_sent.replace("  ", " ").strip()

        if out_sent:
            yield out_sent
