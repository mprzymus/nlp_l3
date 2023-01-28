import typing
from transformers import AutoModel, AutoTokenizer

from nn import BinaryMLP
import torch


tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
herbert = AutoModel.from_pretrained("allegro/herbert-base-cased")
herbert = herbert.cuda().eval()


def calc_text_embedding(text: typing.List):
    with torch.no_grad():
        out = herbert(
            **tokenizer.encode_plus(
                text,
                padding="longest",
                add_special_tokens=True,
                return_tensors="pt",
            ).to("cuda")
        )
    to_return = out.pooler_output.clone()
    return to_return


if __name__ == "__main__":
    text = [
        "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
        "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy.",
    ]
    out1 = calc_text_embedding(text[0])
    out2 = calc_text_embedding(text[0])
    out = torch.cat((out1, out2))
    model = BinaryMLP(out.shape[1], [512, 256, 128, 64], learning_rate=1e-4).cuda()
    print("out =", model(out))
