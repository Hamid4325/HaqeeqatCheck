"""CTC label converter and image normalization (adapted from UTRNet webapp utils.py)."""

import math
from importlib import resources

import torch
import torchvision.transforms as T


def load_urdu_glyphs() -> str:
    """Return the UTRNet Urdu glyph vocabulary as a single string plus trailing space."""
    text = (
        resources.files("ingestion.urdu_ocr")
        .joinpath("UrduGlyphs.txt")
        .read_text(encoding="utf-8")
    )
    glyphs = "".join(line for line in text.splitlines())
    return glyphs + " "


class NormalizePAD:
    """Convert a PIL image to a normalized, right-padded FloatTensor of max_size."""

    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = T.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return pad_img


class CTCLabelConverter:
    """Convert between text labels and CTC indices. Index 0 is the CTC blank."""

    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        self.character = ["[CTCblank]"] + dict_character

    def encode(self, text, batch_max_length=25):
        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            indices = [self.dict[char] for char in t]
            batch_text[i][: len(indices)] = torch.LongTensor(indices)
        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            texts.append("".join(char_list))
        return texts
