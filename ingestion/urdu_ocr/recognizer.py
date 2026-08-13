"""Single text-line recognizer using the UTRNet model (adapted from webapp read.py)."""

import math


def preprocess(pil_crop):
    """Grayscale -> mirror -> resize (h=32, w<=400) -> NormalizePAD(1, 32, 400)."""
    import torch
    from PIL import Image

    from ingestion.urdu_ocr.converter import NormalizePAD

    img = pil_crop.convert("L")
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    resized_w = 400 if math.ceil(32 * ratio) > 400 else math.ceil(32 * ratio)
    img = img.resize((resized_w, 32), Image.Resampling.BICUBIC)
    return NormalizePAD((1, 32, 400))(img)


class UTRNetRecognizer:
    """Recognizes the text in one cropped line image."""

    def __init__(self, checkpoint_path, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._converter = None

    def _load(self):
        if self._model is not None:
            return
        import torch

        from ingestion.urdu_ocr.converter import CTCLabelConverter, load_urdu_glyphs
        from ingestion.urdu_ocr.model import Model

        glyphs = load_urdu_glyphs()
        self._converter = CTCLabelConverter(glyphs)
        model = Model(num_class=len(self._converter.character), device=self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._model = model

    def recognize(self, pil_crop) -> str:
        return self.recognize_with_confidence(pil_crop)[0]

    def recognize_with_confidence(self, pil_crop):
        """Return ``(text, mean_confidence)`` for one cropped line image.

        Confidence is the mean over timesteps of the max softmax probability.
        """
        import torch

        self._load()
        tensor = preprocess(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self._model(tensor)
        probs = torch.softmax(preds, dim=2)
        confidence = float(probs.max(dim=2).values.mean())
        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        text = self._converter.decode(preds_index.data, preds_size.data)[0]
        return text, confidence
