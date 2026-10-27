"""Read Pakistani banknotes in photos. Needs only numpy, pillow and onnxruntime.

    python predict.py samples/500_front_1.jpg
    python predict.py samples/*.jpg

Prints one JSON object per photo: the note value (or "none"), its probability, whether the app
would speak, and what it would say in Urdu and English.
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rgb(path, size):
    """Same preprocessing as training (boltanote/data.py): whole photo, upright, squeezed to size x size."""
    im = Image.open(path)
    if im.format == "JPEG":
        im.draft("RGB", (size * 2, size * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)


class BoltaNote:
    def __init__(self, weights=os.path.join(HERE, "weights", "boltanote.onnx"), card=os.path.join(HERE, "model_card.json")):
        self.card = json.load(open(card, encoding="utf-8"))
        self.classes, self.values = self.card["classes"], self.card["values"]
        self.size = self.card["input"]["shape"][-1]
        self.threshold = self.card["speaking"]["threshold"]
        self.phrases = self.card["speaking"]["phrases"]
        self.sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])

    def value_probs(self, path):
        x = load_rgb(path, self.size).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        p = self.sess.run(None, {"image": x})[0][0]
        v = np.zeros(len(self.values))
        for c, pc in zip(self.classes, p):
            v[self.values.index(c.split("_")[0])] += pc
        return v, p

    def __call__(self, path):
        v, p = self.value_probs(path)
        top = self.values[int(v.argmax())]
        speak = top != "none" and v.max() >= self.threshold
        return {
            "file": path,
            "value": top,
            "probability": round(float(v.max()), 4),
            "side": self.classes[int(p.argmax())].split("_")[-1] if top != "none" else None,
            "speaks": bool(speak),
            "says": {"ur": self.phrases[top][0], "en": self.phrases[top][1]} if speak else None,
            "value_probabilities": {k: round(float(x), 4) for k, x in zip(self.values, v)},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "boltanote.onnx"))
    a = ap.parse_args()
    model = BoltaNote(a.weights)
    for path in a.images:
        print(json.dumps(model(path), ensure_ascii=False))


if __name__ == "__main__":
    main()
