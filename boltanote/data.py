"""Data for BoltaNote. Paths in data/splits.csv are relative to one data folder laid out as:

  rafay/        Pakistan Image Currency Dataset (Kaggle rafayqayyum, CC0): 14 note classes + "others"
  imagenette/   Imagenette 160px (fast.ai): extra "no note in view" photos
  roboflow/     Pakistani Currency Notes Dataset 2kbase (Roboflow / Kaggle muhammadasjadali): external test only

scripts/get_data.sh downloads the first three.
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

from .models import CLASSES, GROUPS


def load_rgb(path, size):
    """The one preprocessing step shared by training, evaluation, predict.py and the website:
    the whole photo, upright, squeezed to size x size."""
    im = Image.open(path)
    if im.format == "JPEG":
        im.draft("RGB", (size * 2, size * 2))  # fast JPEG decode at reduced size
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_all(paths, size, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(lambda p: load_rgb(p, size), paths)))


def read_splits(path):
    """Rows: path, source, split (train/val/test/external), label (class index or -1), group (index into GROUPS)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"path": r["path"], "source": r["source"], "split": r["split"],
                         "label": CLASSES.index(r["class"]) if r["class"] else -1, "group": GROUPS.index(r["group"])})
    return rows


def own_photos(folder):
    """Our own photos: <folder>/<value>/*.jpg, value one of GROUPS. Paths are returned as given."""
    rows = []
    for g in GROUPS:
        d = os.path.join(folder, g)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    rows.append({"path": os.path.join(d, f), "source": "own", "split": "own", "label": -1, "group": GROUPS.index(g)})
    return rows
