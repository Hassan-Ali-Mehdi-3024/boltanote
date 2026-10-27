"""Figure: what training photos look like after augmentation (results/figures/augmentations.png).
    python scripts/augment_examples.py --data data_raw"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boltanote import augment  # noqa: E402
from boltanote.data import load_rgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
a = ap.parse_args()
rows = [r for r in csv.DictReader(open("data/splits.csv")) if r["split"] == "train" and r["source"] == "rafay" and r["class"] != "none"]
rng = np.random.default_rng(3)
pick = [rows[i] for i in rng.choice(len(rows), 4, replace=False)]
torch.manual_seed(5)
fig, axes = plt.subplots(4, 6, figsize=(12, 8.6))
for r, row in zip(pick, axes):
    x = torch.from_numpy(load_rgb(os.path.join(a.data, r["path"]), 320))[None].repeat(5, 1, 1, 1)
    aug = augment.train_batch(x, 224).permute(0, 2, 3, 1).numpy()
    row[0].imshow(load_rgb(os.path.join(a.data, r["path"]), 224))
    row[0].set_title(f"original: Rs {r['group']}", fontsize=9)
    for ax, im in zip(row[1:], aug):
        ax.imshow(im)
    for ax in row:
        ax.axis("off")
fig.suptitle("Training photos as the model sees them: any angle, too close, fingers over the note, dim or yellow light, blur", fontsize=11)
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/augmentations.png", dpi=110)
print("wrote results/figures/augmentations.png")
