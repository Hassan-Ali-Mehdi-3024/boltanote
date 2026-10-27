"""Build data/splits.csv once (the file is committed, so everyone gets the same split).

1. The currency photos repeat the same few physical notes on the same bedsheets, so a plain random
   split puts near-copies of a test photo in training. We hash every photo (perceptual hash), join
   photos within 8 bits of each other into one group, and split whole groups 70/15/15 per class.
2. "none" = the dataset's 280 "others" photos + 1,000 Imagenette photos (100 per class).
3. External test: the Roboflow set's test + valid photos that show exactly one note value, minus any
   within 8 bits of a photo in our data (that set re-uses some of the same photos).

usage: python scripts/make_splits.py --data /path/to/data
"""

import argparse
import collections
import csv
import json
import os
import sys

import imagehash
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boltanote.models import CLASSES  # noqa: E402

ROBOFLOW_NAMES = ["10", "100", "1000", "20", "50", "500", "5000", "75"]
THRESH = 8


def phash(path):
    im = Image.open(path)
    im.draft("RGB", (256, 256))
    return int(str(imagehash.phash(ImageOps.exif_transpose(im).convert("RGB"))), 16)


def hamming_matrix(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(len(a), len(b), 8), axis=2).sum(2)


def group_ids(hashes):
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h = np.array(hashes, dtype=np.uint64)
    for s in range(0, len(h), 512):
        d = hamming_matrix(h[s:s + 512], h)
        for i, j in zip(*np.nonzero(d <= THRESH)):
            a, b = find(s + i), find(j)
            if a != b:
                parent[a] = b
    return [find(i) for i in range(len(h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data/splits.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    rows = []

    for c in CLASSES:
        folder = "others" if c == "none" else c
        for f in sorted(os.listdir(os.path.join(a.data, "rafay", folder))):
            rows.append({"path": f"rafay/{folder}/{f}", "source": "rafay", "class": c})
    print("hashing", len(rows), "currency photos...")
    hashes = [phash(os.path.join(a.data, r["path"])) for r in rows]
    gid = group_ids(hashes)

    for c in CLASSES:  # split whole groups, class by class
        idx = [i for i, r in enumerate(rows) if r["class"] == c]
        groups = collections.defaultdict(list)
        for i in idx:
            groups[gid[i]].append(i)
        order = list(groups.values())
        rng.shuffle(order)
        n, done = len(idx), 0
        for g in order:
            split = "train" if done < 0.70 * n else "val" if done < 0.85 * n else "test"
            for i in g:
                rows[i]["split"] = split
            done += len(g)
    sizes = collections.Counter(gid)
    print("near-duplicate groups:", len(sizes), "| photos in groups of 2+:", sum(v for v in sizes.values() if v > 1))

    inet = os.path.join(a.data, "imagenette", "imagenette2-160", "train")
    for k, wnid in enumerate(sorted(d for d in os.listdir(inet) if d.startswith("n"))):
        files = sorted(f for f in os.listdir(os.path.join(inet, wnid)) if f.endswith(".JPEG"))[:100]
        for j, f in enumerate(files):
            split = "train" if j < 70 else "val" if j < 85 else "test"
            rows.append({"path": f"imagenette/imagenette2-160/train/{wnid}/{f}", "source": "imagenette", "class": "none", "split": split})

    ours = np.array(hashes, dtype=np.uint64)
    kept, dropped, mixed = 0, 0, 0
    for part in ("test", "valid"):
        img_dir = os.path.join(a.data, "roboflow", part, "images")
        for f in sorted(os.listdir(img_dir)):
            lab = os.path.join(a.data, "roboflow", part, "labels", os.path.splitext(f)[0] + ".txt")
            ids = {int(l.split()[0]) for l in open(lab) if l.strip()} if os.path.exists(lab) else set()
            names = {ROBOFLOW_NAMES[i] for i in ids}
            if len(names) != 1 or "75" in names:
                mixed += 1
                continue
            h = np.array([phash(os.path.join(img_dir, f))], dtype=np.uint64)
            if hamming_matrix(h, ours).min() <= THRESH:
                dropped += 1
                continue
            rows.append({"path": f"roboflow/{part}/images/{f}", "source": "roboflow", "class": "", "group": names.pop(), "split": "external"})
            kept += 1
    print(f"external (Roboflow): kept {kept}, dropped {dropped} near-copies of our photos, skipped {mixed} with several values or none")
    os.makedirs("results", exist_ok=True)
    with open("results/dedupe.json", "w") as f:
        json.dump({"hamming_threshold_bits": THRESH, "currency_photos": len(hashes), "near_duplicate_groups": len(set(gid)),
                   "roboflow_total": kept + dropped, "roboflow_dropped": dropped, "roboflow_kept": kept, "roboflow_skipped_multi_or_none": mixed}, f, indent=2)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "split", "class", "group"])
        for r in rows:
            w.writerow([r["path"], r["source"], r["split"], r["class"], r.get("group") or r["class"].split("_")[0]])
    print(collections.Counter((r["split"]) for r in rows))
    print(collections.Counter((r["split"], r.get("group") or r["class"].split("_")[0]) for r in rows if r["split"] == "external"))


if __name__ == "__main__":
    main()
