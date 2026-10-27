"""Test the trained model on labelled photos. Needs only numpy, pillow and onnxruntime.

    python evaluate.py                                # samples/ (labels in samples/labels.csv), no download
    python evaluate.py --folder data/own_photos       # photos sorted into sub-folders 10/ 20/ ... 5000/ none/
    python evaluate.py --data data_raw --split test   # a split from data/splits.csv (after scripts/get_data.sh)

Reports value accuracy, per-value recall, and the two numbers that matter to a blind user at the
model card's threshold: how often it would say a wrong value, and how often it speaks at all on
photos that show a note. Writes evaluation.json.
"""

import argparse
import csv
import json
import os

import numpy as np

from predict import HERE, BoltaNote

PHOTO = (".jpg", ".jpeg", ".png", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", help="folder with one sub-folder per value (10 ... 5000, none)")
    ap.add_argument("--data", help="data folder from scripts/get_data.sh, used with --split")
    ap.add_argument("--split", default="test", choices=["test", "val", "external"])
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "boltanote.onnx"))
    ap.add_argument("--out", default="evaluation.json")
    ap.add_argument("--details", help="also write one CSV row per photo here")
    a = ap.parse_args()
    model = BoltaNote(a.weights)

    if a.folder:
        items = [(os.path.join(a.folder, v, f), v) for v in model.values if os.path.isdir(os.path.join(a.folder, v))
                 for f in sorted(os.listdir(os.path.join(a.folder, v))) if f.lower().endswith(PHOTO)]
    elif a.data:
        items = [(os.path.join(a.data, r["path"]), r["group"]) for r in csv.DictReader(open(os.path.join(HERE, "data", "splits.csv"))) if r["split"] == a.split]
    else:
        items = [(os.path.join(HERE, "samples", r["file"]), r["value"]) for r in csv.DictReader(open(os.path.join(HERE, "samples", "labels.csv")))]

    true, pred, conf, rows = [], [], [], []
    for path, value in items:
        v, _ = model.value_probs(path)
        true.append(value)
        pred.append(model.values[int(v.argmax())])
        conf.append(float(v.max()))
        rows.append([os.path.relpath(path, HERE), value, pred[-1], f"{conf[-1]:.4f}"])
    true, pred, conf = np.array(true), np.array(pred), np.array(conf)
    speaks = (pred != "none") & (conf >= model.threshold)
    note = true != "none"
    out = {
        "photos": int(len(true)),
        "value_accuracy": float((pred == true).mean()),
        "recall_per_value": {v: float((pred[true == v] == v).mean()) for v in model.values if (true == v).any()},
        "threshold": model.threshold,
        "wrong_value_said": float((speaks & (pred != true)).mean()),
        "speaks_on_note_photos": float(speaks[note].mean()) if note.any() else None,
        "accuracy_when_speaking": float((pred[speaks] == true[speaks]).mean()) if speaks.any() else None,
        "false_alarm_on_no_note": float(speaks[~note].mean()) if (~note).any() else None,
        "confusions": sorted(({"true": t, "predicted": p, "count": int(((true == t) & (pred == p)).sum())}
                              for t in set(true) for p in set(pred) if t != p and ((true == t) & (pred == p)).any()), key=lambda d: -d["count"]),
    }
    json.dump(out, open(a.out, "w"), indent=2)
    if a.details:
        with open(a.details, "w", newline="") as f:
            csv.writer(f).writerows([["file", "true_value", "pred_value", "confidence"], *rows])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
