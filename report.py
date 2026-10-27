"""After training: compare the three models, pick the deployed one, export it, draw the figures.

    python report.py --runs outputs --data data_raw [--own data/own_photos]

Every number here is recomputed from the exported ONNX files, with the same preprocessing the
website and predict.py use, so what we report is what the app does.

Speaking threshold. Validation photos come from the same few notes and rooms as training, so the
model is over-confident on them (any threshold looks safe). We therefore calibrate on a *stress*
version of the validation set: each validation photo 4 times with the training augmentations
(rotation, zoom, fingers, dim light, blur), and take the lowest threshold at which the model would
say a wrong value on at most 1% of those. No test photo is used for this choice.

The deployed model is the one with the best validation value accuracy (ties: faster on CPU).
Writes weights/, model_card.json, results/.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from boltanote import augment  # noqa: E402
from boltanote.data import load_all, own_photos, read_splits  # noqa: E402
from boltanote.metrics import speaking_threshold, summarise, to_groups  # noqa: E402
from boltanote.models import CLASSES, GROUPS, MODELS, ExportWrapper, load_checkpoint  # noqa: E402

NICE = {"cnn_scratch": "CNN from scratch", "mobilenetv3_large": "MobileNetV3-Large", "mobilenetv4_conv_medium": "MobileNetV4-Conv-Medium"}
COLORS = {"cnn_scratch": "#9a9a9a", "mobilenetv3_large": "#4C78A8", "mobilenetv4_conv_medium": "#E45756"}
SPEECH = {  # what the app says; Urdu first
    "10": ["دس روپے", "Ten rupees"], "20": ["بیس روپے", "Twenty rupees"], "50": ["پچاس روپے", "Fifty rupees"],
    "100": ["سو روپے", "One hundred rupees"], "500": ["پانچ سو روپے", "Five hundred rupees"],
    "1000": ["ایک ہزار روپے", "One thousand rupees"], "5000": ["پانچ ہزار روپے", "Five thousand rupees"],
}
SIZE, STRESS_PASSES = 224, 4


def clean(o):
    """NaN/inf -> None so the JSON files load in browsers (JSON has no NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(model, path):
    x = torch.rand(1, 3, SIZE, SIZE)
    w = ExportWrapper(model).eval()  # a fresh wrapper is in train mode, and export would restore that onto the model
    torch.onnx.export(w, x, path, input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = w(x).numpy()
    err = float(np.abs(ref - sess.run(None, {"image": x.numpy()})[0]).max())
    t = []
    for _ in range(25):
        t0 = time.perf_counter()
        sess.run(None, {"image": x.numpy()})
        t.append(time.perf_counter() - t0)
    return sess, err, float(np.median(t[5:]) * 1000)


def run(sess, x_u8):
    """uint8 [N, 224, 224, 3] -> probs [N, 15]"""
    out = []
    for s in range(0, len(x_u8), 32):
        x = x_u8[s:s + 32].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        out.append(sess.run(None, {"image": x})[0])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs")
    ap.add_argument("--data", required=True, help="folder from scripts/get_data.sh")
    ap.add_argument("--own", default="data/own_photos", help="our own photos, one sub-folder per value")
    a = ap.parse_args()
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    rows = read_splits("data/splits.csv")
    sets = {s: [r for r in rows if r["split"] == s] for s in ("val", "test", "external")}
    own = own_photos(a.own) if os.path.isdir(a.own) else []
    if own:
        sets["own"] = own
    images = {s: load_all([r["path"] if s == "own" else os.path.join(a.data, r["path"]) for r in v], SIZE) for s, v in sets.items()}
    groups = {s: np.array([r["group"] for r in v]) for s, v in sets.items()}
    torch.manual_seed(0)
    val320 = torch.from_numpy(load_all([os.path.join(a.data, r["path"]) for r in sets["val"]], 320))
    stress = np.concatenate([(augment.train_batch(val320, SIZE).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8) for _ in range(STRESS_PASSES)])
    stress_groups = np.tile(groups["val"], STRESS_PASSES)
    print({s: len(v) for s, v in sets.items()}, "| stress-val:", len(stress))

    runs, preds, stress_probs = {}, {}, {}
    for m in MODELS:
        if not os.path.exists(os.path.join(a.runs, m, "metrics.json")):
            continue
        colab = json.load(open(os.path.join(a.runs, m, "metrics.json")))
        shutil.copy(os.path.join(a.runs, m, "best.pt"), f"weights/{m}.pt")
        model, _ = load_checkpoint(f"weights/{m}.pt")
        sess, err, ms = export_onnx(model, f"weights/_{m}.onnx")
        p = {s: run(sess, x) for s, x in images.items()}
        p_stress = run(sess, stress)
        thr = speaking_threshold(to_groups(p_stress), stress_groups, max_wrong=0.01)
        r = {k: colab[k] for k in ("model", "best_epoch", "epochs", "lr", "batch", "image_size", "params_million", "train_minutes")}
        r.update(onnx_cpu_ms=round(ms, 1), onnx_max_abs_err=err, onnx_mb=round(os.path.getsize(f"weights/_{m}.onnx") / 1e6, 1),
                 speaking_threshold=thr, threshold_from_clean_val=speaking_threshold(to_groups(p["val"]), groups["val"]))
        r["stress_val"] = summarise(p_stress, stress_groups, thr)
        for s in sets:
            r[s] = summarise(p[s], groups[s], thr)
        runs[m], preds[m], stress_probs[m] = r, p, p_stress
        print(f"{m}: {ms:.1f} ms/photo, threshold {thr:.2f} (clean val would give {r['threshold_from_clean_val']:.2f}) | "
              + " | ".join(f"{s} acc {r[s]['value_accuracy']:.4f} wrong {r[s]['speaking']['wrong_value_rate']:.4f}" for s in sets))

    best = max(runs, key=lambda m: (runs[m]["val"]["value_accuracy"], -runs[m]["onnx_cpu_ms"]))
    print("deployed:", best)
    for m in runs:
        if m == best:
            os.replace(f"weights/_{m}.onnx", "weights/boltanote.onnx")
            model, _ = load_checkpoint(f"weights/{m}.pt")
            with torch.no_grad():
                torch.jit.trace(ExportWrapper(model).eval(), torch.rand(1, 3, SIZE, SIZE)).save("weights/boltanote_torchscript.pt")
        else:
            os.remove(f"weights/_{m}.onnx")

    with open("results/test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "true_value", "pred_value", "confidence"])
        for s in sets:
            g = to_groups(preds[best][s])
            for r, gp in zip(sets[s], g):
                w.writerow([s, r["path"], GROUPS[r["group"]], GROUPS[int(gp.argmax())], f"{gp.max():.4f}"])

    hist = {m: list(csv.DictReader(open(os.path.join(a.runs, m, "history.csv")))) for m in runs}

    # 1. training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in runs:
        e = [int(h["epoch"]) for h in hist[m]]
        ax[0].plot(e, [float(h["train_loss"]) for h in hist[m]], color=COLORS[m], label=NICE[m])
        ax[1].plot(e, [float(h["val_value_accuracy"]) for h in hist[m]], color=COLORS[m], label=NICE[m])
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].set(title="Validation accuracy (note value)", xlabel="epoch", ylim=(0.3, 1.01))
    ax[1].legend(frameon=False)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/figures/training_curves.png", dpi=150)

    # 2. accuracy vs speed, on each test set
    tests = [s for s in ("test", "external", "own") if s in sets]
    labels = {"test": "same-dataset test", "external": "other photographers", "own": "our own photos"}
    marks = {"test": "o", "external": "s", "own": "D"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for m in runs:
        for s in tests:
            ax.scatter(runs[m]["onnx_cpu_ms"], runs[m][s]["value_accuracy"], marker=marks[s], s=60, color=COLORS[m],
                       label=f"{NICE[m]}" if s == "test" else None)
        ax.plot([runs[m]["onnx_cpu_ms"]] * len(tests), [runs[m][s]["value_accuracy"] for s in tests], color=COLORS[m], alpha=.4)
    for s in tests:
        ax.scatter([], [], marker=marks[s], color="#555", label=labels[s])
    ax.set(xlabel="milliseconds per photo (CPU, ONNX Runtime)", ylabel="accuracy (note value)", title="Accuracy vs speed")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/figures/accuracy_vs_speed.png", dpi=150)

    # 3. confusion matrices of the deployed model
    fig, axes = plt.subplots(1, len(tests), figsize=(5.4 * len(tests), 5))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, tests):
        cm = np.array(runs[best][s]["confusion_matrix_values"])
        keep = [i for i in range(len(GROUPS)) if cm[i].sum() or cm[:, i].sum()]
        cm = cm[np.ix_(keep, keep)]
        names = [("Rs " + GROUPS[i]) if GROUPS[i] != "none" else "no note" for i in keep]
        ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Greens", vmin=0, vmax=1)
        for i in range(len(keep)):
            for j in range(len(keep)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] / max(cm[i].sum(), 1) > 0.5 else "black")
        ax.set(xticks=range(len(keep)), yticks=range(len(keep)), xticklabels=names, yticklabels=names, xlabel="predicted", ylabel="true",
               title=f"{labels[s]} ({runs[best][s]['n']} photos)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(NICE[best])
    fig.tight_layout()
    fig.savefig("results/figures/confusion_matrix.png", dpi=150)

    # 4. speak or stay silent, deployed model
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ts = np.linspace(0.4, 0.99, 60)
    for s, color in (("stress", "#999"), *[(s, c) for s, c in zip(tests, ("#4C78A8", "#E45756", "#2CA02C"))]):
        g, gt = (to_groups(stress_probs[best]), stress_groups) if s == "stress" else (to_groups(preds[best][s]), groups[s])
        pred, conf = g.argmax(1), g.max(1)
        wrong = [((pred != GROUPS.index("none")) & (conf >= t) & (pred != gt)).mean() for t in ts]
        ax.plot(ts, wrong, color=color, label=f"wrong value said: {labels.get(s, 'stress validation')}")
    ax.axvline(runs[best]["speaking_threshold"], ls=":", color="#555")
    ax.text(runs[best]["speaking_threshold"], ax.get_ylim()[1] * 0.92, f"  chosen {runs[best]['speaking_threshold']:.2f}", fontsize=8)
    ax.set(xlabel="confidence needed before speaking", ylabel="share of photos", title="How often it would say a wrong value")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/figures/speak_threshold.png", dpi=150)

    # 5. own photos per value
    if "own" in sets:
        o = runs[best]["own"]
        vals = list(o["per_value_recall"])
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        n = [int((groups["own"] == GROUPS.index(v)).sum()) for v in vals]
        ax.bar([("Rs " + v) if v != "none" else "no note" for v in vals], [o["per_value_recall"][v] for v in vals], color="#2CA02C")
        for i, (v, k) in enumerate(zip(vals, n)):
            ax.text(i, o["per_value_recall"][v] + 0.02, f"{o['per_value_recall'][v]:.0%}\n(n={k})", ha="center", fontsize=8)
        ax.set(ylim=(0, 1.15), ylabel="correct", title=f"Our own photos: {o['value_accuracy']:.1%} of {o['n']} correct")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("results/figures/own_photos.png", dpi=150)
        sp = o["speaking"]
        json.dump(clean({"photos": o["n"], "value_accuracy": o["value_accuracy"], "recall_per_value": o["per_value_recall"], "threshold": sp["threshold"],
                         "wrong_value_said": sp["wrong_value_rate"], "speaks_on_note_photos": sp["coverage_on_notes"],
                         "accuracy_when_speaking": sp["accuracy_when_speaking"], "false_alarm_on_no_note": sp["false_alarm_on_no_note"]}),
                  open("results/own_photos.json", "w"), indent=2)

    summary = {"deployed_model": best, "speaking_threshold_method": "lowest threshold with <= 1% wrong-value rate on 4 augmented passes of the validation set",
               "models": runs}
    json.dump(clean(summary), open("results/summary.json", "w"), indent=2)
    lines = ["| Model | Params | Test | Other photographers | Own photos | Wrong value said (test / other / own) | CPU ms |", "|---|---|---|---|---|---|---|"]
    for m, r in runs.items():
        f = lambda s: f"{r[s]['value_accuracy']:.1%}" if s in r else "–"
        wr = " / ".join(f"{r[s]['speaking']['wrong_value_rate']:.1%}" for s in tests)
        lines.append(f"| {NICE[m]}{' **(deployed)**' if m == best else ''} | {r['params_million']}M | {f('test')} | {f('external')} | {f('own')} | {wr} | {r['onnx_cpu_ms']:.0f} |")
    open("results/results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    b = runs[best]
    card = {
        "name": "BoltaNote", "task": "Pakistani banknote recognition for blind and low-vision people",
        "classes": CLASSES, "values": GROUPS, "class_to_value": {c: c.split("_")[0] for c in CLASSES},
        "input": {"name": "image", "shape": [1, 3, SIZE, SIZE], "dtype": "float32", "range": [0, 1], "layout": "RGB, channels first",
                  "preprocessing": "whole photo, EXIF-rotated upright, resized (squeezed) to 224x224 bilinear, divide by 255; normalisation is inside the model"},
        "outputs": {"probs": "softmax over the 15 classes, shape [1, 15]; add front + back for the note value"},
        "speaking": {"rule": "speak the top value if it is not 'none' and its probability >= threshold (the app also needs 3 frames in a row)",
                     "threshold": b["speaking_threshold"], "threshold_chosen_on": summary["speaking_threshold_method"], "phrases": SPEECH},
        "deployed_model": best, "architecture": NICE[best],
        "test_metrics": {s: b[s] for s in tests},
        "datasets": [
            {"name": "Pakistan Image Currency Dataset", "url": "https://www.kaggle.com/datasets/rafayqayyum/pakistan-image-currency-dataset", "license": "CC0", "use": "train / val / test"},
            {"name": "Imagenette (160 px)", "url": "https://github.com/fastai/imagenette", "license": "Apache-2.0 (images from ImageNet)", "use": "extra 'no note' photos"},
            {"name": "Pakistani Currency Notes Dataset (2kbase)", "url": "https://www.kaggle.com/datasets/muhammadasjadali/pakistani-currency-notes", "license": "CC BY 4.0", "use": "external test only, near-copies removed"},
            {"name": "Our own photos", "use": "real-world test only", "folder": "data/own_photos"},
        ],
        "splits": "data/splits.csv",
        "files": {f: {"bytes": os.path.getsize(f"weights/{f}"), "sha256": sha256(f"weights/{f}")} for f in sorted(os.listdir("weights"))},
        "intended_use": "assistive demo for reading Pakistani banknotes (2005-2008 series); not for detecting counterfeits",
    }
    json.dump(clean(card), open("model_card.json", "w"), indent=2, ensure_ascii=False)
    print("wrote model_card.json, results/, weights/")


if __name__ == "__main__":
    main()
