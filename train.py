"""Train one banknote model and test it.

    python train.py --data /path/to/data --model mobilenetv4_conv_medium --out outputs

Writes outputs/<model>/: best.pt, history.csv, metrics.json (val, test, external),
test_predictions.csv. Photos are cached on the GPU as uint8 and augmented there.
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from boltanote import augment
from boltanote.data import load_all, read_splits
from boltanote.metrics import speaking_threshold, summarise, to_groups
from boltanote.models import CLASSES, GROUPS, MODELS, build_model

DEFAULTS = {  # epochs, learning rate
    "cnn_scratch": (80, 2e-3),
    "mobilenetv3_large": (40, 1e-3),
    "mobilenetv4_conv_medium": (40, 1e-3),
}
CACHE = 320  # stored size; training crops/zooms from this down to --size


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="use only N photos per split (smoke test)")
    ap.add_argument("--skip-done", action="store_true")
    return ap.parse_args()


def autocast():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, X, idx, size, batch=128):
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        x = augment.eval_batch(X[idx[s:s + batch]], size).contiguous(memory_format=torch.channels_last)
        with autocast():
            out.append(torch.softmax(model(x).float(), 1).cpu())
    return torch.cat(out).numpy()


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    epochs, lr = DEFAULTS[a.model]
    epochs, lr = a.epochs or epochs, a.lr or lr
    out = os.path.join(a.out, a.model)
    if a.skip_done and os.path.exists(os.path.join(out, "metrics.json")):
        print(f"{a.model}: already trained, skipping")
        return
    os.makedirs(out, exist_ok=True)

    rows = read_splits(a.splits)
    split = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "val", "test", "external")}
    if a.limit:
        split = {s: v[:: max(1, len(v) // a.limit)][: a.limit] for s, v in split.items()}
    used = sorted(set(sum(split.values(), [])))
    pos = {i: k for k, i in enumerate(used)}
    print({s: len(v) for s, v in split.items()}, "| device:", dev, torch.cuda.get_device_name() if dev == "cuda" else "")

    t0 = time.time()
    X = torch.from_numpy(load_all([os.path.join(a.data, rows[i]["path"]) for i in used], CACHE)).to(dev)
    lab = torch.tensor([rows[i]["label"] for i in used], device=dev)
    grp = np.array([rows[i]["group"] for i in used])
    idx = {s: torch.tensor([pos[i] for i in v], device=dev) for s, v in split.items()}
    print(f"cached {len(used)} photos at {CACHE}px in {time.time() - t0:.0f}s")

    model = build_model(a.model, pretrained=True).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(idx["train"]) / a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda" and not torch.cuda.is_bf16_supported())

    gv = grp[idx["val"].cpu().numpy()]
    best, history = -1.0, []
    for ep in range(1, epochs + 1):
        model.train()
        t1, total, seen = time.time(), 0.0, 0
        perm = idx["train"][torch.randperm(len(idx["train"]), device=dev)]
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            if len(b) < 2:
                continue
            x = augment.train_batch(X[b], a.size).contiguous(memory_format=torch.channels_last)
            with autocast():
                loss = F.cross_entropy(model(x).float(), lab[b], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, seen = total + loss.item() * len(b), seen + len(b)
        pv = predict(model, X, idx["val"], a.size)
        acc = float((to_groups(pv).argmax(1) == gv).mean())
        history.append({"epoch": ep, "train_loss": total / max(seen, 1), "val_value_accuracy": acc, "seconds": time.time() - t1})
        flag = ""
        if acc > best:
            best, flag = acc, "  *best"
            torch.save({"model": a.model, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "epoch": ep,
                        "classes": CLASSES, "groups": GROUPS, "input": {"size": a.size, "channels": 3, "range": [0, 1]}}, os.path.join(out, "best.pt"))
        print(f"ep {ep:2d}/{epochs}  loss {history[-1]['train_loss']:.4f}  val value-acc {acc:.4f}  {history[-1]['seconds']:.0f}s{flag}", flush=True)

    with open(os.path.join(out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    ck = torch.load(os.path.join(out, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    pv = predict(model, X, idx["val"], a.size)
    thr = speaking_threshold(to_groups(pv), gv)
    result = {"model": a.model, "best_epoch": ck["epoch"], "epochs": epochs, "lr": lr, "batch": a.batch, "image_size": a.size,
              "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
              "train_minutes": round(sum(h["seconds"] for h in history) / 60, 1), "speaking_threshold": thr}
    for s in ("val", "test", "external"):
        if len(idx[s]):
            p = pv if s == "val" else predict(model, X, idx[s], a.size)
            ii = idx[s].cpu().numpy()
            result[s] = summarise(p, grp[ii], thr, lab[idx[s]].cpu().numpy())
            if s == "test":
                with open(os.path.join(out, "test_predictions.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["path", "true_value", "pred_value", "confidence", "pred_class"])
                    gp = to_groups(p)
                    for k, pr, g in zip(ii, p, gp):
                        w.writerow([rows[used[k]]["path"], GROUPS[grp[k]], GROUPS[int(g.argmax())], f"{g.max():.4f}", CLASSES[int(pr.argmax())]])
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    t, e = result["test"], result.get("external")
    print(f"\nTEST {a.model}: value acc {t['value_accuracy']:.4f}  wrong-value {t['speaking']['wrong_value_rate']:.4f}  "
          f"coverage {t['speaking']['coverage_on_notes']:.3f} @ {thr:.2f}" + (f"  | EXTERNAL value acc {e['value_accuracy']:.4f}" if e else ""))


if __name__ == "__main__":
    main()
