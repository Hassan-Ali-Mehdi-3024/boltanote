"""Write README.md from the results (run after report.py)."""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
links = json.load(open("links.json"))
S = json.load(open("results/summary.json"))
D = json.load(open("results/dedupe.json"))
card = json.load(open("model_card.json", encoding="utf-8"))
runs, best = S["models"], S["deployed_model"]
B = runs[best]
ext, own = B.get("external"), B.get("own")
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
gh = links["github"]
slug = gh.replace("https://github.com/", "")
DOCS = {"report": "docs/BoltaNote_report.pdf", "slides": "docs/slides/BoltaNote_slides.pdf"}
for k, path in DOCS.items():
    if not links.get(k) and os.path.exists(path):
        links[k] = path
L = lambda k, text: f"[{text}]({links[k]})" if links.get(k) else f"{text} (coming soon)"
example = subprocess.run([sys.executable, "predict.py", "samples/500_front_1.jpg"], capture_output=True, text=True, check=True).stdout.strip()
example = json.dumps({k: v for k, v in json.loads(example).items() if k != "file"}, indent=1, ensure_ascii=False)
n_own = sum(len([f for f in os.listdir(os.path.join("data/own_photos", d))]) for d in os.listdir("data/own_photos")) if os.path.isdir("data/own_photos") else 0

md = f"""# BoltaNote · بولتا نوٹ

[![test](https://github.com/{slug}/actions/workflows/test.yml/badge.svg)](https://github.com/{slug}/actions/workflows/test.yml)
![license](https://img.shields.io/badge/code-MIT-blue) ![data](https://img.shields.io/badge/data-CC0%20%2B%20own%20photos-yellow)

**A talking banknote reader for blind people in Pakistan. Point the phone camera at a note; it says the value out loud in Urdu (or English). Runs in any phone browser, offline once loaded.**

{L('site', 'Live demo')} · {L('report', 'Report (PDF)')} · {L('slides', 'Slides')} · {L('youtube', 'Video')} · {L('linkedin', 'LinkedIn post')} · [Dataset (Kaggle)]({links['dataset']}) · [Model card](model_card.json)

Deep Learning semester project, BSCS-7A, Superior University, Lahore. Instructor: Dr. Munir Ahmad. Author: {links['author']}.

![Augmented training photos](results/figures/augmentations.png)

## Results

{open('results/results.md').read()}
- Deployed: **{card['architecture']}**, chosen on validation accuracy.
- Same-dataset test: {pct(B['test']['value_accuracy'])} of {B['test']['n']} unseen photos. **Other photographers** ({ext['n'] if ext else 0} photos): {pct(ext['value_accuracy']) if ext else '–'}. **Our own photos**: {(pct(own['value_accuracy']) + ' of ' + str(own['n'])) if own else 'being collected'}.
- The app speaks only when the top value's probability is ≥ **{B['speaking_threshold']:.2f}** (and 3 frames agree). That threshold keeps wrong values ≤ 1 in 500 (0.2%) on a stress-tested validation set (each validation photo × 4 augmentations); no test photo was used to set it.
- **Duplicate check:** {D['roboflow_dropped']} of {D['roboflow_total']} test photos in the second public dataset are near-copies (perceptual hash ≤ 8 bits) of the training dataset's photos; only {D['roboflow_kept']} are used as the external test.

## Test it in 3 commands

```bash
git clone {gh} && cd boltanote
pip install -r requirements.txt          # or just: pip install numpy pillow onnxruntime
python evaluate.py                       # 16 bundled, labelled photos in samples/
```

```bash
python predict.py samples/500_front_1.jpg            # value, probability, what it would say
{"python evaluate.py --folder data/own_photos          # our own photos (one folder per value)" if n_own else ""}
```

`predict.py` output for `samples/500_front_1.jpg` (real):

```json
{example}
```

## Weights

| File | What | Load with |
|---|---|---|
| `weights/boltanote.onnx` | deployed model, outputs `probs` [1,15] | `onnxruntime.InferenceSession(...)` |
| `weights/boltanote_torchscript.pt` | same model, TorchScript | `torch.jit.load(...)` |
| `weights/<model>.pt` | PyTorch checkpoints of all three models | `boltanote.models.load_checkpoint(...)` |

Input: the whole photo, upright, squeezed to 224×224 RGB, float32 in [0, 1], shape `[1, 3, 224, 224]`; normalisation is inside the model. 15 classes: each of Rs 10, 20, 50, 100, 500, 1000, 5000 front and back, plus `none`; add front + back for the value. SHA-256 of every file: [model_card.json](model_card.json).

## Data

| Source | Use | Licence |
|---|---|---|
| [Pakistan Image Currency Dataset]({links['dataset']}) (3,611 note photos + 280 others) | train / val / test, 70/15/15 per class, near-copies kept together | CC0 |
| [Imagenette](https://github.com/fastai/imagenette) (1,000 photos) | extra "no note" photos | Apache-2.0 |
| [Pakistani Currency Notes Dataset]({links['dataset_external']}) ({D['roboflow_kept']} photos after de-duplication) | external test only | CC BY 4.0 |
{f"| [`data/own_photos/`](data/own_photos) ({n_own} photos) | our own notes, real rooms: real-world test only | ours (CC BY 4.0) |" if n_own else ""}

Exact files: [`data/splits.csv`](data/splits.csv). Rebuild: `bash scripts/get_data.sh data_raw && python scripts/make_splits.py --data data_raw`.

## Method

Three models, identical data and recipe (AdamW, cosine schedule, label smoothing 0.1, GPU augmentation built around how blind users hold notes: any rotation, too close, partly out of frame, fingers, dim/yellow light, blur; no mirror flips):

1. `cnn_scratch`: 5-block CNN from random weights
2. `mobilenetv3_large`: MobileNetV3-Large (2019), ImageNet-pretrained
3. `mobilenetv4_conv_medium`: MobileNetV4-Conv-Medium (2024), ImageNet-pretrained

## Reproduce

Training ran on one NVIDIA L4 in Google Colab: [`notebooks/BoltaNote_train.ipynb`](notebooks/BoltaNote_train.ipynb) (all outputs kept).

```bash
bash scripts/get_data.sh data_raw
python train.py --data data_raw --model mobilenetv3_large --out outputs     # ~2 min on an L4
python report.py --runs outputs --data data_raw{" --own data/own_photos" if n_own else ""}       # weights/, figures, model card
```

## Repository

```
predict.py  evaluate.py          inference (numpy + pillow + onnxruntime only)
train.py  report.py              training, export, threshold calibration, figures
boltanote/                       models, data, augmentation, metrics
data/splits.csv{"  data/own_photos/" if n_own else ""}
weights/                         ONNX, TorchScript, PyTorch checkpoints
samples/                         16 labelled test photos
results/                         metrics, per-photo predictions, figures, duplicate check
docs/                            report (PDF + HTML), slides (PPTX, PDF), voiceover script
site/  scripts/build_site.py     the website (deployed on Vercel from web/); site/audio = Urdu/English voice clips
notebooks/                       executed Colab training notebook
```

## Limits

A few physical notes per value in training; 2005–2008 note designs only (the State Bank's new series will need new photos); one note at a time; not a counterfeit detector.

## Credits

Data: Pakistan Image Currency Dataset (Kaggle, CC0), Imagenette (fast.ai), Pakistani Currency Notes Dataset (Roboflow, CC BY 4.0). Models: timm (Ross Wightman). Voice: Microsoft Edge neural voices (ur-PK Uzma, en-IN Neerja). Code: MIT licence.
"""
open("README.md", "w", encoding="utf-8").write(md)
print("wrote README.md")
