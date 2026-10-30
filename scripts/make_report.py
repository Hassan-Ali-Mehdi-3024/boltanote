"""Write the project report (HTML + PDF) from the results. Run after report.py.

    python scripts/make_report.py      -> docs/BoltaNote_report.pdf, docs/report/index.html
"""

import html
import json
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "tools"))

NICE = {"cnn_scratch": "CNN from scratch", "mobilenetv3_large": "MobileNetV3-Large", "mobilenetv4_conv_medium": "MobileNetV4-Conv-Medium"}
pct = lambda v, d=1: f"{v * 100:.{d}f}%" if v is not None else "–"
links = json.load(open("links.json"))
S = json.load(open("results/summary.json"))
runs, best = S["models"], S["deployed_model"]
B = runs[best]
D = json.load(open("results/dedupe.json"))
tests = [s for s in ("test", "external", "own") if s in B]
LBL = {"test": "Same-dataset test", "external": "Other photographers", "own": "Our own photos"}


def table(rows, head):
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


rows = []
for m, r in runs.items():
    rows.append([NICE[m] + (" <b>(deployed)</b>" if m == best else ""), f"{r['params_million']}M", f"{r['onnx_mb']} MB", f"{r['onnx_cpu_ms']:.0f}",
                 *[pct(r[s]["value_accuracy"]) if s in r else "–" for s in ("test", "external", "own")], f"{r['speaking_threshold']:.2f}"])
speak_rows = []
for s in tests:
    sp = B[s]["speaking"]
    speak_rows.append([LBL[s], B[s]["n"], pct(B[s]["value_accuracy"]), pct(sp["wrong_value_rate"]), pct(sp["coverage_on_notes"]),
                       pct(sp["accuracy_when_speaking"]), pct(sp["false_alarm_on_no_note"])])
recall_rows = [[("Rs " + v) if v != "none" else "no note", *[pct(B[s]["per_value_recall"].get(v)) for s in tests]] for v in B["test"]["per_value_recall"]]

ext, own = B.get("external"), B.get("own")
gap = (B["test"]["value_accuracy"] - ext["value_accuracy"]) if ext else None
pre = {m: r for m, r in runs.items() if m != "cnn_scratch"}
scratch = runs.get("cnn_scratch")
speed_txt = ""
if "mobilenetv3_large" in runs and "mobilenetv4_conv_medium" in runs:
    v3, v4 = runs["mobilenetv3_large"], runs["mobilenetv4_conv_medium"]
    speed_txt = (f"MobileNetV3-Large runs in {v3['onnx_cpu_ms']:.0f} ms per photo on a 2-core CPU, MobileNetV4-Conv-Medium in {v4['onnx_cpu_ms']:.0f} ms "
                 f"({v4['onnx_cpu_ms'] / v3['onnx_cpu_ms']:.1f}× slower, {v4['params_million'] / v3['params_million']:.1f}× the parameters). ")
own_txt = (f"On {own['n']} photos of our own notes, taken with a phone in ordinary rooms and never used for any decision, the deployed model names the value "
           f"correctly {pct(own['value_accuracy'])} of the time and, at its speaking threshold, would say a wrong value on {pct(own['speaking']['wrong_value_rate'])} of them."
           if own else "The test on our own photos was still being collected when this version was written.")

ref = [
    "Hassan B, et al. A comprehensive study capturing vision loss burden in Pakistan (1990–2025): findings from the Global Burden of Disease (GBD) 2017 study. <i>PLoS One</i> 14(5), 2019.",
    "Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3. <i>ICCV</i>, 2019.",
    "Qin D, Leichner C, Delakis M, et al. MobileNetV4: universal models for the mobile ecosystem. <i>ECCV</i>, 2024.",
    "Wightman R. PyTorch Image Models (timm). https://github.com/huggingface/pytorch-image-models, 2019–2026.",
    "Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L. ImageNet: a large-scale hierarchical image database. <i>CVPR</i>, 2009.",
    "Howard J. Imagenette: a smaller subset of 10 easily classified classes from ImageNet. https://github.com/fastai/imagenette, 2019.",
    "Qayyum R. Pakistan Image Currency Dataset. Kaggle (CC0), https://www.kaggle.com/datasets/rafayqayyum/pakistan-image-currency-dataset.",
    "Ali MA. Pakistani Currency Notes Dataset (2kbase). Roboflow / Kaggle (CC BY 4.0), https://www.kaggle.com/datasets/muhammadasjadali/pakistani-currency-notes.",
    "Zauner C. Implementation and benchmarking of perceptual image hash functions. Master's thesis, Upper Austria University of Applied Sciences, 2010.",
    "Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. <i>ICML</i>, 2017.",
    "Geirhos R, Jacobsen J-H, Michaelis C, et al. Shortcut learning in deep neural networks. <i>Nature Machine Intelligence</i> 2, 2020.",
    "Loshchilov I, Hutter F. Decoupled weight decay regularization. <i>ICLR</i>, 2019.",
    "Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the Inception architecture for computer vision (label smoothing). <i>CVPR</i>, 2016.",
    "ONNX Runtime contributors. ONNX Runtime Web. https://onnxruntime.ai, 2024.",
]

hp = [["Input", "RGB photo, whole frame squeezed to 224 × 224, values in [0, 1] (cached at 320 px for augmentation)"], ["Classes", "15 = 7 values × front/back + none"],
      ["Optimiser", "AdamW, weight decay 0.05, label smoothing 0.1"], ["Learning rate", "1e-3 (MobileNets), 2e-3 (scratch); 5% warm-up then cosine"],
      ["Epochs", "40 (MobileNets), 80 (scratch); best epoch by validation value accuracy"], ["Batch", "64, bfloat16 mixed precision, 1 × NVIDIA L4 (Colab)"],
      ["Augmentation (GPU)", "rotation ±180°, zoom 0.85–1.9×, stretch 0.8–1.25, shift ±18%, 0–2 finger/shadow patches, brightness/contrast ±35%, gamma ±30%, saturation ±40%, colour cast ±8%, noise, blur (p = 0.3); no mirror flips"],
      ["Speaking threshold", S["speaking_threshold_method"]]]

H = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Pakistani Banknote Reader: report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet">
<style>
body{{font-family:"Source Serif 4",Georgia,serif;font-size:10.6pt;line-height:1.5;color:#111;max-width:820px;margin:0 auto;padding:0 14px}}
h1,h2,h3,.meta,table,figcaption,.abstract b{{font-family:Inter,system-ui,sans-serif}}
h1{{font-size:21pt;line-height:1.15;margin:.2em 0 .25em;letter-spacing:-.01em}} h2{{font-size:13pt;margin:1.5em 0 .4em;border-bottom:1px solid #ddd;padding-bottom:3px}}
.meta{{font-size:9pt;color:#444;margin-bottom:12px}} .meta a{{color:#0a5c36}} .ur{{font-family:"Noto Nastaliq Urdu",serif}}
.abstract{{background:#fbf7e6;border-left:4px solid #d4a017;padding:10px 14px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:8.4pt;margin:8px 0 4px}} th,td{{border-bottom:1px solid #ddd;padding:4px 5px;text-align:left;vertical-align:top}} th{{background:#f5f5f5}}
figure{{margin:12px 0;break-inside:avoid}} figure img{{width:100%;border:1px solid #e5e5e5}} figcaption{{font-size:8.6pt;color:#444;margin-top:3px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} code{{font-size:8.6pt;background:#f2f2f2;padding:0 3px}}
ol.refs{{font-size:8.8pt;padding-left:18px}} .key{{background:#eef6f0;border-left:4px solid #0a5c36;padding:8px 12px;margin:10px 0}}
@media print{{h2{{break-after:avoid}} table{{break-inside:avoid}}}}
</style></head><body>
<h1>A talking Pakistani banknote reader for blind people, tested beyond its own dataset</h1>
<div class="meta">{html.escape(links['author'])} · BS Computer Science (7A), Superior University, Lahore · Deep Learning, instructor Dr. Munir Ahmad · {links.get('month', 'November 2026')}<br>
Live demo: <a href="{links.get('site') or '#'}">{links.get('site') or '(Vercel link)'}</a> · Code, weights &amp; photos: <a href="{links['github']}">{links['github']}</a>{(' · Video: <a href="' + links['youtube'] + '">' + links['youtube'] + '</a>') if links.get('youtube') else ''}</div>

<div class="abstract"><b>Abstract.</b> Blind people in Pakistan (about 1.1 million [1]) cannot tell banknotes apart without help, and a Rs 500 / Rs 5,000 mix-up costs Rs 4,500. The reader is a web app that reads the phone camera about four times a second and says the note's value aloud in Urdu (<span class="ur">پانچ سو روپے</span>) or English, offline once loaded. We fine-tune MobileNetV3-Large [2] and MobileNetV4-Conv-Medium [3] and train a CNN from scratch on a public dataset of 3,611 phone photos (15 classes: seven notes × front/back, plus "no note"). Before measuring anything we checked how much the public data repeats itself: {D['roboflow_dropped']} of the {D['roboflow_total']} test photos in a second public currency dataset are near-copies of the first, leaving {D['roboflow_kept']} genuinely new photos. The deployed {NICE[best]} scores {pct(B['test']['value_accuracy'])} on unseen photos from the training dataset but {pct(ext['value_accuracy']) if ext else '–'} on photos by other people{(' and ' + pct(own['value_accuracy']) + ' on ' + str(own['n']) + ' photos of our own notes') if own else ''}. Because saying a wrong amount is worse than saying nothing, the app speaks only above a confidence threshold calibrated on a stress-tested validation set.</div>

<h2>1. Problem</h2>
<p>Cash is still how most everyday purchases in Pakistan are paid. The seven notes in circulation (Rs 10 to Rs 5,000) differ only modestly in size, so a blind person usually has to ask someone and trust the answer. Commercial apps exist for some currencies, but an assistant for Pakistani notes should work on cheap phones, speak Urdu, and need no installation. The cost of errors is asymmetric: silence just means "move the note and try again", while a wrong value can cost real money.</p>
<p><b>Contributions.</b> (1) A working Urdu-speaking reader that runs in any phone browser. (2) A measurement of hidden duplication in public currency datasets, and an evaluation that goes beyond the training dataset: other photographers and our own photos. (3) A speaking rule that trades coverage for safety, calibrated without touching test data. (4) An accuracy-versus-speed comparison of three architectures for phones.</p>

<h2>2. Data</h2>
<p><b>Training data.</b> The Pakistan Image Currency Dataset [7] (CC0) has 3,611 phone photos of the 2005–2008 note series, each value front and back, many folded, partly visible, or in a hand, plus 280 "other" photos. We add 1,000 Imagenette photos [6] (100 per class) to the "no note" class so the model learns to stay silent.</p>
<p><b>Duplicate check.</b> We computed a 64-bit perceptual hash [9] of every photo and joined photos within 8 bits of each other. Inside the training dataset, near-copies are kept on the same side of the 70 / 15 / 15 per-class split (seed 42). Against the Roboflow currency dataset [8], which we intended as an independent test, <b>{D['roboflow_dropped']} of {D['roboflow_total']}</b> single-note test photos turned out to be near-copies of training photos; only {D['roboflow_kept']} remain as a genuine external test. Without this check that external test would have reported near-perfect accuracy.</p>
<p><b>Our own photos.</b> {('We photographed our own notes with a phone: flat, folded, in hand, partly covered, in dim and yellow light, on different backgrounds, plus photos with no note. ' + str(own['n']) + ' photos, sorted by value in <code>data/own_photos</code>, used only for testing.') if own else 'Being collected: about 300 photos of our own notes under real conditions, used only for testing.'}</p>
{table([["Train", "3,429"], ["Validation", "735"], ["Same-dataset test", "727"], ["External test (other photographers)", str(D['roboflow_kept'])], ["Our own photos", str(own['n']) if own else "–"]], ["Split", "Photos"])}

<h2>3. Method</h2>
<p>Each model sees the whole camera frame squeezed to 224 × 224 (the same preprocessing in training, <code>predict.py</code> and the browser) and outputs 15 class probabilities; front and back are separate classes because they look nothing alike, and the app adds them per value. Augmentation is designed around how blind users actually hold notes: any rotation, too close or too far, half the note out of frame, fingers over it, dim or yellow light, motion blur. Mirror flips are excluded because the camera never sees a mirrored note.</p>
{table(hp, ["Setting", "Value"])}
<div class="key"><b>When to speak.</b> On ordinary validation photos the model is almost never wrong, so any threshold looks safe there (the lowest we searched, {B['threshold_from_clean_val']:.2f}, already gives no more than 1 wrong answer in 500). Modern networks are over-confident on data like their training data [10]. We therefore calibrate on a <i>stress</i> validation set: each validation photo four times with the training augmentations. Because a wrong amount costs real money, we require at most 1 wrong value in 500 there (0.2%); the lowest threshold that achieves it is <b>{B['speaking_threshold']:.2f}</b>. The app additionally requires the same answer in three consecutive frames before speaking.</div>

<h2>4. Results</h2>
{table(rows, ["Model", "Params", "ONNX", "CPU ms", "Same-dataset test", "Other photographers", "Own photos", "Threshold"])}
<p>All models are near-perfect on unseen photos from the training dataset, which mostly shows how similar those photos are to each other. The informative columns are the other two: {('on photos by other people the deployed model drops by ' + f'{gap * 100:.1f}' + ' points') if gap is not None else ''}{(', and the scratch CNN, which matched the pretrained models on the same-dataset test, falls to ' + pct(scratch['external']['value_accuracy'])) if scratch and scratch.get('external') else ''}. ImageNet pretraining buys robustness that the in-dataset test cannot show. {speed_txt}{own_txt}</p>
{table(speak_rows, ["Test set", "Photos", "Value accuracy", "Says wrong value", "Speaks on note photos", "Correct when speaking", "Speaks with no note"])}
{table(recall_rows, ["Value", *[LBL[s] for s in tests]])}
<div class="two"><figure><img src="figures/accuracy_vs_speed.png"><figcaption>Figure 1. Accuracy on each test set against CPU time per photo.</figcaption></figure>
<figure><img src="figures/speak_threshold.png"><figcaption>Figure 2. Share of photos on which the deployed model would say a wrong value, by threshold.</figcaption></figure></div>
<figure><img src="figures/confusion_matrix.png"><figcaption>Figure 3. Confusion matrices of the deployed model on each test set.</figcaption></figure>
{'<figure><img src="figures/own_photos.png"><figcaption>Figure 4. Our own photos: share correct per value.</figcaption></figure>' if own else ''}
<figure><img src="figures/training_curves.png"><figcaption>Figure {5 if own else 4}. Training loss and validation accuracy.</figcaption></figure>

<h2>5. Deployment</h2>
<p>The deployed model is exported to ONNX (opset 17, {B['onnx_mb']} MB, softmax included) and TorchScript. The website runs it with ONNX Runtime Web on the phone, on the live rear camera, about four frames a second; the speech is pre-recorded with neural Urdu and English voices so it works even on phones without Urdu text-to-speech, and Android phones also vibrate a code (1–4 short pulses for Rs 10–100, 1–3 long for Rs 500–5,000). Results are announced in an ARIA live region so screen readers (TalkBack, VoiceOver) read them too. The repository holds code, splits, own photos, all weights with SHA-256 hashes in <code>model_card.json</code>, the executed notebook, and a GitHub Actions test.</p>

<h2>6. Limitations</h2>
<ul><li>The training photos show a few physical notes per value on similar indoor backgrounds; very worn, torn or taped notes are under-represented.</li>
<li>It knows the 2005–2008 designs; the State Bank of Pakistan's announced new series will need new photos and retraining.</li>
<li>One note at a time; it is not a counterfeit detector.</li>
<li>The external test has only {D['roboflow_kept']} photos, so its uncertainty is large (one photo ≈ {100 / max(D['roboflow_kept'], 1):.1f} points).</li></ul>

<h2>7. Conclusion</h2>
<p>A phone-sized network reads Pakistani banknotes aloud in Urdu, offline, in any browser. The headline in-dataset accuracy is close to perfect, but the honest numbers come from photos the dataset never saw, and a speaking rule tuned for safety matters as much as raw accuracy. Next: photos of worn notes, the new note series, counting several notes in one frame, and a field test with blind users.</p>

<h2>References</h2><ol class="refs">{''.join(f'<li>{r}</li>' for r in ref)}</ol>
<p class="meta">This report was drafted with the help of an AI assistant (Claude) from the project's code and results, as the course requires, and checked against them.</p>
</body></html>"""

os.makedirs("docs/report/figures", exist_ok=True)
for f in os.listdir("results/figures"):
    shutil.copy(f"results/figures/{f}", f"docs/report/figures/{f}")
open("docs/report/index.html", "w", encoding="utf-8").write(H)
from htmlpdf import render  # noqa: E402

render("docs/report/index.html", "docs/BoltaNote_report.pdf")
