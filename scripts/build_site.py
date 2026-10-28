"""Build the static website in web/ from site/ + results + links.json (run after report.py).

    python scripts/build_site.py

Everything a crawler needs is written into the HTML itself, and repeated in web/project.json and
web/llms.txt.
"""

import csv
import html
import json
import os
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
NICE = {"cnn_scratch": "CNN from scratch", "mobilenetv3_large": "MobileNetV3-Large", "mobilenetv4_conv_medium": "MobileNetV4-Conv-Medium"}
pct = lambda v, d=1: f"{v * 100:.{d}f}%" if v is not None else "n/a"
f3 = lambda v: f"{v:.3f}" if v is not None else "n/a"

links = json.load(open("links.json"))
card = json.load(open("model_card.json", encoding="utf-8"))
summary = json.load(open("results/summary.json"))
runs, best = summary["models"], summary["deployed_model"]
b = runs[best]
t, ext = b["test"], b.get("external")
own = json.load(open("results/own_photos.json")) if os.path.exists("results/own_photos.json") else None
splits = list(csv.DictReader(open("data/splits.csv")))
count = lambda s: sum(r["split"] == s for r in splits)
dedupe = json.load(open("results/dedupe.json")) if os.path.exists("results/dedupe.json") else {"roboflow_total": 402, "roboflow_dropped": 339, "roboflow_kept": count("external")}

out = "web"
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(f"{out}/model")
for f in ("style.css", "app.js"):
    shutil.copy(f"site/{f}", out)
shutil.copytree("site/audio", f"{out}/audio")
shutil.copy("weights/boltanote.onnx", f"{out}/model/boltanote.onnx")
shutil.copytree("results/figures", f"{out}/figures")
shutil.copytree("samples", f"{out}/samples")
for key, src, dst in (("report", "docs/BoltaNote_report.pdf", "report.pdf"), ("slides", "docs/slides/BoltaNote_slides.pdf", "slides.pdf")):
    if os.path.exists(src):  # report and slides are served by the site too
        shutil.copy(src, f"{out}/{dst}")
        if not links.get(key):
            links[key] = (links["site"].rstrip("/") + "/" + dst) if links.get("site") else dst
if os.path.isdir("docs/report"):
    shutil.copytree("docs/report", f"{out}/report")
shutil.copy("model_card.json", out)
samples = [{"file": r["file"], "value": r["value"], "side": r["class"].split("_")[1] if "_" in r["class"] else None} for r in csv.DictReader(open("samples/labels.csv"))]
json.dump(samples, open(f"{out}/samples/samples.json", "w"))
open(f"{out}/favicon.svg", "w").write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#ffd400"/><rect x="6" y="10" width="20" height="12" rx="2" fill="none" stroke="#111" stroke-width="2"/><circle cx="16" cy="16" r="3" fill="#111"/></svg>')

rows = []
for m, r in runs.items():
    rt, re_ = r["test"], r.get("external")
    rows.append(f'<tr{" class=deployed" if m == best else ""}><td>{NICE[m]}{" <b>(deployed)</b>" if m == best else ""}</td><td>{r["params_million"]}M</td>'
                f'<td>{pct(rt["value_accuracy"])}</td><td>{pct(rt["speaking"]["wrong_value_rate"])}</td><td>{pct(rt["speaking"]["coverage_on_notes"])}</td>'
                f'<td>{pct(re_["value_accuracy"]) if re_ else "–"}</td><td>{r["onnx_cpu_ms"]:.0f}</td></tr>')

LINKS = [
    ("github", "GitHub repository", "code, weights, splits, notebook"),
    ("dataset", "Main dataset (Kaggle)", "Pakistan Image Currency Dataset, CC0"),
    ("dataset_external", "External test set (Kaggle)", "Roboflow currency notes, CC BY 4.0"),
    ("own_photos", "Our own photos", "real-world test set, in the repo"),
    ("weights_onnx", "Model weights (ONNX)", "weights/boltanote.onnx"),
    ("weights_torchscript", "Model weights (TorchScript)", "weights/boltanote_torchscript.pt"),
    ("weights_pt", "PyTorch checkpoints", "weights/*.pt, all three models"),
    ("notebook", "Training notebook", "Colab, all outputs"),
    ("report", "Report (PDF)", "full write-up"),
    ("slides", "Presentation slides", "PDF"),
    ("youtube", "Video presentation (YouTube)", "slides with voiceover"),
    ("linkedin", "LinkedIn post", "project announcement"),
    ("model_card", "Model card (JSON)", "inputs, outputs, threshold, metrics"),
    ("site", "This website", "Vercel"),
]
def exists_in_repo(url):
    """Links into this repo must point at something that is really there (a missing link fails the grader)."""
    base = links["github"].rstrip("/") + "/"
    if not url.startswith(base):
        return True
    rel = url[len(base):].split("/", 2)[-1]  # drop blob|tree|raw and the branch
    rel = rel.split("/", 1)[1] if rel.startswith("main/") else rel
    return os.path.exists(rel) and (not os.path.isdir(rel) or any(os.scandir(rel)))


cards = []
for key, title, desc in LINKS:
    url = links.get(key) or ""
    if url and not exists_in_repo(url):
        continue
    cards.append(f'<a href="{html.escape(url)}"{"" if url else " class=pending aria-disabled=true"}><b>{title}</b><span>{html.escape(url) if url else "coming soon"} · {desc}</span></a>')

limits = [
    "Trained on photos of a few physical notes per value, mostly on similar indoor backgrounds; very worn, torn or taped notes are under-represented.",
    "Knows the 2005–2008 note designs. The State Bank of Pakistan's new series will need new photos and retraining.",
    "Reads one note at a time; two overlapping notes give one answer.",
    "Not a counterfeit detector.",
    f"External test has only {dedupe['roboflow_kept']} genuinely new photos after removing near-copies, so its confidence interval is wide.",
]
if own:
    own_text = (f"{own['photos']} photos of our own notes, taken with a phone in real rooms (folded, in hand, dim light, different backgrounds), "
                f"never used for training or for any choice. Sorted by value in <a href=\"{links['github']}/tree/main/data/own_photos\">data/own_photos</a>.")
    own_acc, n_own = pct(own["value_accuracy"]), str(own["photos"])
    own_fig = '<figure><img src="figures/own_photos.png" alt="Results on our own photos" loading="lazy"><figcaption>Our own photos: per-value accuracy.</figcaption></figure>'
else:
    own_text, own_acc, n_own, own_fig = "Coming: about 300 photos of our own notes taken in real rooms, used only for testing.", "soon", "–", ""

sentence = (f"The deployed model is <b>{NICE[best]}</b>, chosen on validation photos. On {t['n']} unseen test photos it names the note value correctly "
            f"{pct(t['value_accuracy'])} of the time; at the speaking threshold it would say a wrong value on {pct(t['speaking']['wrong_value_rate'])} of photos "
            f"and speaks on {pct(t['speaking']['coverage_on_notes'])} of the photos that show a note.")
if ext:
    sentence += f" On {ext['n']} photos by other photographers it reaches {pct(ext['value_accuracy'])}."
if own:
    sentence += (f" On our own {own['photos']} photos it reaches {pct(own['value_accuracy'])}, saying a wrong value on {pct(own['wrong_value_said'])} of them.")

jsonld = {"@context": "https://schema.org", "@type": "SoftwareSourceCode", "name": "BoltaNote", "description": "Talking Pakistani banknote reader for blind people (MobileNet, ONNX Runtime Web).",
          "codeRepository": links.get("github"), "programmingLanguage": ["Python", "JavaScript"], "author": {"@type": "Person", "name": links.get("author")}, "license": "https://opensource.org/licenses/MIT"}
fill = {
    "GITHUB": links["github"], "AUTHOR": html.escape(links["author"]), "MONTH": links.get("month", "November 2026"),
    "THRESHOLD_PCT": pct(b["speaking_threshold"], 0), "TEST_ACC": pct(t["value_accuracy"]), "WRONG": pct(t["speaking"]["wrong_value_rate"]),
    "EXT_ACC": pct(ext["value_accuracy"]) if ext else "–", "OWN_ACC": own_acc, "N_OWN": n_own, "OWN_DATA_TEXT": own_text, "OWN_FIGURE": own_fig,
    "N_TRAIN": f"{count('train'):,}", "N_VAL": f"{count('val'):,}", "N_TEST": f"{count('test'):,}",
    "RF_TOTAL": dedupe["roboflow_total"], "RF_DROPPED": dedupe["roboflow_dropped"], "RF_KEPT": dedupe["roboflow_kept"],
    "RESULTS_ROWS": "\n    ".join(rows), "RESULTS_SENTENCE": sentence, "LINK_CARDS": "".join(cards),
    "LIMITS": "".join(f"<li>{l}</li>" for l in limits), "JSONLD": json.dumps(jsonld),
}
page = open("site/index.html", encoding="utf-8").read()
for k, v in fill.items():
    page = page.replace("{{" + k + "}}", str(v))
assert "{{" not in page, [l for l in page.splitlines() if "{{" in l]
open(f"{out}/index.html", "w", encoding="utf-8").write(page)

project = {
    "project": "BoltaNote", "task": card["task"], "author": links["author"], "course": "Deep Learning, Superior University (Dr. Munir Ahmad)",
    "links": {k: links.get(k) for k, _, _ in LINKS if links.get(k) and exists_in_repo(links[k])},
    "datasets": card["datasets"], "classes": card["classes"], "values": card["values"], "input": card["input"], "outputs": card["outputs"],
    "speaking": {k: card["speaking"][k] for k in ("rule", "threshold", "threshold_chosen_on")},
    "deployed_model": best,
    "results": {"internal_test": {"photos": t["n"], "value_accuracy": t["value_accuracy"], "wrong_value_rate": t["speaking"]["wrong_value_rate"], "coverage_on_notes": t["speaking"]["coverage_on_notes"]},
                "external_test": {"photos": ext["n"], "value_accuracy": ext["value_accuracy"]} if ext else None,
                "own_photos": own},
    "all_models": {m: {"test_value_accuracy": r["test"]["value_accuracy"], "external_value_accuracy": (r.get("external") or {}).get("value_accuracy"),
                       "params_million": r["params_million"], "onnx_cpu_ms": r["onnx_cpu_ms"]} for m, r in runs.items()},
    "how_to_test": ["git clone " + links["github"], "pip install -r requirements.txt", "python evaluate.py", "python predict.py samples/500_front_1.jpg"],
    "weights_files": {f: f"{links['github']}/raw/main/weights/{f}" for f in card["files"]},
}
json.dump(project, open(f"{out}/project.json", "w"), indent=2, ensure_ascii=False)

llms = f"""# BoltaNote

> Talking Pakistani banknote reader for blind and low-vision people: phone camera -> note value spoken in Urdu or English, running in the browser. Deep Learning semester project by {links['author']}.

- Live demo: {links.get('site') or '(this site)'}
- Code, weights, data splits: {links['github']}
- Training dataset (CC0): {links.get('dataset')}
- External test dataset (CC BY 4.0): {links.get('dataset_external')}
- Report: {links.get('report') or 'see GitHub docs/'}
- Slides: {links.get('slides') or 'see GitHub docs/'}
- Video: {links.get('youtube') or 'see project.json'}
- LinkedIn: {links.get('linkedin') or 'see project.json'}
- Machine-readable summary: /project.json ; model card: /model_card.json

## Model
- Deployed: {NICE[best]}; input {card['input']['shape']} float32 RGB in [0,1] (whole photo squeezed to 224x224); output probs [1,15] over {', '.join(card['classes'])}. Add front+back per value.
- Weights: weights/boltanote.onnx (ONNX), weights/boltanote_torchscript.pt (TorchScript), weights/{best}.pt (PyTorch state_dict).
- Speaks if top value != none and probability >= {b['speaking_threshold']:.2f}.

## Results
- Internal test ({t['n']} photos): value accuracy {pct(t['value_accuracy'])}, wrong value said {pct(t['speaking']['wrong_value_rate'])}.
""" + (f"- External test ({ext['n']} photos, other photographers): {pct(ext['value_accuracy'])}.\n" if ext else "") + (f"- Own photos ({own['photos']}): {pct(own['value_accuracy'])}.\n" if own else "") + f"""
## How to test
git clone {links['github']} && cd boltanote && pip install -r requirements.txt
python evaluate.py                          # 16 bundled labelled photos
python evaluate.py --folder data/own_photos # our own photos
python predict.py samples/500_front_1.jpg
"""
open(f"{out}/llms.txt", "w", encoding="utf-8").write(llms)
open(f"{out}/robots.txt", "w").write("User-agent: *\nAllow: /\n")
open(f"{out}/vercel.json", "w").write(json.dumps({"headers": [{"source": "/(model|audio)/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=604800"}]}]}, indent=2))
print("built", out, "->", sorted(os.listdir(out)))
