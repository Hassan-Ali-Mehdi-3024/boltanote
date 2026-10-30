"""Build the static website in web/ from site/ + results + links.json (run after report.py).

    python scripts/build_site.py

Everything a crawler needs is written into the HTML itself, and repeated in web/project.json,
web/llms.txt and web/llms-full.txt. Every part of the project is linked twice, on separate hosts.
"""

import csv
import datetime
import glob
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from xml.sax.saxutils import escape as xml

from PIL import Image

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

SITE, PAGES, REL, GH = links["site"].rstrip("/"), links["pages"].rstrip("/"), links["release"].rstrip("/"), links["github"]
RAW = f"{GH}/raw/main"
TODAY = datetime.date.today().isoformat()
NAME = "Pakistani Banknote Reader"
TITLE = "Pakistani Banknote Reader for Blind People: Speaks in Urdu"
DESC = "Free web app that reads Pakistani banknotes through the phone camera and says the value aloud in Urdu or English. No install, works offline, made for blind users."
assert len(TITLE) <= 60 and len(DESC) <= 165, (len(TITLE), len(DESC))
NOTE = links["linkedin_note"]
FILM = "video/pakistani-banknote-reader-film.mp4"
OG = f"{SITE}/og.png"
OG_ALT = "Pakistani Banknote Reader: a phone reading a Rs 500 note and showing the value in Urdu."

out = "web"
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(f"{out}/model")
for f in ("style.css", "app.js", "404.html", "og.png"):
    shutil.copy(f"site/{f}", out)
for d in ("audio", "vendor", "icons", "video"):
    shutil.copytree(f"site/{d}", f"{out}/{d}")
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
os.makedirs(f"{out}/notebooks"); shutil.copy("notebooks/BoltaNote_train.ipynb", f"{out}/notebooks")
os.makedirs(f"{out}/data"); shutil.copy("data/splits.csv", f"{out}/data")
os.makedirs(f"{out}/downloads")
subprocess.run(["git", "archive", "HEAD", "--format=zip", "-o", f"{out}/downloads/boltanote-source.zip", "--", ".", ":!weights", ":!web", ":!site/audio", ":!site/video", ":!site/vendor", ":!docs/slides"], check=True)
samples = [{"file": r["file"], "value": r["value"], "side": r["class"].split("_")[1] if "_" in r["class"] else None} for r in csv.DictReader(open("samples/labels.csv"))]
json.dump(samples, open(f"{out}/samples/samples.json", "w"))
open(f"{out}/favicon.svg", "w").write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#ffd400"/><rect x="6" y="10" width="20" height="12" rx="2" fill="none" stroke="#111" stroke-width="2"/><circle cx="16" cy="16" r="3" fill="#111"/></svg>')
open(f"{out}/{links['indexnow_key']}.txt", "w").write(links["indexnow_key"])

rows = []
for m, r in runs.items():
    rt, re_ = r["test"], r.get("external")
    rows.append(f'<tr{" class=deployed" if m == best else ""}><td>{NICE[m]}{" <b>(deployed)</b>" if m == best else ""}</td><td>{r["params_million"]}M</td>'
                f'<td>{pct(rt["value_accuracy"])}</td><td>{pct(rt["speaking"]["wrong_value_rate"])}</td><td>{pct(rt["speaking"]["coverage_on_notes"])}</td>'
                f'<td>{pct(re_["value_accuracy"]) if re_ else "–"}</td><td>{r["onnx_cpu_ms"]:.0f}</td></tr>')

# key, title, what it is, every copy (first = the main one). Copies live on separate hosts wherever the file size allows.
COPIES = [
    ("site", "This website", "the live reader and this page (Vercel, GitHub Pages)", [SITE, f"{PAGES}/"]),
    ("github", "GitHub repository", "code, weights, splits, notebook; second copy is the source code as a zip", [GH, f"{SITE}/downloads/boltanote-source.zip"]),
    ("youtube", "Video presentation (YouTube)", "under four minutes, in my own voice; also on this site, and as a 4K download", [links["youtube"], f"{SITE}/{FILM}", f"{REL}/Pakistani_Banknote_Reader_film_4K60.mp4"]),
    ("report", "Report (PDF)", "full write-up", [links["report"], f"{RAW}/docs/BoltaNote_report.pdf"]),
    ("report_web", "Report (web page)", "the same report, as a page", [f"{SITE}/report/", f"{PAGES}/report/"]),
    ("slides", "Presentation slides (PDF)", "the deck", [links["slides"], f"{RAW}/docs/slides/BoltaNote_slides.pdf"]),
    ("weights_onnx", "Model weights (ONNX)", "the deployed model, weights/boltanote.onnx", [links["weights_onnx"], f"{SITE}/model/boltanote.onnx"]),
    ("weights_torchscript", "Model weights (TorchScript)", "weights/boltanote_torchscript.pt", [links["weights_torchscript"], f"{REL}/boltanote_torchscript.pt"]),
    ("weights_pt", "PyTorch checkpoints", "weights/*.pt, all three models", [links["weights_pt"], links["release_page"]]),
    ("notebook", "Training notebook", "Colab, all outputs", [links["notebook"], f"{SITE}/notebooks/BoltaNote_train.ipynb"]),
    ("splits", "Data splits (CSV)", "every file and its split", [f"{GH}/blob/main/data/splits.csv", f"{SITE}/data/splits.csv"]),
    ("dataset", "Main dataset", "Pakistan Image Currency Dataset, CC0; Kaggle and an unchanged mirror", [links["dataset"], f"{REL}/pakistan-image-currency-dataset.zip"]),
    ("dataset_external", "External test set", "Roboflow currency notes, CC BY 4.0; Kaggle and an unchanged mirror", [links["dataset_external"], f"{REL}/pakistani-currency-notes-roboflow.zip"]),
    ("dataset_none", "\"No note\" photos (Imagenette)", "fast.ai's ImageNet subset; project page and download", ["https://github.com/fastai/imagenette", "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"]),
    ("model_card", "Model card (JSON)", "inputs, outputs, threshold, metrics", [f"{SITE}/model_card.json", f"{RAW}/model_card.json"]),
    ("project_json", "Project summary (JSON)", "every result and link, for machines", [f"{SITE}/project.json", f"{PAGES}/project.json"]),
]
if own and os.path.isdir("data/own_photos"):
    COPIES.insert(11, ("own_photos", "Our own photos", "real-world test set, in the repo", [links["own_photos"], links["release_page"]]))


def link_card(key, title, desc, urls):
    copies = "".join(f'<a href="{html.escape(u)}"><b>Copy {i}</b> {html.escape(u)}</a>' for i, u in enumerate(urls, 1))
    return f'<div class="link" id="l-{key}"><h3>{html.escape(title)}</h3><p>{html.escape(desc)}</p>{copies}</div>'


cards = [link_card(*c) for c in COPIES]
cards.insert(3, f'<div class="link note" id="linkedin"><h3>LinkedIn post</h3><p>{html.escape(NOTE)}</p></div>')

limits = [
    "Trained on photos of a few physical notes per value, mostly on similar indoor backgrounds; very worn, torn or taped notes are under-represented.",
    "Knows the 2005–2008 note designs. The State Bank of Pakistan's new series will need new photos and retraining.",
    "Reads one note at a time; two overlapping notes give one answer.",
    "Not a counterfeit detector.",
    f"External test has only {dedupe['roboflow_kept']} genuinely new photos after removing near-copies, so its confidence interval is wide.",
    "Not yet tested with blind users; that is the next step.",
]
n_val = count("val")
ext_silent = ext["n"] - round(ext["speaking"]["coverage_on_notes"] * ext["n"]) if ext else 0
ext_wrong = round(ext["speaking"]["wrong_value_rate"] * ext["n"]) if ext else 0
if own:
    own_text = (f"{own['photos']} photos of our own notes, taken with a phone in real rooms (folded, in hand, dim light, different backgrounds), "
                f"never used for training or for any choice. Sorted by value in <a href=\"{links['github']}/tree/main/data/own_photos\">data/own_photos</a>.")
    own_fig = '<figure><img src="figures/own_photos.png" alt="Share of our own photos read correctly, per note value" loading="lazy"><figcaption>Our own photos: per-value accuracy.</figcaption></figure>'
    own_card = f'<div class="card"><h3>4. Our own photos: the real test</h3><p class="meta">{own_text}</p></div>'
    own_row = f'<tr><td>Our own photos</td><td>{own["photos"]}</td><td>does it work on real notes, in real rooms?</td></tr>'
    fourth = f'<div class="card stat"><div class="num">{pct(own["value_accuracy"])}</div><div class="lbl">on our own photos ({own["photos"]} photos)</div></div>'
else:
    own_fig, own_card, own_row = "", "", ""
    fourth = f'<div class="card stat"><div class="num">{ext_wrong} wrong</div><div class="lbl">amounts spoken on those {ext["n"]} photos (silent on {ext_silent} of them)</div></div>'

sentence = (f"The deployed model is <b>{NICE[best]}</b>, chosen on validation photos. On {t['n']} unseen test photos it names the note value correctly "
            f"{pct(t['value_accuracy'])} of the time; at the speaking threshold it would say a wrong value on {pct(t['speaking']['wrong_value_rate'])} of photos "
            f"and speaks on {pct(t['speaking']['coverage_on_notes'])} of the photos that show a note.")
if ext:
    sentence += f" On {ext['n']} photos by other photographers it reaches {pct(ext['value_accuracy'])}, and at the speaking threshold it says {'no wrong amount' if not ext_wrong else f'{ext_wrong} wrong amounts'} (it stays silent on {ext_silent})."
if own:
    sentence += (f" On our own {own['photos']} photos it reaches {pct(own['value_accuracy'])}, saying a wrong value on {pct(own['wrong_value_said'])} of them.")

# Answers come first and stand alone, so a search or AI engine can quote any one of them as it is.
FAQ = [
    ("What is the Pakistani Banknote Reader?", "A free web app for blind and low-vision people in Pakistan. You point a phone camera at a banknote and it says the value out loud in Urdu or English. It runs in the browser, so there is no app to install."),
    ("How do I use it?", f"Open {SITE} in a phone browser, tap Start camera, and hold a note in front of the camera. It says the value as soon as it is sure. With no camera, you can upload a photo or tap a test photo."),
    ("Which Pakistani notes can it read?", "All seven notes in circulation: Rs 10, 20, 50, 100, 500, 1,000 and 5,000, front or back, including folded or upside down. It knows the 2005–2008 designs, not the State Bank's new series."),
    ("Is it free, and does it need an account?", "It is free, with no account and no install. The code is open source under the MIT licence."),
    ("Does it work offline?", "Yes, after the first visit. The model, the runtime and the voice clips are saved on the phone, so it keeps working and reopens with no internet."),
    ("Are my camera pictures uploaded anywhere?", "No. Every picture is processed on the phone, inside the browser. Nothing is sent to a server."),
    ("How accurate is it?", f"It names the value correctly on {pct(t['value_accuracy'])} of {t['n']} unseen photos from the same dataset, and {pct(ext['value_accuracy'])} of {ext['n']} photos taken by other people. On those {ext['n']} photos it spoke {'no wrong amount' if not ext_wrong else f'{ext_wrong} wrong amounts'} and stayed silent on {ext_silent}. The first number mostly shows how alike the dataset's photos are; the second is closer to real use, but {ext['n']} photos is a small test."),
    ("When does it stay silent?", f"It speaks only when it is at least {pct(b['speaking_threshold'], 0)} sure and three frames in a row agree, because saying a wrong amount is worse than saying nothing. The {b['speaking_threshold']:.2f} threshold keeps wrong answers to at most 1 in 500 on a stress test of {n_val * 4:,} hard, augmented validation photos."),
    ("Does it work with screen readers?", "It is built for them: results are announced to TalkBack and VoiceOver through a live region, every control is a large button, and on Android it also vibrates a code for each note. It has not yet been tested with blind users; that is the next step."),
    ("Can it detect counterfeit notes?", "No. It reads the value of a note and does not check whether the note is genuine. It also reads only one note at a time."),
    ("Which model does it use?", f"{NICE[best]}, pretrained on ImageNet and fine-tuned to 15 classes: seven notes, front and back, plus \"no note\". It runs with ONNX Runtime Web and takes about {b['onnx_cpu_ms']:.0f} ms per photo on a CPU. It was compared with a CNN trained from scratch and MobileNetV4."),
    ("What data was it trained on?", f"The Pakistan Image Currency Dataset (CC0) of 3,611 phone photos, plus 1,000 Imagenette photos for the \"no note\" class. A hash check found {dedupe['roboflow_dropped']} of {dedupe['roboflow_total']} photos in a second public dataset were near-copies of the first, so only the {dedupe['roboflow_kept']} new ones were used as the external test."),
    ("Who made it?", f"{links['author']}, as a Deep Learning semester project at Superior University, Lahore (BSCS-7A), under Dr. Munir Ahmad, {links.get('month', 'November 2026')}."),
    ("Where are the code, weights and report?", f"The code, weights and data splits are at {GH}, and the report is at {SITE}/report.pdf. Every link, with a second copy, is in the Everything, linked section of this page."),
    ("Is there a LinkedIn post?", f"No. {NOTE}"),
]
autolink = lambda s: re.sub(r"https?://[^\s,;]+?(?=[.,;]?(?:\s|$))", lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', html.escape(s, quote=False))
faq_html = "".join(f"<div><h3>{html.escape(q)}</h3><p>{autolink(a)}</p></div>" for q, a in FAQ)

transcript = open("site/video/transcript.txt", encoding="utf-8").read().strip()
transcript_html = "".join(f"<p>{html.escape(p.strip())}</p>" for p in transcript.split("\n\n"))
video_desc = (f'The whole project in my own voice: the problem, the data, the honest numbers and a demo. Also on <a href="{links["youtube"]}">YouTube</a>, '
              f'or <a href="{REL}/Pakistani_Banknote_Reader_film_4K60.mp4">download it in 4K</a>.')

author = {"@type": "Person", "@id": f"{SITE}/#author", "name": links["author"], "url": GH.rsplit("/", 1)[0], "sameAs": [GH.rsplit("/", 1)[0], links["youtube_channel"]],
          "affiliation": {"@type": "CollegeOrUniversity", "name": "Superior University, Lahore"}}
jsonld = {"@context": "https://schema.org", "@graph": [
    {"@type": "WebSite", "@id": f"{SITE}/#website", "url": f"{SITE}/", "name": NAME, "inLanguage": "en", "publisher": {"@id": f"{SITE}/#author"}},
    {"@type": "WebPage", "@id": f"{SITE}/#page", "url": f"{SITE}/", "name": TITLE, "description": DESC, "isPartOf": {"@id": f"{SITE}/#website"}, "inLanguage": "en", "datePublished": "2026-10-28", "dateModified": TODAY,
     "about": {"@id": f"{SITE}/#app"}, "mainEntity": {"@id": f"{SITE}/#app"}, "primaryImageOfPage": OG, "speakable": {"@type": "SpeakableSpecification", "cssSelector": ["#summary", ".faq p"]},
     "hasPart": [{"@type": "Report", "name": f"{NAME}: project report", "url": links["report"], "encodingFormat": "application/pdf", "author": {"@id": f"{SITE}/#author"}},
                 {"@type": "PresentationDigitalDocument", "name": f"{NAME}: slides", "url": links["slides"], "encodingFormat": "application/pdf", "author": {"@id": f"{SITE}/#author"}}]},
    {"@type": "WebApplication", "@id": f"{SITE}/#app", "name": NAME, "alternateName": ["Pakistani currency reader for blind people", "Talking banknote reader"], "url": f"{SITE}/", "description": DESC,
     "applicationCategory": "UtilitiesApplication", "applicationSubCategory": "Accessibility", "operatingSystem": "Any, in a web browser", "browserRequirements": "Needs a modern browser with WebAssembly; the camera is optional",
     "inLanguage": ["en", "ur"], "isAccessibleForFree": True, "offers": {"@type": "Offer", "price": "0", "priceCurrency": "PKR"}, "countriesSupported": "PK",
     "featureList": ["Reads all seven Pakistani notes, Rs 10 to Rs 5,000, front or back", "Speaks the value in Urdu or English with a recorded voice", "Stays silent when unsure", "Works offline after the first visit", "Announces results to screen readers", "Vibrates a code for each note on Android", "Processes every picture on the phone; nothing is uploaded"],
     "accessibilityFeature": ["ARIA", "largePrint", "synthesizedSpeech", "highContrastDisplay"], "audience": {"@type": "PeopleAudience", "audienceType": "Blind and low-vision people in Pakistan"},
     "screenshot": [f"{SITE}/figures/site_phone.png", f"{SITE}/figures/site_desktop.png"], "author": {"@id": f"{SITE}/#author"}, "sameAs": GH, "softwareHelp": f"{SITE}/#faq"},
    {"@type": "SoftwareSourceCode", "@id": f"{SITE}/#code", "name": NAME, "description": "Talking Pakistani banknote reader for blind people (MobileNet, ONNX Runtime Web).", "codeRepository": GH, "programmingLanguage": ["Python", "JavaScript"], "runtimePlatform": ["PyTorch", "ONNX Runtime Web"],
     "license": "https://opensource.org/licenses/MIT", "author": {"@id": f"{SITE}/#author"}, "targetProduct": {"@id": f"{SITE}/#app"},
     "isBasedOn": [{"@type": "Dataset", "name": d["name"], "url": d["url"], "license": d["license"]} for d in card["datasets"]]},
    {"@type": "VideoObject", "@id": f"{SITE}/#video", "name": "Pakistani Banknote Reader for blind people (Deep Learning project)", "description": "A talking banknote reader for blind people in Pakistan: the problem, the data, the honest numbers and a demo, in the maker's own voice.",
     "thumbnailUrl": OG, "uploadDate": links["video_date"], "duration": f"PT{links['video_seconds'] // 60}M{links['video_seconds'] % 60}S", "contentUrl": f"{SITE}/{FILM}", "embedUrl": f"https://www.youtube.com/embed/{links['youtube'].rsplit('/', 1)[1]}", "sameAs": links["youtube"],
     "inLanguage": "en", "transcript": transcript, "author": {"@id": f"{SITE}/#author"}},
    {"@type": "FAQPage", "@id": f"{SITE}/#faq", "mainEntity": [{"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in FAQ]},
    author,
]}
HEAD = f"""<title>{html.escape(TITLE)}</title>
<meta name="description" content="{html.escape(DESC)}">
<link rel="canonical" href="{SITE}/">
<meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1">
<meta name="author" content="{html.escape(links['author'])}">
<meta name="application-name" content="{NAME}">
<meta name="apple-mobile-web-app-title" content="Banknote Reader">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta property="og:type" content="website">
<meta property="og:site_name" content="{NAME}">
<meta property="og:title" content="{html.escape(TITLE)}">
<meta property="og:description" content="{html.escape(DESC)}">
<meta property="og:url" content="{SITE}/">
<meta property="og:locale" content="en_PK">
<meta property="og:locale:alternate" content="ur_PK">
<meta property="og:image" content="{OG}">
<meta property="og:image:width" content="1280">
<meta property="og:image:height" content="720">
<meta property="og:image:alt" content="{html.escape(OG_ALT)}">
<meta property="og:video" content="{SITE}/{FILM}">
<meta property="og:video:type" content="video/mp4">
<meta property="og:video:width" content="1280">
<meta property="og:video:height" content="720">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{html.escape(TITLE)}">
<meta name="twitter:description" content="{html.escape(DESC)}">
<meta name="twitter:image" content="{OG}">
<meta name="twitter:image:alt" content="{html.escape(OG_ALT)}">"""
fill = {
    "GITHUB": links["github"], "AUTHOR": html.escape(links["author"]), "MONTH": links.get("month", "November 2026"), "TODAY": TODAY, "HEAD": HEAD,
    "THRESHOLD_PCT": pct(b["speaking_threshold"], 0), "TEST_ACC": pct(t["value_accuracy"]), "WRONG": pct(t["speaking"]["wrong_value_rate"]),
    "EXT_ACC": pct(ext["value_accuracy"]) if ext else "–", "OWN_FIGURE": own_fig, "OWN_CARD": own_card, "OWN_ROW": own_row, "FOURTH_STAT": fourth,
    "OWN_CREDIT": ", and our own photos" if own else "", "DEDUPE_GRID": "two" if own else "one",
    "DATA_H2": "Data: public photos, checked for copies, plus our own" if own else "Data: public photos, checked for copies",
    "DATA_SUB": "Three public datasets and one we collected ourselves." if own else "Three public datasets.",
    "N_TRAIN": f"{count('train'):,}", "N_VAL": f"{count('val'):,}", "N_TEST": f"{count('test'):,}",
    "RF_TOTAL": dedupe["roboflow_total"], "RF_DROPPED": dedupe["roboflow_dropped"], "RF_KEPT": dedupe["roboflow_kept"],
    "RESULTS_ROWS": "\n    ".join(rows), "RESULTS_SENTENCE": sentence, "LINK_CARDS": "".join(cards), "FAQ": faq_html,
    "VIDEO_DESC": video_desc, "TRANSCRIPT": transcript_html,
    "LIMITS": "".join(f"<li>{l}</li>" for l in limits), "JSONLD": json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/"),
    "OWN_CMD": "Our own photos: <code>python evaluate.py --folder data/own_photos</code>. " if own else "",
}
page = open("site/index.html", encoding="utf-8").read()
for k, v in fill.items():
    page = page.replace("{{" + k + "}}", str(v))
assert "{{" not in page, [l for l in page.splitlines() if "{{" in l]


def sized(m):
    """Give each figure its real size, so the page doesn't jump while images load."""
    w, h = Image.open(f"{out}/{m.group(1)}").size
    return f'<img src="{m.group(1)}" width="{w}" height="{h}" decoding="async"'


page = re.sub(r'<img src="(figures/[^"]+)"', sized, page)
open(f"{out}/index.html", "w", encoding="utf-8").write(page)

# The report page gets its own title, description and canonical address.
report_desc = "Full report on the Pakistani Banknote Reader: data, duplicate check, three models compared, a calibrated speaking threshold, results and limits."
report_ld = {"@context": "https://schema.org", "@type": "Report", "name": f"{NAME}: project report", "headline": "A talking Pakistani banknote reader for blind people, tested beyond its own dataset", "description": report_desc, "url": f"{SITE}/report/", "datePublished": "2026-10-28", "dateModified": TODAY,
             "author": {"@type": "Person", "name": links["author"], "url": GH.rsplit("/", 1)[0]}, "encoding": {"@type": "MediaObject", "contentUrl": links["report"], "encodingFormat": "application/pdf"}, "about": {"@type": "WebApplication", "name": NAME, "url": f"{SITE}/"}, "inLanguage": "en"}
rp = f"{out}/report/index.html"
if os.path.exists(rp):
    r = open(rp, encoding="utf-8").read()
    assert "<title>Pakistani Banknote Reader: report</title>" in r
    r = r.replace("<title>Pakistani Banknote Reader: report</title>", f"""<title>Pakistani Banknote Reader: Full Project Report</title>
<meta name="description" content="{report_desc}">
<link rel="canonical" href="{SITE}/report/">
<meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1">
<meta name="author" content="{html.escape(links['author'])}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="{NAME}">
<meta property="og:title" content="Pakistani Banknote Reader: Full Project Report">
<meta property="og:description" content="{report_desc}">
<meta property="og:url" content="{SITE}/report/">
<meta property="og:image" content="{OG}">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="alternate" type="application/pdf" href="../report.pdf" title="This report as a PDF">
<script type="application/ld+json">{json.dumps(report_ld, ensure_ascii=False)}</script>""", 1)
    r = re.sub(r"<img src=\"(figures/[^\"]+)\">(<figcaption>(?:Figure \d+\. )?([^<]*)</figcaption>)?", lambda m: f'<img src="{m.group(1)}" alt="{html.escape(m.group(3) or "Figure from the report")}" loading="lazy">{m.group(2) or ""}', r)
    open(rp, "w", encoding="utf-8").write(r)

project = {
    "project": NAME, "task": card["task"], "author": links["author"], "course": "Deep Learning, Superior University (Dr. Munir Ahmad)", "updated": TODAY,
    "links": {key: urls[0] for key, _, _, urls in COPIES},
    "copies": {key: {"title": title, "what": desc, "urls": urls} for key, title, desc, urls in COPIES},
    "linkedin": {"posted": False, "note": NOTE},
    "datasets": card["datasets"], "classes": card["classes"], "values": card["values"], "input": card["input"], "outputs": card["outputs"],
    "speaking": {k: card["speaking"][k] for k in ("rule", "threshold", "threshold_chosen_on")},
    "deployed_model": best,
    "results": {"internal_test": {"photos": t["n"], "value_accuracy": t["value_accuracy"], "wrong_value_rate": t["speaking"]["wrong_value_rate"], "coverage_on_notes": t["speaking"]["coverage_on_notes"]},
                "external_test": {"photos": ext["n"], "value_accuracy": ext["value_accuracy"], "wrong_value_rate": ext["speaking"]["wrong_value_rate"], "coverage_on_notes": ext["speaking"]["coverage_on_notes"]} if ext else None,
                "own_photos": own},
    "all_models": {m: {"test_value_accuracy": r["test"]["value_accuracy"], "external_value_accuracy": (r.get("external") or {}).get("value_accuracy"),
                       "params_million": r["params_million"], "onnx_cpu_ms": r["onnx_cpu_ms"]} for m, r in runs.items()},
    "how_to_test": ["git clone " + links["github"], "pip install -r requirements.txt", "python evaluate.py", "python predict.py samples/500_front_1.jpg"],
    "weights_files": {f: f"{links['github']}/raw/main/weights/{f}" for f in card["files"]},
    "weights_files_copies": {f: f"{REL}/{f}" for f in card["files"]},
    "faq": [{"question": q, "answer": a} for q, a in FAQ],
}
json.dump(project, open(f"{out}/project.json", "w"), indent=2, ensure_ascii=False)

copy_lines = "\n".join(f"- [{title}]({urls[0]}): {desc}. Second copy: {urls[1]}" + (f" Third copy: {urls[2]}" if len(urls) > 2 else "") for _, title, desc, urls in COPIES)
llms = f"""# {NAME}

> A free web app for blind and low-vision people in Pakistan: point a phone camera at a banknote and it says the value out loud in Urdu or English. It runs in the browser, works offline after the first visit, and uploads nothing. Deep Learning semester project by {links['author']}, Superior University, Lahore, under Dr. Munir Ahmad.

Full text for AI engines: {SITE}/llms-full.txt

## Links (each with a second copy)
{copy_lines}
- LinkedIn post: none. {NOTE}

## Model
- Deployed: {NICE[best]}; input {card['input']['shape']} float32 RGB in [0,1] (whole photo squeezed to 224x224); output probs [1,15] over {', '.join(card['classes'])}. Add front+back per value.
- Weights: weights/boltanote.onnx (ONNX), weights/boltanote_torchscript.pt (TorchScript), weights/{best}.pt (PyTorch state_dict).
- Speaks if top value != none and probability >= {b['speaking_threshold']:.2f}.

## Results
- Internal test ({t['n']} photos): value accuracy {pct(t['value_accuracy'])}, wrong value said {pct(t['speaking']['wrong_value_rate'])}.
""" + (f"- External test ({ext['n']} photos, other photographers): {pct(ext['value_accuracy'])}; {ext_wrong} wrong amounts spoken, silent on {ext_silent}.\n" if ext else "") + (f"- Own photos ({own['photos']}): {pct(own['value_accuracy'])}.\n" if own else "") + f"""
## How to test
git clone {links['github']} && cd boltanote && pip install -r requirements.txt
python evaluate.py                          # 16 bundled labelled photos
""" + ("python evaluate.py --folder data/own_photos # our own photos\n" if own else "") + """python predict.py samples/500_front_1.jpg
"""
open(f"{out}/llms.txt", "w", encoding="utf-8").write(llms)

report_text = ""
if os.path.exists(rp):
    body = open(rp, encoding="utf-8").read()
    body = re.sub(r"(?s)<(style|script|head)\b.*?</\1>", "", body)
    report_text = html.unescape(re.sub(r"[ \t]+", " ", re.sub(r"(?s)</(p|h\d|li|tr|div|figcaption)>", "\n", body).replace("</td>", " | ")))
    report_text = "\n".join(l.strip() for l in re.sub(r"<[^>]+>", "", report_text).splitlines() if l.strip())
llms_full = llms + "\n## Questions and answers\n" + "".join(f"\n### {q}\n{a}\n" for q, a in FAQ) + f"\n## Film transcript\n\n{transcript}\n\n## Full report\n\n{report_text}\n"
open(f"{out}/llms-full.txt", "w", encoding="utf-8").write(llms_full)

AI_BOTS = ["GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot", "Claude-User", "Claude-SearchBot", "PerplexityBot", "Perplexity-User", "Google-Extended", "Applebot-Extended", "Bingbot", "Amazonbot", "DuckAssistBot", "CCBot", "meta-externalagent", "cohere-ai", "MistralAI-User"]
open(f"{out}/robots.txt", "w").write("User-agent: *\nAllow: /\n\n# Answer engines and AI crawlers are welcome. Start with /llms.txt.\n" + "".join(f"User-agent: {x}\n" for x in AI_BOTS) + f"Allow: /\n\nSitemap: {SITE}/sitemap.xml\n")
video_xml = (f"<video:video><video:thumbnail_loc>{OG}</video:thumbnail_loc><video:title>{xml(jsonld['@graph'][4]['name'])}</video:title><video:description>{xml(jsonld['@graph'][4]['description'])}</video:description>"
             f"<video:content_loc>{SITE}/{FILM}</video:content_loc><video:player_loc>{jsonld['@graph'][4]['embedUrl']}</video:player_loc><video:duration>{links['video_seconds']}</video:duration><video:publication_date>{links['video_date']}</video:publication_date></video:video>")
pages = [(f"{SITE}/", video_xml), (f"{SITE}/report/", ""), (f"{SITE}/report.pdf", ""), (f"{SITE}/slides.pdf", "")]
open(f"{out}/sitemap.xml", "w", encoding="utf-8").write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:video="http://www.google.com/schemas/sitemap-video/1.1">\n'
    + "".join(f"<url><loc>{u}</loc><lastmod>{TODAY}</lastmod>{x}</url>\n" for u, x in pages) + "</urlset>\n")
json.dump({"name": NAME, "short_name": "Banknote Reader", "description": DESC, "lang": "en", "start_url": "./", "scope": "./", "display": "standalone", "background_color": "#fbfaf6", "theme_color": "#ffd400", "categories": ["accessibility", "utilities"],
           "icons": [{"src": "icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"}, {"src": "icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}]},
          open(f"{out}/site.webmanifest", "w"), indent=2)
week = [{"key": "Cache-Control", "value": "public, max-age=604800"}]
open(f"{out}/vercel.json", "w").write(json.dumps({
    "headers": [
        {"source": "/(.*)", "headers": [{"key": "X-Content-Type-Options", "value": "nosniff"}, {"key": "Permissions-Policy", "value": "camera=(self)"}]},
        {"source": "/(model|audio|vendor|samples|icons|video)/(.*)", "headers": week},
        {"source": "/sw.js", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
        {"source": "/(project.json|model_card.json|llms.txt|llms-full.txt|sitemap.xml|robots.txt|report.pdf|slides.pdf)", "headers": [{"key": "Access-Control-Allow-Origin", "value": "*"}]},
        {"source": "/model/(.*)", "headers": [{"key": "Access-Control-Allow-Origin", "value": "*"}]},
        {"source": "/(llms.txt|llms-full.txt)", "headers": [{"key": "Content-Type", "value": "text/plain; charset=utf-8"}]},
    ],
    "redirects": [
        {"source": "/linkedin", "destination": "/#linkedin", "permanent": False},
        {"source": "/video", "destination": "/#film", "permanent": False},
        {"source": "/slides", "destination": "/slides.pdf", "permanent": False},
        {"source": "/github", "destination": GH, "permanent": False},
    ],
}, indent=2))

# The offline worker saves exactly what the reader needs; its version changes whenever any of that changes.
core = ["app.js", "style.css", "favicon.svg", "site.webmanifest", "model_card.json", "samples/samples.json", "model/boltanote.onnx", "icons/icon-192.png"]
core += [os.path.relpath(p, out) for pat in ("vendor/ort/*", "audio/*/*.mp3", "samples/*.jpg") for p in sorted(glob.glob(f"{out}/{pat}"))]
digest = hashlib.sha256(open(f"{out}/index.html", "rb").read())
for f in core:
    digest.update(open(f"{out}/{f}", "rb").read())
sw = open("site/sw.js", encoding="utf-8").read().replace("{{VERSION}}", digest.hexdigest()[:12]).replace("{{CORE}}", json.dumps(["./", *core]))
open(f"{out}/sw.js", "w", encoding="utf-8").write(sw)
print("built", out, "->", sorted(os.listdir(out)))
