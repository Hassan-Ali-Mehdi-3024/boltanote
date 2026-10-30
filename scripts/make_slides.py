"""Build the presentation: docs/slides/BoltaNote_slides.{pptx,pdf,mp4} + voiceover script.
    python scripts/make_slides.py [--voice en-US-AndrewMultilingualNeural]"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "tools"))
from slides import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="en-US-AndrewMultilingualNeural")
a = ap.parse_args()
links = json.load(open("links.json"))
S = json.load(open("results/summary.json"))
D = json.load(open("results/dedupe.json"))
runs, best = S["models"], S["deployed_model"]
B = runs[best]
NICE = {"cnn_scratch": "CNN from scratch", "mobilenetv3_large": "MobileNetV3-Large", "mobilenetv4_conv_medium": "MobileNetV4-Conv-Medium"}
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
ext, own = B.get("external"), B.get("own")
site = links.get("site") or "boltanote.vercel.app"
gh = links["github"].replace("https://", "")
author = links["author"]

CSS = """
body{background:#e9e6dc}
.slide{background:#fbfaf6;color:#111;padding:80px 120px 110px;display:flex;flex-direction:column;justify-content:center}
.slide:after{content:"Pakistani Banknote Reader";position:absolute;left:110px;bottom:44px;font-size:22px;color:#8a8a80;font-family:Inter}
.bar{position:absolute;left:0;top:0;bottom:0;width:22px;background:#ffd400}
h1{font-size:96px;line-height:1.02;margin:0 0 24px;letter-spacing:-.03em;font-weight:900}
h2{font-size:72px;line-height:1.08;margin:0 0 50px;letter-spacing:-.02em;font-weight:850}
.ur{font-family:"Noto Nastaliq Urdu",serif;direction:rtl;line-height:2.1}
.lead{font-size:44px;color:#3a3a36;line-height:1.4;max-width:1500px}
ul{margin:0;padding-left:48px} li{font-size:44px;line-height:1.42;margin:14px 0;color:#222}
li b{color:#000}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:60px;align-items:center}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:36px}
.stat{background:#fff;border:3px solid #ece8da;border-radius:28px;padding:36px 40px}
.stat .n{font-size:128px;font-weight:900;letter-spacing:-.03em;line-height:1}
.stat .l{font-size:34px;color:#555;margin-top:14px;line-height:1.3}
.yellow{background:#ffd400;border-color:#ffd400}
.green{color:#0a5c36}
.red{color:#b42318}
img.fig{width:100%;max-height:820px;object-fit:contain;border-radius:18px;border:3px solid #ece8da;background:#fff}
.muted{color:#6b6b63;font-size:28px}
.tag{display:inline-block;background:#111;color:#ffd400;font-weight:800;font-size:26px;padding:8px 18px;border-radius:12px;margin-bottom:28px;letter-spacing:.04em}
table{border-collapse:collapse;font-size:38px;width:100%} td,th{padding:20px 22px;border-bottom:3px solid #ece8da;text-align:left} th{font-size:24px;color:#777;text-transform:uppercase;letter-spacing:.06em}
tr.dep td{background:#fff4b8}
.phone{width:430px;border-radius:48px;border:14px solid #111;box-shadow:0 30px 60px rgba(0,0,0,.25)}
"""

own_line = (f"On <b>{own['n']}</b> photos of my own notes: <b>{pct(own['value_accuracy'])}</b>" if own else "My own photos: being collected")
rows = "".join(f"<tr class='{'dep' if m == best else ''}'><td>{NICE[m]}</td><td>{r['params_million']}M</td><td>{pct(r['test']['value_accuracy'])}</td>"
               f"<td>{pct(r['external']['value_accuracy']) if r.get('external') else '–'}</td>" + (f"<td>{pct(r['own']['value_accuracy']) if r.get('own') else '–'}</td>" if own else "")
               + f"<td>{r['onnx_cpu_ms']:.0f} ms</td></tr>" for m, r in runs.items())
own_th = "<th>My photos</th>" if own else ""
third = (f"""<div class=stat><div class="n green">{pct(own['value_accuracy'], 0)}</div><div class=l>my own {own['n']} photos</div></div>""" if own else
         f"""<div class=stat><div class="n green">{pct(ext['speaking']['wrong_value_rate'], 0) if ext else '–'}</div><div class=l>wrong amounts spoken on the photos by other people</div></div>""")

slides = [
    {"html": f"""<div class=bar></div><div class=grid2><div><div class=tag>DEEP LEARNING · SEMESTER PROJECT</div><h1 style="font-size:96px">Pakistani Banknote Reader</h1>
<p class=lead>Point your phone at a banknote. It says the value out loud, in Urdu.</p>
<p class=muted style="margin-top:60px">{author} · BSCS-7A · Superior University, Lahore<br>Instructor: Dr. Munir Ahmad</p></div>
<div><img class=fig src="figures/augmentations.png" style="transform:rotate(-2deg)"></div></div>""",
     "say": f"Assalam o alaikum. I'm {author}, and this is a Pakistani banknote reader for blind people. You point your phone at a note, and it tells you the value out loud, in Urdu."},
    {"html": """<div class=bar></div><h2>A blind person pays in cash every day.<br>Every note is a question.</h2>
<div class=grid3><div class="stat yellow"><div class=n>1.1M</div><div class=l>blind people in Pakistan (GBD 2017)</div></div>
<div class=stat><div class=n>7</div><div class=l>notes in daily use, several close in size and colour</div></div>
<div class=stat><div class="n red">Rs 4,500</div><div class=l>lost in one 500 / 5,000 mix-up</div></div></div>
<p class=lead style="margin-top:50px">Today the answer is: ask a stranger, and trust them.</p>""",
     "say": "Pakistan has about one point one million blind people, and most daily payments are still made in cash. Our seven notes differ only a little in size, so a blind person usually has to ask a stranger which note is which, and trust the answer. Mixing up a five hundred and a five thousand costs four and a half thousand rupees."},
    {"html": """<div class=bar></div><h2>A web page, not an app</h2><div class=grid2><div><ul>
<li>Reads the live camera <b>4 times a second</b></li><li>Speaks the value in <b>Urdu</b> or English</li>
<li>Stays <b>silent when unsure</b>: a wrong amount is worse than none</li><li>Works <b>offline</b> once loaded, any phone browser</li>
<li>Vibration code and screen-reader support</li></ul></div>
<div style="text-align:center"><div class=ur style="font-size:110px;font-weight:700">پانچ سو روپے</div><div class=muted style="font-size:34px">"Five hundred rupees"</div></div></div>""",
     "say": "The reader is just a web page. It reads the camera four times a second and speaks the value in Urdu or English, with a recorded voice, so it works even on phones that have no Urdu speech. It stays silent when it's unsure, because saying a wrong amount is worse than saying nothing. Once loaded, it works offline, and it also vibrates a code and works with screen readers."},
    {"html": f"""<div class=bar></div><h2>Data: 3,611 real phone photos, 15 classes</h2><div class=grid2><div><ul>
<li>Kaggle <b>Pakistan Image Currency Dataset</b> (CC0)</li><li>7 notes × <b>front and back</b> = 14 classes</li>
<li>+ <b>"no note"</b>: 280 photos + 1,000 everyday Imagenette photos</li><li>Folded, in hand, half visible, on bedsheets</li>
<li>Split 70 / 15 / 15, near-copies kept together</li></ul></div><img class=fig src="figures/augmentations.png"></div>""",
     "say": "I trained on a public dataset of three thousand six hundred real phone photos: seven notes, front and back, many folded or half visible. I added a 'no note' class with a thousand everyday photos, so the model learns when to stay quiet."},
    {"html": f"""<div class=bar></div><h2>Before trusting any number:<br>are the test photos really new?</h2>
<div class=grid2><div class="stat yellow"><div class=n>{D['roboflow_dropped']} / {D['roboflow_total']}</div><div class=l>test photos in a second public dataset are near-copies of the first</div></div>
<div><ul><li>Perceptual hash of every photo (64 bits)</li><li>Photos within 8 bits = the same photo</li><li>Only <b>{D['roboflow_kept']}</b> genuinely new photos left</li><li>Without this check: near-perfect, and meaningless</li></ul></div></div>""",
     "say": f"Before trusting any number, I checked whether the test photos were really new. I hashed every photo, and found that {D['roboflow_dropped']} of the {D['roboflow_total']} test photos in a second public currency dataset are near-copies of the first. Only {D['roboflow_kept']} were genuinely new. Without this check, my external test would have looked perfect and meant nothing."},
    {"html": """<div class=bar></div><h2>Training for how blind people actually hold notes</h2><img class=fig src="figures/augmentations.png" style="max-height:740px;width:auto">""",
     "say": "Training images are changed on the GPU every time they're seen: any angle, too close, half out of frame, fingers over the note, dim or yellow light, and blur. No mirror flips, because the camera never sees a mirrored note."},
    {"html": f"""<div class=bar></div><h2>Three models: accuracy vs speed</h2>
<table><tr><th>Model</th><th>Params</th><th>Same dataset</th><th>Other people</th>{own_th}<th>CPU</th></tr>{rows}</table>
<p class=muted style="margin-top:30px">Same data, same recipe. Highlighted: deployed (best on validation).</p>""",
     "say": f"I compared a CNN trained from scratch with two phone networks pretrained on ImageNet: MobileNet version three and the new MobileNet version four. On photos from the same dataset, all three are almost perfect. On photos taken by other people, the difference shows: transfer learning holds up far better. I deployed {NICE[best]}, the best on validation" + (", and also the fastest." if B["onnx_cpu_ms"] == min(r["onnx_cpu_ms"] for r in runs.values()) else ".")},
    {"html": f"""<div class=bar></div><h2>The honest numbers</h2><div class=grid3>
<div class=stat><div class=n>{pct(B['test']['value_accuracy'], 0)}</div><div class=l>unseen photos from the same dataset</div></div>
<div class="stat yellow"><div class=n>{pct(ext['value_accuracy'], 0) if ext else '–'}</div><div class=l>photos by other photographers ({D['roboflow_kept']})</div></div>
{third}</div>
<p class=lead style="margin-top:46px">The same-dataset score mostly shows how alike those photos are.<br>The other two are what a user would get.</p>""",
     "say": f"Here are the honest numbers. {pct(B['test']['value_accuracy'], 0)} on unseen photos from the same dataset, but {pct(ext['value_accuracy'], 0) if ext else 'less'} on photos by other people" + (f", and {pct(own['value_accuracy'], 0)} on {own['n']} photos I took of my own notes." if own else ", and on those photos it never said a wrong amount.") + " The first number mostly shows how alike the dataset's photos are; the others are what a real user would get."},
    {"html": f"""<div class=bar></div><h2>When to speak, and when to stay silent</h2><div class=grid2><img class=fig src="figures/speak_threshold.png"><div><ul>
<li>Validation photos look too easy: any threshold seems safe</li><li>So: calibrate on a <b>stress test</b>, validation photos × 4 augmentations</li>
<li>Threshold <b>{B['speaking_threshold']:.2f}</b>: at most 1 wrong value in 500 there</li><li>Plus: same answer in <b>3 frames in a row</b></li></ul></div></div>""",
     "say": f"The most important design decision is when to speak. On ordinary validation photos the model is almost never wrong, so any threshold looks safe. So I calibrated on a stress test: every validation photo four times, with the hard augmentations. The app speaks only above {B['speaking_threshold']:.2f} confidence, and only when three frames in a row agree."},
    {"html": f"""<div class=bar></div><div class=grid2><div><h2>Live demo</h2><ul><li>Open <b>{site}</b> on any phone</li><li>Tap <b>Start camera</b></li><li>Hold a note ~20 cm away</li><li>It speaks: <span class=ur>پانچ سو روپے</span></li></ul></div>
<div style="text-align:center"><img class=phone src="figures/site_phone.png"></div></div>""",
     "say": f"Here's the live demo. I open the site on my phone, tap Start camera, and hold a note in front of it. It reads several frames a second and speaks the value, with nothing installed."},
    {"html": """<div class=bar></div><h2>Limits, and what's next</h2><div class=grid2><div><b style="font-size:34px">Limits</b><ul>
<li>A few physical notes per value in training</li><li>2005–2008 designs; the new series needs new photos</li><li>One note at a time; not a counterfeit detector</li></ul></div>
<div><b style="font-size:34px">Next</b><ul><li>Worn, torn and taped notes</li><li>Count a whole handful of notes</li><li>Test with blind users</li></ul></div></div>""",
     "say": "Its limits: the training photos show only a few physical notes per value, it knows the current note designs but not the State Bank's new series, it reads one note at a time, and it doesn't detect counterfeits. Next: worn notes, counting a whole handful, and a test with blind users."},
    {"html": f"""<div class=bar></div><h2>Everything is open</h2><ul style="margin-top:20px">
<li>Try it: <b>{site}</b></li><li>Code, weights, splits{', my photos' if own else ''}: <b>{gh}</b></li><li>Report, slides, model card: linked on the site</li></ul>
<p class=lead style="margin-top:70px">Thank you, Dr. Munir Ahmad.</p><div class=ur style="font-size:60px;margin-top:20px">شکریہ</div>""",
     "say": "Everything is open: the website, the code, the weights and the data splits" + (" and my own photos" if own else "") + " are all linked. Thank you, Dr. Munir Ahmad, and thank you for watching."},
]

assets = {"results/figures": "figures"}
os.makedirs("docs/slides", exist_ok=True)
build(slides, CSS, "docs/slides", voice=a.voice, assets=assets, title="BoltaNote_slides", author=author)
