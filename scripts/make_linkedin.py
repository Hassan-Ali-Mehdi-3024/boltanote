"""Write docs/linkedin_post.md from the results."""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
links = json.load(open("links.json")); S = json.load(open("results/summary.json")); D = json.load(open("results/dedupe.json"))
B = S["models"][S["deployed_model"]]; ext, own = B.get("external"), B.get("own")
pct = lambda v, d=0: f"{v * 100:.{d}f}%"
post = f"""About 1.1 million people in Pakistan are blind, and most payments are still cash. For them, every banknote is a question: is this 500 or 5,000? The usual answer is to ask a stranger and trust them.

For my Deep Learning semester project I built a Pakistani banknote reader for blind people: point your phone at a Pakistani note and it says the value out loud, in Urdu or English. It's a web page, so there's nothing to install, and it works offline once loaded.

What I learned building it:
• Check your test data first: {D['roboflow_dropped']} of {D['roboflow_total']} test photos in a second public currency dataset were near-copies of the first. Only {D['roboflow_kept']} were genuinely new.
• Near-perfect accuracy on the dataset's own photos ({pct(B['test']['value_accuracy'])}) means little. On photos by other people it scored {pct(ext['value_accuracy']) if ext else '–'}{(', and on ' + str(own['n']) + ' photos I took of my own notes, ' + pct(own['value_accuracy'])) if own else ''}.
• Silence beats a wrong answer: it only speaks above a confidence threshold calibrated on stress-tested photos, and when 3 frames in a row agree.
• MobileNet (ImageNet-pretrained) generalised far better than a CNN trained from scratch, and runs fast enough for a cheap phone.

Try it on your phone: {links.get('site') or '[Vercel link]'}
Code, weights{' and my own test photos' if own else ' and data splits'}: {links['github']}
Walkthrough video: {links.get('youtube') or '[YouTube link]'}

Thanks to Dr. Munir Ahmad, and to the authors of the open Pakistani currency datasets.

#DeepLearning #Accessibility #ComputerVision #AssistiveTechnology #Pakistan #SuperiorUniversity
"""
os.makedirs("docs", exist_ok=True)
open("docs/linkedin_post.md", "w", encoding="utf-8").write(post)
print(post)
