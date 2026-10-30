"""Open every link a grading agent could follow on a deployed project site and report anything broken.
Checks the page's <a href>, <img src>, <script src>, project.json links/files, and llms.txt URLs.
usage: check_links.py https://site.vercel.app"""
import json
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = {"User-Agent": "Mozilla/5.0 (link check)"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "link" and a.get("rel") in ("preconnect", "dns-prefetch"):
            return  # connection hints, not links
        for k in ("href", "src", "poster"):
            if a.get(k) and not a[k].startswith(("#", "mailto:", "javascript:", "data:")):
                self.urls.add(a[k])


def fetch(url, method="GET"):
    req = urllib.request.Request(url, headers=UA, method=method)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.status, r.read() if method == "GET" else b""


def check(url):
    for method in ("HEAD", "GET"):
        try:
            return fetch(url, method)[0]
        except urllib.error.HTTPError as e:
            if method == "GET":
                return e.code
        except Exception as e:  # noqa: BLE001
            if method == "GET":
                return f"ERR {type(e).__name__}: {e}"
    return "ERR"


site = sys.argv[1].rstrip("/") + "/"
status, body = fetch(site)
p = Links()
p.feed(body.decode("utf-8", "replace"))
urls = {urllib.parse.urljoin(site, u) for u in p.urls}
for extra in ("project.json", "llms.txt", "llms-full.txt", "model_card.json", "robots.txt", "sitemap.xml"):
    urls.add(site + extra)
try:
    proj = json.loads(fetch(site + "project.json")[1])
    for v in list(proj.get("links", {}).values()) + list(proj.get("weights_files", {}).values()) + list(proj.get("weights_files_copies", {}).values()):
        if v:
            urls.add(urllib.parse.urljoin(site, v))
    for c in proj.get("copies", {}).values():
        urls.update(urllib.parse.urljoin(site, v) for v in c["urls"])
        if len(set(c["urls"])) < 2:
            print("ONLY ONE COPY:", c["title"])
except Exception as e:  # noqa: BLE001
    print("project.json unreadable:", e)
try:
    urls |= set(re.findall(r"https?://[^\s);,]+", fetch(site + "llms.txt")[1].decode()))
    urls |= set(re.findall(r"<loc>([^<]+)</loc>", fetch(site + "sitemap.xml")[1].decode()))
except Exception as e:  # noqa: BLE001
    print("llms.txt unreadable:", e)

bad = []
for u in sorted(urls):
    s = check(u)
    ok = isinstance(s, int) and s < 400
    # LinkedIn/Kaggle answer bots with 999/403/429 even when the page exists
    soft = isinstance(s, int) and s in (403, 429, 999) and any(h in u for h in ("linkedin.com", "kaggle.com"))
    print(f"{'ok ' if ok else 'SOFT' if soft else 'BAD'} {s}  {u}")
    if not ok and not soft:
        bad.append((u, s))
print(f"\n{len(urls)} links checked, {len(bad)} broken")
sys.exit(1 if bad else 0)
