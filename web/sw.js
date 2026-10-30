// Keeps the reader working with no internet. Pages and data check the network first and fall back to
// the saved copy; the model, runtime, voice clips and photos come from the saved copy, which is
// replaced whenever a new build changes the version below. Anything else goes straight to the network.
const VERSION = "fc9151c99339", CACHE = "reader-" + VERSION;
const CORE = ["./", "app.js", "page.js", "style.css", "art/note-500.jpg", "art/note-5000.jpg", "favicon.svg", "site.webmanifest", "model_card.json", "samples/samples.json", "model/boltanote.onnx", "icons/icon-192.png", "fonts/BricolageGrotesque-var-latin.woff2", "fonts/IBMPlexMono-400-latin.woff2", "fonts/NotoNastaliqUrdu-arabic.woff2", "vendor/ort/ort-wasm-simd-threaded.mjs", "vendor/ort/ort-wasm-simd-threaded.wasm", "vendor/ort/ort.min.js", "audio/en/10.mp3", "audio/en/100.mp3", "audio/en/1000.mp3", "audio/en/20.mp3", "audio/en/50.mp3", "audio/en/500.mp3", "audio/en/5000.mp3", "audio/en/ready.mp3", "audio/ur/10.mp3", "audio/ur/100.mp3", "audio/ur/1000.mp3", "audio/ur/20.mp3", "audio/ur/50.mp3", "audio/ur/500.mp3", "audio/ur/5000.mp3", "audio/ur/ready.mp3", "samples/1000_back_1.jpg", "samples/1000_front_1.jpg", "samples/100_back_1.jpg", "samples/100_front_1.jpg", "samples/10_back_1.jpg", "samples/10_front_1.jpg", "samples/20_back_1.jpg", "samples/20_front_1.jpg", "samples/5000_back_1.jpg", "samples/5000_front_1.jpg", "samples/500_back_1.jpg", "samples/500_front_1.jpg", "samples/50_back_1.jpg", "samples/50_front_1.jpg", "samples/none_1.jpg", "samples/none_2.jpg"];
const FRESH = /(^|\/)($|index\.html|app\.js|style\.css|[\w.-]+\.json|site\.webmanifest)$/;
const scope = new URL(self.registration.scope);
const rel = (url) => url.pathname.slice(scope.pathname.length);

self.addEventListener("install", (e) => e.waitUntil(
  caches.open(CACHE).then((c) => Promise.allSettled(CORE.map((u) => c.add(u)))).then(() => self.skipWaiting())
));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith("reader-") && k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())
));

// Audio elements ask for byte ranges, which a saved response has to answer by hand.
async function ranged(req, res) {
  const m = /bytes=(\d+)-(\d*)/.exec(req.headers.get("range"));
  if (!m) return res;
  const buf = await res.arrayBuffer(), start = +m[1], end = m[2] ? +m[2] : buf.byteLength - 1;
  return new Response(buf.slice(start, end + 1), { status: 206, headers: { "Content-Type": res.headers.get("Content-Type") || "application/octet-stream", "Content-Range": `bytes ${start}-${end}/${buf.byteLength}`, "Content-Length": String(end - start + 1), "Accept-Ranges": "bytes" } });
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== scope.origin) return;
  const path = rel(url);
  if (e.request.mode !== "navigate" && !CORE.includes(path || "./")) return;
  e.respondWith((async () => {
    const c = await caches.open(CACHE), key = e.request.mode === "navigate" ? "./" : path;
    if (e.request.mode === "navigate" || FRESH.test(path)) {
      try {
        const res = await fetch(e.request);
        if (res.ok && e.request.mode !== "navigate") c.put(key, res.clone());
        else if (res.ok && (path === "" || path === "index.html")) c.put("./", res.clone());
        return res;
      } catch (err) {
        const saved = await c.match(key);
        if (saved) return saved;
        throw err;
      }
    }
    const saved = await c.match(key);
    if (saved) return e.request.headers.has("range") ? ranged(e.request, saved) : saved;
    return fetch(e.request);
  })());
});
