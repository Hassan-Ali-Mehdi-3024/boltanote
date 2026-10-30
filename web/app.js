// Pakistani Banknote Reader in the browser. About four times a second the whole camera frame is squeezed to
// 224x224 (the same preprocessing as predict.py) and run through the model with ONNX Runtime Web.
// Front + back probabilities are added per note value. It speaks only when the same value has
// been on top, above the threshold, for 3 frames in a row; it repeats after 6 s, or straight away
// after the note leaves the view and comes back.

const SIZE = 224, STABLE = 3, REPEAT_MS = 6000, FRAME_MS = 250;
let session, card, samples = [], values, classes, threshold;
let running = false, stream = null, history = [], lastSaid = null, lastSaidAt = 0, noneRun = 0;
let lang = localGet("lang") || "ur", vibrate = localGet("vibrate") !== "0";
const $ = (s) => document.querySelector(s);
const canvas = document.createElement("canvas");
canvas.width = canvas.height = SIZE;
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const audio = {};

function localGet(k) { try { return localStorage.getItem("boltanote." + k); } catch { return null; } }
function localSet(k, v) { try { localStorage.setItem("boltanote." + k, v); } catch {} }

async function init() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  setLang(lang);
  setVibrate(vibrate);
  try {
    [card, samples] = await Promise.all([fetch("model_card.json").then((r) => r.json()), fetch("samples/samples.json").then((r) => r.json())]);
    values = card.values; classes = card.classes; threshold = card.speaking.threshold;
    buildGallery();
    session = await ort.InferenceSession.create("model/boltanote.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    state("Ready. Tap “Start camera”, or try a sample photo below.");
    if (pendingStill) still(...pendingStill);  // a sample tapped while the model was loading
  } catch (e) {
    console.error(e);
    state("Could not load the model in this browser: " + e.message);
  }
}

function state(t) { $("#state").textContent = t; }

function preload(l) {
  for (const k of [...values.filter((v) => v !== "none"), "ready"]) {
    if (!audio[l + k]) { audio[l + k] = new Audio(`audio/${l}/${k}.mp3`); audio[l + k].preload = "auto"; }
  }
}

function say(key) {
  const a = audio[lang + key];
  if (!a) return;
  Object.values(audio).forEach((x) => { x.pause(); x.currentTime = 0; });
  a.play().catch(() => {});
}

function buzz(value) {
  if (!vibrate || !navigator.vibrate) return;
  // short pulses for small notes, long for big ones: 10/20/50/100 = 1-4 short, 500/1000/5000 = 1-3 long
  const i = ["10", "20", "50", "100", "500", "1000", "5000"].indexOf(value);
  const n = i < 4 ? i + 1 : i - 3, d = i < 4 ? 90 : 350;
  navigator.vibrate(Array.from({ length: n * 2 - 1 }, (_, k) => (k % 2 ? 120 : d)));
}

function tensorFrom(source, w, h) {
  ctx.drawImage(source, 0, 0, w, h, 0, 0, SIZE, SIZE); // whole frame, squeezed (as in training)
  const d = ctx.getImageData(0, 0, SIZE, SIZE).data, n = SIZE * SIZE;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { x[i] = d[4 * i] / 255; x[n + i] = d[4 * i + 1] / 255; x[2 * n + i] = d[4 * i + 2] / 255; }
  return new ort.Tensor("float32", x, [1, 3, SIZE, SIZE]);
}

async function classify(source, w, h) {
  const out = await session.run({ image: tensorFrom(source, w, h) });
  const p = out.probs.data, v = new Array(values.length).fill(0);
  classes.forEach((c, i) => { v[values.indexOf(c.split("_")[0])] += p[i]; });
  const k = v.indexOf(Math.max(...v));
  return { value: values[k], prob: v[k], side: values[k] === "none" ? null : classes[p.indexOf(Math.max(...p))].split("_")[1] };
}

function show(r, speaking) {
  const box = $("#result");
  const ph = card.speaking.phrases[r.value];
  if (r.value === "none" || r.prob < threshold) {
    box.className = "result none";
    $("#value").textContent = r.value === "none" ? "No note in view" : "Not sure yet…";
    $("#value-ur").textContent = r.value === "none" ? "کوئی نوٹ نظر نہیں آ رہا" : "نوٹ ٹھیک سے دکھائیں";
  } else {
    box.className = "result";
    $("#value").textContent = "Rs " + Number(r.value).toLocaleString("en-PK");
    $("#value-ur").textContent = ph[0];
  }
  $("#conf").style.width = (r.prob * 100).toFixed(0) + "%";
  $("#prob").textContent = `${r.value === "none" ? "no note" : "Rs " + r.value}${r.side ? " (" + r.side + ")" : ""} · ${(r.prob * 100).toFixed(1)}%${speaking ? " · speaking" : ""}`;
}

async function loop() {
  const video = $("#video");
  while (running) {
    const t0 = performance.now();
    if (video.readyState >= 2) {
      const r = await classify(video, video.videoWidth, video.videoHeight);
      history.push(r);
      if (history.length > STABLE) history.shift();
      noneRun = r.value === "none" ? noneRun + 1 : 0;
      if (noneRun >= 2) lastSaid = null; // note removed: say it again when it comes back
      const stable = history.length === STABLE && history.every((h) => h.value === r.value && h.value !== "none" && h.prob >= threshold);
      const now = Date.now();
      let speaking = false;
      if (stable && (r.value !== lastSaid || now - lastSaidAt > REPEAT_MS)) {
        say(r.value); buzz(r.value);
        lastSaid = r.value; lastSaidAt = now; speaking = true;
        $("#live").textContent = card.speaking.phrases[r.value][lang === "ur" ? 0 : 1];
      }
      show(r, speaking);
      $("#fps").textContent = `${Math.round(performance.now() - t0)} ms per frame on this phone`;
    }
    await new Promise((res) => setTimeout(res, Math.max(0, FRAME_MS - (performance.now() - t0))));
  }
}

async function start() {
  if (running) return stop();
  if (!session) return state("The model is still loading…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 960 }, height: { ideal: 1280 } }, audio: false });
  } catch (e) {
    return state("Camera not available (" + e.name + "). You can still try the sample photos.");
  }
  const video = $("#video");
  video.srcObject = stream;
  video.hidden = false; $("#still").hidden = true; $("#placeholder").hidden = true;
  await video.play();
  preload(lang);
  say("ready");
  running = true; history = []; lastSaid = null;
  $("#start").textContent = "Stop camera"; $("#start").classList.add("stop");
  state("Camera on. Hold a note about 20 cm from the phone.");
  loop();
}

function stop() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  $("#start").textContent = "Start camera"; $("#start").classList.remove("stop");
  state("Camera off.");
}

let pendingStill = null;
async function still(url, label) {
  if (!session) { pendingStill = [url, label]; return state("The model is still loading… this photo will be read as soon as it is ready."); }
  if (running) stop();
  const img = $("#still");
  img.onload = async () => {
    $("#video").hidden = true; $("#placeholder").hidden = true; img.hidden = false;
    const t0 = performance.now();
    const r = await classify(img, img.naturalWidth, img.naturalHeight);
    show(r, false);
    preload(lang);
    if (r.value !== "none" && r.prob >= threshold) { say(r.value); $("#live").textContent = card.speaking.phrases[r.value][lang === "ur" ? 0 : 1]; }
    $("#fps").textContent = `${Math.round(performance.now() - t0)} ms on this device` + (label ? ` · true value: ${label === "none" ? "no note" : "Rs " + label}` : "");
  };
  img.src = url;
}

function buildGallery() {
  const g = $("#gallery");
  samples.forEach((s) => {
    const b = document.createElement("button");
    b.type = "button";
    const name = s.value === "none" ? "no note" : "Rs " + s.value;
    b.setAttribute("aria-label", `Sample photo: ${name}${s.side ? ", " + s.side : ""}`);
    b.innerHTML = `<img src="samples/${s.file}" alt="" loading="lazy"><span>${name}</span>`;
    b.addEventListener("click", () => still(`samples/${s.file}`, s.value));
    g.appendChild(b);
  });
}

function setLang(l) {
  lang = l; localSet("lang", l);
  document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.lang === l));
  if (values) preload(l);
}
function setVibrate(v) {
  vibrate = v; localSet("vibrate", v ? "1" : "0");
  $("#vib").setAttribute("aria-pressed", v);
}

$("#start").addEventListener("click", start);
document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
$("#vib").addEventListener("click", () => setVibrate(!vibrate));
$("#again").addEventListener("click", () => { if (lastSaid) say(lastSaid); });
$("#file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) still(URL.createObjectURL(f), null); });
init();
