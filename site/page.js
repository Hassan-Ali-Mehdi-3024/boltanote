// The page around the reader: guilloche rosettes, dot counts, tap-to-hear notes, and scroll reveals.
// Everything is drawn in its finished state first, so a crawler, a screenshot or a person who
// never scrolls sees the whole page. Animation only starts once someone actually scrolls.
(() => {
  const motion = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ease = (x) => 1 - Math.pow(1 - x, 3);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // A guilloche: each ring is one true hypotrochoid, p lobes wound over q orbits, as on a banknote.
  const RINGS = [[372, 104, 113, 13], [232, 66, 89, 11], [118, 46, 53, 7]];
  const curves = RINGS.map(([C, d, p, q]) => {
    const r = C / (p / q - 1), k = C / r, n = p * 34, pts = new Float32Array((n + 1) * 2);
    for (let i = 0; i <= n; i++) { const t = (i / n) * q * 2 * Math.PI; pts[2 * i] = C * Math.cos(t) + d * Math.cos(k * t); pts[2 * i + 1] = C * Math.sin(t) - d * Math.sin(k * t); }
    return pts;
  });
  // Draws from fraction f0 to f1 on top of what is there, so drawing it in costs almost nothing per frame.
  function rosette(cv, f1, f0 = 0) {
    const size = cv.clientWidth, dpr = Math.min(devicePixelRatio || 1, 2), c = cv.getContext("2d");
    if (!size) return;
    if (!f0) { cv.width = cv.height = Math.round(size * dpr); c.setTransform((size * dpr) / 960, 0, 0, (size * dpr) / 960, (size * dpr) / 2, (size * dpr) / 2); }
    c.lineWidth = (.85 * 960) / size;
    c.strokeStyle = `rgba(${cv.dataset.color},${cv.dataset.alpha})`;
    curves.forEach((pts) => {
      const n = pts.length / 2 - 1, from = Math.floor(n * f0), to = Math.floor(n * f1);
      c.beginPath(); c.moveTo(pts[2 * from], pts[2 * from + 1]);
      for (let i = from + 1; i <= to; i++) c.lineTo(pts[2 * i], pts[2 * i + 1]);
      c.stroke();
    });
  }
  function run(ms, fn, done) {
    const t0 = performance.now();
    const step = (t) => { const x = Math.min(1, (t - t0) / ms); fn(ease(x)); if (x < 1) requestAnimationFrame(step); else if (done) done(); };
    requestAnimationFrame(step);
  }
  addEventListener("load", () => $$("canvas.rosette, canvas.note-rose").forEach((cv) => {
    if (motion && cv.dataset.draw) { let f = 0; run(2600, (x) => { rosette(cv, x, f); f = x; }); } else rosette(cv, 1);
  }));

  // Rows of dots that stand for real photos: filled, crossed out, or a hollow ring.
  function dots(cv, f) {
    const groups = JSON.parse(cv.dataset.groups), n = groups.reduce((a, g) => a + g[0], 0), cols = +cv.dataset.cols;
    const w = cv.clientWidth, dpr = Math.min(devicePixelRatio || 1, 2);
    if (!w) return;
    const step = w / cols, rows = Math.ceil(n / cols), c = cv.getContext("2d");
    cv.width = Math.round(w * dpr); cv.height = Math.round(rows * step * dpr); cv.style.height = rows * step + "px";
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const R = step * (cv.dataset.size === "big" ? .3 : .31), show = Math.ceil(n * f);
    let i = 0;
    for (const [count, col, kind] of groups) for (let k = 0; k < count && i < show; k++, i++) {
      const x = (i % cols + .5) * step, y = (Math.floor(i / cols) + .5) * step;
      c.fillStyle = c.strokeStyle = `rgb(${col})`;
      c.lineWidth = Math.max(1.2, R * .32);
      c.beginPath();
      if (kind === "x") { const a = R * .78; c.moveTo(x - a, y - a); c.lineTo(x + a, y + a); c.moveTo(x + a, y - a); c.lineTo(x - a, y + a); c.stroke(); }
      else if (kind === "ring") { c.arc(x, y, R * .85, 0, 7); c.stroke(); }
      else { c.arc(x, y, R, 0, 7); c.fill(); }
    }
  }
  const dotCanvases = $$("canvas.dots");
  const redraw = () => { dotCanvases.forEach((cv) => dots(cv, 1)); $$("canvas.rosette, canvas.note-rose").forEach((cv) => { if (!cv.dataset.draw || performance.now() > 3000) rosette(cv, 1); }); };
  dotCanvases.forEach((cv) => dots(cv, 1));
  let resize; addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(redraw, 150); });

  // Lazy pictures still load soon after the page does, so a screenshot taken without scrolling has no holes.
  addEventListener("load", () => setTimeout(() => $$('main img[loading="lazy"]:not(#gallery img)').forEach((i) => { i.loading = "eager"; }), 600));

  // Scroll progress runs up the security thread.
  const thread = document.getElementById("thread");
  const onScroll = () => { const h = document.documentElement.scrollHeight - innerHeight; thread.style.setProperty("--p", h > 0 ? scrollY / h : 0); };
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  // The note stack: tap a note, hear the same recorded voice the reader uses.
  const line = document.getElementById("said-line"), clips = {};
  $$(".note").forEach((b) => b.addEventListener("click", () => {
    Object.values(clips).forEach((a) => { a.pause(); a.currentTime = 0; });
    const a = clips[b.dataset.v] || (clips[b.dataset.v] = new Audio(`audio/ur/${b.dataset.v}.mp3`));
    a.play().catch(() => {});
    $$(".note.on").forEach((n) => n.classList.remove("on")); b.classList.add("on");
    line.replaceChildren(Object.assign(document.createElement("span"), { className: "ur", lang: "ur", textContent: b.dataset.ur }));
  }));

  // The hero button starts the camera as well as scrolling to it.
  const heroStart = document.getElementById("hero-start");
  if (heroStart) heroStart.addEventListener("click", () => { if (window.readerStart) window.readerStart(); });

  // A clear play button over the film's poster.
  const film = document.querySelector("video.film"), play = document.getElementById("play");
  if (film && play) {
    play.hidden = false;
    play.addEventListener("click", () => { film.play().catch(() => {}); film.focus(); });
    film.addEventListener("play", () => { play.hidden = true; });
    film.addEventListener("pause", () => { if (!film.seeking && film.currentTime === 0) play.hidden = false; });
  }

  if (!motion || !("IntersectionObserver" in window)) return;

  const counters = (root) => $$("[data-count]", root).forEach((el) => {
    const to = +el.dataset.count, dp = +el.dataset.dp || 0, text = el.textContent;
    if (!to) return;
    run(1300, (x) => { const v = to * x; el.textContent = el.dataset.comma ? Math.round(v).toLocaleString("en-PK") : v.toFixed(dp); }, () => { el.textContent = text; });
  });
  const gauge = (g) => {
    const to = +g.dataset.to, mark = g.querySelector(".gauge-mark"), label = mark.querySelector("b"), said = label.textContent;
    // starts where an untested threshold would sit, then settles where the stress test put it
    run(1800, (x) => { const v = .4 + (to - .4) * x; g.style.setProperty("--at", v); label.textContent = v.toFixed(2); }, () => { g.style.setProperty("--at", to); label.textContent = said; });
  };
  const enter = (t) => {
    if (t.classList.contains("pre")) { t.classList.remove("pre"); t.classList.add("in"); }
    if (t.classList.contains("bar-t")) { t.classList.remove("pre-b"); t.classList.add("in-b"); }
    if (t.matches("canvas.dots")) run(1500, (x) => dots(t, x));
    if (t.classList.contains("gauge")) gauge(t);
    counters(t);
  };
  const seen = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { seen.unobserve(e.target); enter(e.target); }
  }), { rootMargin: "0px 0px -8% 0px", threshold: .05 });

  // Only what is about to scroll into view is held back, and never for more than a moment,
  // so a crawler that scrolls and then takes a page screenshot still sees everything.
  const reveal = (el) => {
    if (el.matches(".bar-t")) el.classList.add("pre-b");
    else if (!el.matches("canvas.dots, .gauge")) el.classList.add("pre");
    seen.observe(el);
    setTimeout(() => { if (el.classList.contains("pre") || el.classList.contains("pre-b")) { seen.unobserve(el); el.classList.remove("pre", "pre-b"); } }, 2400);
  };
  const near = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    near.unobserve(e.target);
    if (e.boundingClientRect.top > innerHeight * .92) reveal(e.target);
  }), { rootMargin: "0px 0px 55% 0px" });
  let armed = false;
  addEventListener("scroll", () => {
    if (armed) return; armed = true;
    $$("section h2, .sub, .huge, .stack-fig, .mixup > *, .film-frame, .feature, .speak > *, .pair, .wall, .sources > div, .table-wrap, .model-row, .augment > *, .score, .figs > *, .limits-grid > *, .qa, .foot, .bar-t, canvas.dots, .gauge").forEach((el) => near.observe(el));
  }, { passive: true, once: true });
})();
