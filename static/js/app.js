/* ContextDepth project page: interactions. */
(function () {
  'use strict';
  const CD = window.CD, R = window.CD_RESULTS;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const MODEL_NAMES = { da3: 'Depth Anything 3', moge2: 'MoGe-2', moge3: 'MoGe-3' };
  const pct = (v, d) => (v * 100).toFixed(d == null ? 1 : d) + '%';
  const signed = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
  const m = CD.fmtMeters;
  const ease = (t) => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  let LUT = null;
  const SCENES = {};
  const url = (s, f) => `static/scenes/${s.id}/${f}`;
  const depthOf = (s, name) => CD.sceneDepth(s, name);
  const selName = (s, mk) => `${mk}_c${s.models[mk].sel.choice}`;

  function bindSeg(root, onChange) {
    root.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b || b.getAttribute('aria-selected') === 'true') return;
      setSeg(root, b.dataset.v != null ? b.dataset.v : b.dataset.state);
      onChange(b.dataset.v != null ? b.dataset.v : b.dataset.state, e);
    });
  }
  function setSeg(root, v) {
    $$('button', root).forEach(b => b.setAttribute('aria-selected', String((b.dataset.v != null ? b.dataset.v : b.dataset.state) === String(v))));
  }
  function put(host, canvas) { host.replaceChildren(canvas); return canvas; }
  function tween(from, to, ms, fn) {
    if (reduceMotion || ms <= 0) { fn(to); return; }
    const t0 = performance.now();
    (function step(now) {
      const t = Math.min(1, (now - t0) / ms);
      fn(from + (to - from) * ease(t));
      if (t < 1) requestAnimationFrame(step);
    })(t0);
  }
  function onVisible(node, fn, opts) {
    const io = new IntersectionObserver((es) => es.forEach(e => fn(e.isIntersecting, e)), opts || { threshold: 0.15 });
    io.observe(node);
    return io;
  }
  /* Spectral colours for a depth sub-range, as a CSS gradient. */
  function depthGradient(vis, lo, hi) {
    const l0 = Math.log(vis[0]), l1 = Math.log(vis[1]) - l0, stops = [];
    for (let i = 0; i <= 14; i++) {
      const d = Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * i / 14);
      const c = LUT[Math.max(0, Math.min(255, Math.round((Math.log(d) - l0) / l1 * 255)))];
      stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${(i * 100 / 14).toFixed(1)}%`);
    }
    return `linear-gradient(to right, ${stops.join(', ')})`;
  }

  /* ───────────────────────── Image wall ─────────────────────────
   * Every tile is an evaluation query on which ContextDepth improves the frozen model.
   * Rank 0 sits in the centre and later ranks spiral outwards. Scroll position drives it:
   * query -> outpainted canvas -> depth (query only | ContextDepth | ground truth)
   *       -> point clouds (query only | ContextDepth) -> dive into the interactive 3D stage. */
  function initWall() {
    const sec = $('#wall'), pin = $('#wall-pin'), cv = $('#wall-canvas');
    if (!sec || !cv) return;
    const ctx = cv.getContext('2d'), base = 'static/wall/';
    const W = { meta: null, img: {}, p: 0, ph: -1, hover: -1, cells: null, rankAt: null, cols: 0, rows: 0, s: 0, th: 0, ox: 0, oy: 0, dpr: 1, born: 0,
                pages: { rgb: new Map(), depth: new Map(), cloud: new Map() }, pending: false, mouse: null, over: null, cardKey: '' };
    const cl = (x) => x < 0 ? 0 : x > 1 ? 1 : x, ss = (x, a, b) => { const t = cl((x - a) / (b - a)); return t * t * (3 - 2 * t); };
    const lin = (x, k) => cl((x - k[0]) / (k[1] - k[0])), num = (v) => v.toLocaleString('en-US');
    const cap = $('#wall-cap'), hero = $('#wall-hero'), veil = $('#wall-veil'), sweep = $('#wall-sweep'), rail = $('#wall-rail');
    const card = $('#wall-card'), cardCv = $('#wall-card-canvas'), cardMeta = $('#wall-card-meta'), hint = $('#wall-scroll');
    ['Queries', 'Outpaint', 'Depth · query only', 'Depth · ContextDepth', 'Depth · ground truth', 'Point clouds · query only', 'Point clouds · ContextDepth', 'Explore in 3D']
      .forEach((n) => rail.appendChild(el('li', '', `<i></i><span>${n}</span>`)));
    const CAPS = [
      (mt) => `<b>${num(mt.n)} queries where ContextDepth beats the frozen model.</b> Each one is a keyhole view of its scene.` +
              `<small>${num(mt.n)} of ${num(mt.total)} evaluation images: δ<sub>1</sub> gain ≥ ${Math.round(mt.min_gain * 100)} points and lower AbsRel · hover any tile</small>`,
      () => '<b>Outpaint.</b> Generated surroundings give the model more to see. The query pixels are pasted back untouched.',
      () => '<b>Query only.</b> Metric depth from the frozen model, cropped to the query. One colour scale per scene.',
      () => '<b>ContextDepth.</b> Same pixels, same weights, richer context.',
      () => '<b>Ground truth.</b> What every prediction is scored against.',
      () => '<b>Point clouds, query only.</b> Each query lifted into 3D with the frozen model\'s depth. The dotted surfaces are ground truth: the points miss them.',
      () => '<b>Point clouds, ContextDepth.</b> Same viewpoint, same pixels. The points land on the ground truth.',
      () => '<b>Explore one in 3D.</b> Keep scrolling.'];
    // scroll keyframes, as fractions of the pinned scroll
    const K = { zoomIn: [0.19, 0.26], zoomBack: [0.34, 0.40], sw1: [0.46, 0.55], sw2: [0.58, 0.67], toCloud: [0.70, 0.76], sw3: [0.80, 0.89], dive: [0.93, 1] };

    function layout() {
      if (!W.meta) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2), w = pin.clientWidth, h = pin.clientHeight, N = W.meta.n;
      if (!w || !h) return;                    // not laid out yet (hidden tab); a later resize lays it out
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); W.hover = -1;
      let s = Math.max(Math.sqrt(w * h / (0.75 * N)), 24), cols, rows;
      for (;;) { cols = Math.ceil(w / s); rows = Math.ceil(h / (s * 0.75)); if (cols * rows <= N) break; s *= 1.01; }
      const list = [];
      for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const dx = (x + 0.5 - cols / 2) / (cols / 2), dy = (y + 0.5 - rows / 2) / (rows / 2);
        list.push({ x, y, d: Math.hypot(dx, dy) + Math.atan2(dy, dx) * 1e-4 });
      }
      list.sort((a, b) => a.d - b.d);          // best tiles first: the centre fills before the edges
      const rankAt = new Int32Array(cols * rows);
      list.forEach((c, r) => { rankAt[c.y * cols + c.x] = r; c.d = Math.min(1, c.d / 1.42); });
      Object.assign(W, { cells: list, rankAt, cols, rows, s, th: s * 0.75, ox: (w - cols * s) / 2, oy: (h - rows * s * 0.75) / 2, dpr });
    }

    /* one tile in its current state, drawn into an arbitrary rectangle */
    function tile(r, d, x, y, w, h, alpha, st) {
      const mt = W.meta, R = mt.rgb, D = mt.depth, C = mt.cloud, I = W.img;
      const k = cl(st.zoom * 1.5 - d * 0.5), f = 2 / 3 + k / 3;
      const td = I.q ? cl(st.toDepth * 1.5 - d * 0.5) : 0, tc = I.pq && C ? cl(st.toCloud * 1.5 - d * 0.5) : 0;
      const cut = (line) => cl((line - st.gx(x)) / st.gw(w));
      const part = (im, T, from, to) => {      // a vertical slice [from, to] of sprite r from atlas im
        if (to - from > 0.002) ctx.drawImage(im, (r % T.cols) * T.tw + 0.5 + (T.tw - 1) * from, Math.floor(r / T.cols) * T.th + 0.5, (T.tw - 1) * (to - from), T.th - 1, x + w * from, y, w * (to - from), h);
      };
      if (td < 1) {
        const sx = (r % R.cols) * R.tw, sy = Math.floor(r / R.cols) * R.th;
        ctx.globalAlpha = alpha;
        ctx.drawImage(I.rgb, sx + R.tw * (1 - f) / 2 + 0.5, sy + R.th * (1 - f) / 2 + 0.5, R.tw * f - 1, R.th * f - 1, x, y, w, h);
        if (k > 0.04) {                        // the query pixels inside the outpainted canvas
          const q = (2 / 3) / f;
          ctx.globalAlpha = alpha * k * (1 - td) * 0.72; ctx.lineWidth = Math.max(1, W.dpr * (w > 90 ? 1.6 : 1)); ctx.strokeStyle = '#fff';
          ctx.strokeRect(x + w * (1 - q) / 2, y + h * (1 - q) / 2, w * q, h * q);
        }
      }
      if (td > 0 && tc < 1) {                  // depth: left of a sweep line is the newer layer
        const a2 = I.g && st.sw2 > 0 ? cut(st.x2) : 0, a1 = Math.max(a2, I.c && st.sw1 > 0 ? cut(st.x1) : 0);
        ctx.globalAlpha = alpha * td;
        part(I.g, D, 0, a2); part(I.c, D, a2, a1); part(I.q, D, a1, 1);
      }
      if (tc > 0) {                            // point clouds
        const a3 = I.pc && st.sw3 > 0 ? cut(st.x3) : 0;
        ctx.globalAlpha = alpha * tc;
        part(I.pc, C, 0, a3); part(I.pq, C, a3, 1);
      }
    }

    function draw() {
      if (!W.meta || !W.img.rgb || !W.cells || !W.cells.length) return;
      const p = W.p, dpr = W.dpr, cw = cv.width, ch = cv.height;
      const st = { zoom: ss(p, K.zoomIn[0], K.zoomIn[1]) - ss(p, K.zoomBack[0], K.zoomBack[1]), toDepth: ss(p, K.zoomBack[0], K.zoomBack[1]),
                   toCloud: ss(p, K.toCloud[0], K.toCloud[1]), sw1: lin(p, K.sw1), sw2: lin(p, K.sw2), sw3: lin(p, K.sw3), gx: (x) => x, gw: (w) => w };
      const dive = ss(p, K.dive[0], K.dive[1]), fade = 1 - ss(p, 0.975, 1);
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cw, ch);
      ctx.imageSmoothingQuality = 'high';
      const s = W.s * dpr, th = W.th * dpr, gap = Math.max(1, Math.round(dpr)), born = (performance.now() - W.born) / 1000, c0 = W.cells[0];
      // the wall moves in on its centre for the point clouds, then dives into the middle tile
      const fx = W.ox * dpr + (c0.x + 0.5) * s, fy = W.oy * dpr + (c0.y + 0.5) * th, S = (1 + 0.95 * st.toCloud) * (1 + Math.pow(dive, 3) * 46);
      const inv = (X) => fx + (X - fx) / S;          // screen x -> wall x
      st.x1 = inv(st.sw1 * cw); st.x2 = inv(st.sw2 * cw); st.x3 = inv(st.sw3 * cw);
      Object.assign(W, { S, fx: fx / dpr, fy: fy / dpr });
      if (S !== 1) { ctx.translate(fx, fy); ctx.scale(S, S); ctx.translate(-fx, -fy); }
      const vx0 = inv(0) - s, vx1 = inv(cw), vy0 = fy + (0 - fy) / S - th, vy1 = fy + (ch - fy) / S;
      for (let r = 0; r < W.cells.length; r++) {
        const c = W.cells[r], e = cl((born - c.d * 0.9) / 0.45);
        if (e <= 0) continue;
        const tx = Math.round(W.ox * dpr + c.x * s), ty = Math.round(W.oy * dpr + c.y * th);
        if (tx < vx0 || tx > vx1 || ty < vy0 || ty > vy1) continue;          // off screen while zoomed in
        tile(r, c.d, tx, ty, Math.round(s) - gap, Math.round(th) - gap, e * fade, st);
      }
      if (W.hover >= 0 && dive === 0) {        // lift the hovered tile
        const c = W.cells[W.hover], x = W.ox * dpr + c.x * s, y = W.oy * dpr + c.y * th, g = s * 0.34, x0 = x - g, y0 = y - g * 0.75, w = s + 2 * g, h = th + 1.5 * g;
        ctx.globalAlpha = 1; ctx.save();
        ctx.shadowColor = 'rgba(18,18,40,.5)'; ctx.shadowBlur = 22 * dpr; ctx.shadowOffsetY = 6 * dpr; ctx.fillStyle = '#fff';
        ctx.fillRect(x0 - 2 * dpr, y0 - 2 * dpr, w + 4 * dpr, h + 4 * dpr); ctx.restore();
        st.gx = () => x; st.gw = () => s;       // keep the sweep cut where it is on the tile itself
        tile(W.hover, c.d, x0, y0, w, h, 1, st);
      }
      ctx.globalAlpha = 1;
    }
    function schedule() {
      if (W.pending) return;
      W.pending = true;
      const run = () => { if (!W.pending) return; W.pending = false; draw(); if (performance.now() - W.born < 2600) schedule(); };
      requestAnimationFrame(run); setTimeout(run, 90);
    }

    /* text and controls that sit on top of the wall */
    function chrome() {
      const p = W.p, h0 = 1 - ss(p, 0.012, 0.07);
      hero.style.opacity = h0; hero.style.transform = `translateY(${-(1 - h0) * 46}px)`; hero.style.pointerEvents = h0 < 0.4 ? 'none' : '';
      veil.style.opacity = 1 - ss(p, 0.015, 0.085);
      hint.style.opacity = 1 - ss(p, 0, 0.03);
      const ph = p < K.zoomIn[0] ? 0 : p < K.zoomBack[0] ? 1 : p < K.sw1[0] ? 2 : p < K.sw2[0] ? 3 : p < K.toCloud[0] ? 4 : p < K.sw3[0] ? 5 : p < K.dive[0] ? 6 : 7;
      const on = p > 0.065 && p < 0.99;
      cap.style.opacity = rail.style.opacity = on ? 1 : 0;
      if (W.meta && W.ph !== ph) { W.ph = ph; cap.innerHTML = CAPS[ph](W.meta); $$('li', rail).forEach((li, i) => li.classList.toggle('on', i === ph)); }
      const s = [lin(p, K.sw1), lin(p, K.sw2), lin(p, K.sw3)], act = s.findIndex(v => v > 0 && v < 1);
      sweep.style.opacity = act >= 0 ? 1 : 0;
      if (act >= 0) {
        sweep.style.left = s[act] * 100 + '%'; sweep.classList.toggle('s2', act === 1);
        $('.sw-l', sweep).textContent = act === 1 ? 'Ground truth' : 'ContextDepth'; $('.sw-r', sweep).textContent = act === 1 ? 'ContextDepth' : 'Query only';
      }
    }
    function onScroll() {
      const r = sec.getBoundingClientRect(), span = sec.offsetHeight - pin.clientHeight;
      const p = cl(-r.top / span);
      if (p === W.p && W.ph >= 0) return;
      W.p = p; chrome(); hoverAt(); schedule();
    }

    /* hover: an enlarged view of whatever the tile currently shows */
    function page(kind, i) {
      const store = W.pages[kind];
      if (!store.has(i)) {
        store.set(i, null);
        CD.loadImage(`${base}med_${kind}_${String(i).padStart(2, '0')}.jpg`).then((im) => { store.set(i, im); W.cardKey = ''; showCard(); }).catch(() => store.delete(i));
      }
      return store.get(i);
    }
    function showCard() {
      const r = W.hover, mt = W.meta;
      if (r < 0 || !mt || !W.mouse) { card.hidden = true; W.cardKey = ''; return; }
      const mode = W.p < K.zoomIn[0] ? 'query' : W.p < K.zoomBack[0] ? 'canvas' : W.p < K.toCloud[0] ? 'depth' : 'cloud', key = r + mode, it = mt.items[r], M = mt.med;
      if (key !== W.cardKey) {
        W.cardKey = key;
        const pi = Math.floor(r / M.per), j = r % M.per, x = cardCv.getContext('2d');
        card.classList.toggle('depth', mode === 'depth'); card.classList.toggle('cloud', mode === 'cloud');
        const strip = (kind, T, P, lows, n) => {       // n panels side by side: hover-preview page if loaded, wall sprite meanwhile
          const g = 8, w = P.w, h = P.h, im = page(kind, pi);
          cardCv.width = w * n + g * (n - 1); cardCv.height = h;
          x.fillStyle = '#fff'; x.fillRect(0, 0, cardCv.width, h);
          for (let t = 0; t < n; t++) {
            if (im) x.drawImage(im, ((j % P.cols) * n + t) * w, Math.floor(j / P.cols) * h, w, h, t * (w + g), 0, w, h);
            else if (W.img[lows[t]]) x.drawImage(W.img[lows[t]], (r % T.cols) * T.tw + 0.5, Math.floor(r / T.cols) * T.th + 0.5, T.tw - 1, T.th - 1, t * (w + g), 0, w, h);
          }
        };
        if (mode === 'depth') strip('depth', mt.depth, M.depth, ['q', 'c', 'g'], 3);
        else if (mode === 'cloud') strip('cloud', mt.cloud, M.cloud, ['pq', 'pc'], 2);
        else {
          const w = M.rgb.w, h = M.rgb.h, im = page('rgb', pi), R = mt.rgb, f = mode === 'query' ? 2 / 3 : 1;
          cardCv.width = w; cardCv.height = h;
          if (im) x.drawImage(im, (j % M.rgb.cols) * w + w * (1 - f) / 2, Math.floor(j / M.rgb.cols) * h + h * (1 - f) / 2, w * f, h * f, 0, 0, w, h);
          else x.drawImage(W.img.rgb, (r % R.cols) * R.tw + R.tw * (1 - f) / 2, Math.floor(r / R.cols) * R.th + R.th * (1 - f) / 2, R.tw * f, R.th * f, 0, 0, w, h);
          if (mode === 'canvas') {
            x.setLineDash([9, 7]); x.lineWidth = 4; x.strokeStyle = 'rgba(18,18,40,.5)'; x.strokeRect(w / 6, h / 6, w * 2 / 3, h * 2 / 3);
            x.lineWidth = 2.5; x.strokeStyle = '#fff'; x.strokeRect(w / 6, h / 6, w * 2 / 3, h * 2 / 3); x.setLineDash([]);
          }
        }
        const head = mode === 'depth' ? '<div class="tri"><span>query only</span><span>ContextDepth</span><span>ground truth</span></div>'
          : mode === 'cloud' ? '<div class="tri two"><span>query only</span><span>ContextDepth</span></div>'
          : mode === 'canvas' ? '<div class="tri one"><span>outpainted context · dashed: query pixels</span></div>' : '';
        cardMeta.innerHTML = head + `<b>${mt.datasets[it[0]]}</b> · δ<sub>1</sub> ${(it[1] / 10).toFixed(1)}% → <span class="up">${(it[2] / 10).toFixed(1)}%</span>` +
          `<br>median depth ${m(it[3])} → ${m(it[4])} m · ground truth ${m(it[5])} m` + (mode === 'cloud' ? '<br>dots: ground truth' : '');
      }
      card.hidden = false;
      const b = pin.getBoundingClientRect(), cwid = card.offsetWidth, chei = card.offsetHeight, mx = W.mouse[0] - b.left, my = W.mouse[1] - b.top;
      card.style.left = Math.max(8, mx + 22 + cwid > b.width ? mx - 22 - cwid : mx + 22) + 'px';
      card.style.top = Math.max(64, Math.min(b.height - chei - 8, my + 22 + chei > b.height ? my - 22 - chei : my + 22)) + 'px';
    }
    function hoverAt() {
      let r = -1;
      if (W.mouse && W.cells && W.cells.length && W.p < K.dive[0] && !(W.over && W.over.closest && W.over.closest('.wall-hero a, .wall-hero .btn'))) {
        const b = pin.getBoundingClientRect(), x = W.mouse[0] - b.left, y = W.mouse[1] - b.top, S = W.S || 1;
        const wx = S === 1 ? x : W.fx + (x - W.fx) / S, wy = S === 1 ? y : W.fy + (y - W.fy) / S;
        const cx = Math.floor((wx - W.ox) / W.s), cy = Math.floor((wy - W.oy) / W.th);
        const hb = W.p < 0.04 && hero.getBoundingClientRect();
        const veiled = W.p < 0.04 && (Math.hypot((x / b.width - 0.5) / 0.6, (y / b.height - 0.27) / 0.44) < 0.72 ||      // under the title glow
          (W.mouse[0] > hb.left - 24 && W.mouse[0] < hb.right + 24 && W.mouse[1] > hb.top - 24 && W.mouse[1] < hb.bottom + 24));   // or the title block (narrow screens)
        if (!veiled && cx >= 0 && cy >= 0 && cx < W.cols && cy < W.rows) r = W.rankAt[cy * W.cols + cx];
      }
      if (r !== W.hover) { W.hover = r; schedule(); }
      showCard();
    }
    pin.addEventListener('pointermove', (e) => { if (e.pointerType !== 'mouse') return; W.touch = false; W.mouse = [e.clientX, e.clientY]; W.over = e.target; hoverAt(); });
    pin.addEventListener('pointerdown', (e) => { W.touch = e.pointerType !== 'mouse'; W.mouse = [e.clientX, e.clientY]; W.over = e.target; hoverAt(); });
    pin.addEventListener('pointerleave', (e) => { if (e.pointerType !== 'mouse') return; W.mouse = null; hoverAt(); });
    // A tap shows one tile; on touch screens the card goes away as soon as the page scrolls.
    window.addEventListener('scroll', () => { if (W.touch && W.mouse) { W.mouse = null; hoverAt(); } }, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
    const relay = () => { layout(); W.ph = -1; onScroll(); schedule(); };
    window.addEventListener('resize', relay);
    if ('ResizeObserver' in window) new ResizeObserver(relay).observe(pin);

    Promise.resolve(window.CD_WALL).then((mt) => {
      if (!mt) throw new Error('static/wall/wall.js is missing');
      W.meta = mt; layout(); onScroll();
      return CD.loadImage(base + 'wall_rgb.jpg');
    }).then((im) => {
      W.img.rgb = im; W.born = performance.now(); schedule();
      [['q', 'depth_query'], ['c', 'depth_context'], ['g', 'depth_gt'], ['pq', 'cloud_query'], ['pc', 'cloud_context']]
        .forEach(([k, n]) => CD.loadImage(`${base}wall_${n}.jpg`).then((d) => { W.img[k] = d; schedule(); }).catch(() => {}));
    }).catch(console.error);
  }

  /* ───────────────────────── Demo stage ─────────────────────────
   * One scene played end to end: query only -> outpaint -> select -> result.
   * The 3D cloud, the callouts, the picture-in-picture and the read-out are all
   * functions of one timeline, so the demo can be paused, scrubbed and orbited. */
  function initStage() {
    const root = $('#stage'), mount = $('#st-cloud'), order = window.CD_SCENE_ORDER;
    const PH = [['query', 'Query only', 0, 4], ['outpaint', 'Outpaint', 4, 7], ['select', 'Select', 7, 9.6], ['result', 'Result', 9.6, 14]], DUR = 14;
    const S = { scene: order[0], model: 'da3', mode: '3d', t: reduceMotion ? 11 : 0, playing: !reduceMotion, token: 0, d: null, pair: '', touched: false, userCam: false, visible: false, pos: 0.5, last: {}, busy: false, hudW: 0 };
    const clamp01 = (x) => x < 0 ? 0 : x > 1 ? 1 : x, P = (t, a, b) => clamp01((t - a) / (b - a));
    const put1 = (key, node, html) => { if (S.last[key] !== html) { S.last[key] = html; node.innerHTML = html; } };
    const cls = (key, node, name, on) => { const k = key + name; if (S.last[k] !== on) { S.last[k] = on; node.classList.toggle(name, on); } };

    /* scene strip */
    const strip = $('#st-strip');
    order.forEach((id, i) => {
      const s = SCENES[id], b = el('button');
      b.dataset.v = id; b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', String(i === 0));
      b.innerHTML = `<img src="${url(s, 'thumb.jpg')}" alt="" loading="lazy"><span>${s.title}<small>${s.dataset}</small></span><i></i>`;
      strip.appendChild(b);
    });
    bindSeg(strip, (id) => { S.touched = false; load(id, S.model); });
    bindSeg($('#st-model'), (v) => { S.touched = true; load(S.scene, v); });
    bindSeg($('#st-view'), (v) => { S.mode = v; S.touched = true; applyMode(); });

    /* timeline */
    const track = $('#st-track');
    const segs = PH.map(([k, name, a, b]) => { const e = el('span', 'st-seg', `<em>${name}</em>`); e.style.flex = String(b - a); track.appendChild(e); return e; });
    const head = el('i', 'st-head'); track.appendChild(head);
    const seek = (t) => { S.t = Math.max(0, Math.min(DUR - 0.01, t)); S.touched = true; };
    let scrub = false;
    track.addEventListener('pointerdown', (e) => { scrub = true; track.setPointerCapture(e.pointerId); const r = track.getBoundingClientRect(); seek((e.clientX - r.left) / r.width * DUR); });
    track.addEventListener('pointermove', (e) => { if (!scrub) return; const r = track.getBoundingClientRect(); seek((e.clientX - r.left) / r.width * DUR); });
    track.addEventListener('pointerup', () => { scrub = false; }); track.addEventListener('pointercancel', () => { scrub = false; });
    track.addEventListener('keydown', (e) => { if (e.key === 'ArrowRight') seek(S.t + 0.5); else if (e.key === 'ArrowLeft') seek(S.t - 0.5); });
    const playBtn = $('#st-play');
    const setPlaying = (v) => { S.playing = v; root.classList.toggle('paused', !v); playBtn.setAttribute('aria-label', v ? 'Pause' : 'Play'); };
    playBtn.addEventListener('click', () => { setPlaying(!S.playing); S.touched = true; });
    setPlaying(S.playing);

    /* 3D cloud + orbit */
    const cloud = new CD.Cloud(mount, { fov: 36 }); cloud.dprMax = 1.6;
    if (cloud.failed) { S.mode = 'depth'; setSeg($('#st-view'), 'depth'); $('#st-view button[data-v="3d"]').disabled = true; }
    let drag = null;
    mount.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, yaw: cloud.view.yaw, pitch: cloud.view.pitch }; mount.setPointerCapture(e.pointerId); S.userCam = true; S.touched = true; });
    mount.addEventListener('pointermove', (e) => {
      if (!drag) return;
      cloud.view.yaw = drag.yaw - (e.clientX - drag.x) * 0.006;
      cloud.view.pitch = Math.max(-0.15, Math.min(1.25, drag.pitch + (e.clientY - drag.y) * 0.005));
    });
    const endDrag = () => { drag = null; };
    mount.addEventListener('pointerup', endDrag); mount.addEventListener('pointercancel', endDrag);

    /* callouts anchored in 3D */
    const labels = $('#st-labels'), svg = $('#st-leaders'), NS = 'http://www.w3.org/2000/svg';
    const mk = (tag, c) => { const e = document.createElementNS(NS, tag); if (c) e.setAttribute('class', c); svg.appendChild(e); return e; };
    const G = { ray: mk('line', 'ray'), gap: mk('line', 'gap'), l1: mk('line'), l2: mk('line'), l3: mk('line'), d1: mk('circle'), d2: mk('circle'), d3: mk('circle') };
    const L = { cam: el('span', 'lab cam', 'camera'), pred: el('span', 'lab pred'), gt: el('span', 'lab gt'), gap: el('span', 'lab gap') };
    Object.values(L).forEach(n => labels.appendChild(n));
    let gridLabs = [];
    const lineTo = (ln, a, b) => { ln.setAttribute('x1', a[0]); ln.setAttribute('y1', a[1]); ln.setAttribute('x2', b[0]); ln.setAttribute('y2', b[1]); };
    const dot = (c, p) => { c.setAttribute('cx', p[0]); c.setAttribute('cy', p[1]); c.setAttribute('r', 4.5); };

    /* 2D compare */
    const cmp = $('#cmp'), ca = $('#cmp-a'), cb = $('#cmp-b'), handle = $('#cmp-handle'), probe = $('#probe');
    const setPos = (p) => { S.pos = Math.max(0.02, Math.min(0.98, p)); cb.style.clipPath = `inset(0 0 0 ${S.pos * 100}%)`; handle.style.left = S.pos * 100 + '%'; };
    const drawTo = (target, src) => { target.width = src.width; target.height = src.height; target.getContext('2d').drawImage(src, 0, 0); };
    let slide = false;
    function move(e) {
      const r = cmp.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      if (slide) setPos(x);
      if (!S.d || x < 0 || x > 1 || y < 0 || y > 1) { probe.hidden = true; return; }
      const { gt, Z, sel } = S.d, dq = Z[0], dc = Z[sel];
      const i = Math.min(gt.h - 1, Math.floor(y * gt.h)) * gt.w + Math.min(gt.w - 1, Math.floor(x * gt.w));
      const g = gt.valid[i] ? gt.d[i] : null, rel = (v) => g ? ` <small>${signed((v - g) / g * 100, 0)}%</small>` : '';
      probe.innerHTML = `<span class="r-g">ground truth  <b>${g ? m(g) + ' m' : 'n/a'}</b></span><br>` +
        `<span class="r-q">query only    <b>${m(dq.d[i])} m</b>${rel(dq.d[i])}</span><br><span class="r-c">ContextDepth  <b>${m(dc.d[i])} m</b>${rel(dc.d[i])}</span>`;
      probe.style.whiteSpace = 'pre'; probe.style.left = (x * 100) + '%'; probe.style.top = (y * 100) + '%';
      probe.classList.toggle('flip', x > 0.6); probe.hidden = false;
    }
    cmp.addEventListener('pointerdown', (e) => { slide = true; cmp.setPointerCapture(e.pointerId); move(e); });
    cmp.addEventListener('pointerup', () => { slide = false; }); cmp.addEventListener('pointercancel', () => { slide = false; });
    cmp.addEventListener('pointerleave', () => { probe.hidden = true; }); cmp.addEventListener('pointermove', move);

    const overlayHud = () => getComputedStyle($('#st-hud')).top.indexOf('%') < 0 && $('#st-hud').offsetTop < root.clientHeight;
    function fit2d() {
      if (!S.d) return;
      const s = S.d.s, W = root.clientWidth, H = root.clientHeight, hud = overlayHud() ? $('#st-hud').offsetWidth + 40 : 20;
      const bx = 20, by = 102, bw = W - bx - hud, bh = H - by - 86, ar = s.w / s.h;
      const w = Math.min(bw, bh * ar), h = w / ar;
      cmp.style.cssText = `left:${bx + (bw - w) / 2}px;top:${by + (bh - h) / 2}px;width:${w}px;height:${h}px`;
    }
    function draw2d() {
      const { s, gt, Z, sel } = S.d, err = S.mode === 'error';
      drawTo(ca, err ? CD.colorizeError(Z[0], gt, 0.5) : CD.colorizeDepth(Z[0], s.vis, LUT));
      drawTo(cb, err ? CD.colorizeError(Z[sel], gt, 0.5) : CD.colorizeDepth(Z[sel], s.vis, LUT));
      setPos(S.pos); fit2d();
      const host = $('#st-legend');
      if (err) host.innerHTML = `<div class="ex-legend"><span>0</span><div class="bar" style="background:${CD.errorGradient()}"></div><span>≥ 50% relative error</span></div>`;
      else {
        const l0 = Math.log(s.vis[0]), l1 = Math.log(s.vis[1]) - l0;
        const ticks = CD.depthTicks(s.vis[0], s.vis[1], 6).map(v => `<span style="left:${((Math.log(v) - l0) / l1 * 100).toFixed(2)}%">${v} m</span>`).join('');
        host.innerHTML = `<div class="ex-legend has-ticks"><span>near</span><div class="bar" style="background:${CD.lutGradient(LUT)}">${ticks}</div><span>far</span></div>`;
      }
    }
    function applyMode() {
      const two = S.mode !== '3d';
      root.dataset.mode = two ? '2d' : '3d';
      $('#st-2d').hidden = !two; $('#st-timeline').hidden = two; $('#st-legend').hidden = !two;
      $('#st-hint').textContent = two ? 'drag the divider · hover to read metres' : 'drag to orbit';
      S.last = {};
      if (two && S.d) draw2d();
    }

    /* load one scene for one model */
    async function load(id, mk2) {
      const s = SCENES[id], mm = s.models[mk2], my = ++S.token;
      root.classList.add('loading');
      const names = ['gt', mk2 + '_direct', mk2 + '_c0', mk2 + '_c1', mk2 + '_c2'];
      const [maps, qim] = await Promise.all([Promise.all(names.map(n => depthOf(s, n))), CD.loadMaps(s.id).then((mp) => CD.loadImage(mp.query)), ...[0, 1, 2].map(i => CD.loadImage(url(s, `cand${i}.jpg`)))]);
      if (my !== S.token) return;
      const [gt, d0, c0, c1, c2] = maps, Z = [d0, c0, c1, c2], sel = 1 + mm.sel.choice;
      // anchor: a pixel with ground truth, as close to the image centre as possible
      let ai = -1;
      for (let r = 0; r < Math.max(s.w, s.h) && ai < 0; r += 2) for (let k = 0; k < 16 && ai < 0; k++) {
        const u = Math.round(s.w / 2 + r * Math.cos(k * Math.PI / 8)), v = Math.round(s.h / 2 + r * 0.6 * Math.sin(k * Math.PI / 8));
        if (u >= 0 && v >= 0 && u < s.w && v < s.h && gt.valid[v * s.w + u]) ai = v * s.w + u;
      }
      const au = ai % s.w, av = (ai / s.w) | 0;
      const meds = [mm.direct.med, mm.cand[0].med, mm.cand[1].med, mm.cand[2].med];
      S.d = { s, mm, gt, Z, sel, ai, ray: [(au + 0.5 - s.K[2]) / s.K[0], (av + 0.5 - s.K[3]) / s.K[1]], meds,
              lo: Math.min(s.gt_med, ...meds) / 1.4, hi: Math.max(s.gt_med, ...meds) * 1.4 };
      S.scene = id; S.model = mk2; S.pair = ''; S.userCam = false; S.last = {};
      if (!reduceMotion) S.t = 0;
      if (!cloud.failed) {
        const vals = []; for (let i = 0; i < gt.d.length; i += 7) if (gt.valid[i]) vals.push(gt.d[i]);
        vals.sort((a, b) => a - b);
        cloud.setScene({ w: s.w, h: s.h, K: s.K, rgba: CD.imagePixels(qim, s.w, s.h), zA: Z[0], zB: Z[1], gt, zc: s.gt_med,
          zmax: Math.min(s.enc[1], vals[Math.floor(vals.length * 0.97)] * 1.45), distScale: 2.05 });
        S.d.side = { target: cloud.view.target.slice(), dist: cloud.view.dist };
        gridLabs.forEach(n => n.remove());
        gridLabs = cloud.labels.map(Lb => { const n = el('span', 'lab m', Lb.text); labels.appendChild(n); return n; });
      }
      // picture-in-picture: the input the model receives
      const c = s.canvas, zoom = $('#st-pip-zoom'), kfit = Math.min(c.rw / c.w, c.rh / c.h);
      $('#st-pip-view').style.aspectRatio = `${c.rw} / ${c.rh}`;
      zoom.replaceChildren(...[0, 1, 2].map(i => { const im = new Image(); im.src = url(s, `cand${i}.jpg`); im.alt = ''; im.style.cssText = `width:${c.w / c.rw * 100}%;height:${c.h / c.rh * 100}%;left:${-c.x / c.rw * 100}%;top:${-c.y / c.rh * 100}%`; return im; }));
      S.d.kfit = kfit;
      // candidates
      const cand = $('#st-cands'); cand.replaceChildren();
      const sl = mm.sel;
      for (let i = 0; i < 3; i++) {
        const tag = i === sl.choice ? 'selected' : i === sl.anchor ? 'anchor' : sl.eligible[i] ? 'eligible' : 'rejected';
        cand.appendChild(el('div', 'st-cand', `<span>c${i + 1}</span><span>${m(mm.cand[i].med)} m</span><span>Δ ${signed(sl.responses[i], 3)}</span><span class="bd">${tag}</span>`));
      }
      $('#st-bias').innerHTML = `reference bias ${signed(sl.e_y, 3)}`;
      $('#st-gt').textContent = `ground truth ${m(s.gt_med)} m`;
      const gp = (v) => clamp01((Math.log(v) - Math.log(S.d.lo)) / (Math.log(S.d.hi) - Math.log(S.d.lo))) * 100 + '%';
      S.d.gp = gp; $('.g-gt', root).style.left = gp(s.gt_med);
      setSeg(strip, id); setSeg($('#st-model'), mk2);
      $$('button i', strip).forEach(n => { n.style.width = '0'; });
      if (S.mode !== '3d') draw2d();
      render(S.t);
      root.classList.remove('loading');
    }

    /* everything on screen is a function of t */
    function render(t) {
      if (!S.d) return;
      const { s, mm, gt, Z, sel, ai, ray, meds } = S.d, sl = mm.sel, two = S.mode !== '3d';
      let a = 0, b = 1, mx = 0, state = 0;
      if (two) { a = 0; b = sel; mx = 1; state = 'sel'; t = DUR; }
      else if (t < 4) { /* query only */ }
      else if (t < 5) { mx = ease(P(t, 4, 4.8)); state = mx > 0.5 ? 1 : 0; }
      else if (t < 6) { a = 1; b = 2; mx = ease(P(t, 5, 5.8)); state = mx > 0.5 ? 2 : 1; }
      else if (t < 7.8) { a = 2; b = 3; mx = ease(P(t, 6, 6.8)); state = mx > 0.5 ? 3 : 2; }
      else { a = 3; b = sel; mx = ease(P(t, 7.8, 8.6)); state = mx > 0.5 ? 'sel' : 3; }
      const ph = t < 4 ? 0 : t < 7 ? 1 : t < 9.6 ? 2 : 3;

      if (!two && !cloud.failed) {
        const key = a + '-' + b;
        if (key !== S.pair) { cloud.setZ(Z[a], Z[b]); S.pair = key; }
        cloud.mix = mx;
        cloud.resize();
        cloud.shiftX = -S.hudW / (mount.clientWidth || 1);
        const k = S.userCam ? 1 : ease(P(t, 0.9, 3.0)), zc = s.gt_med, asp = (mount.width / mount.height) || 1.78, side = S.d.side;
        const fovCam = 2 * Math.atan(Math.max((s.h / 2) / s.K[1], ((s.w / 2) / s.K[0]) / asp)) * 1.06, fovSide = 36 * Math.PI / 180;
        cloud.fovy = fovCam + (fovSide - fovCam) * k;
        cloud.view.target = [0, side.target[1] * k, -zc + (side.target[2] + zc) * k];
        cloud.view.dist = zc + (side.dist - zc) * k;
        if (!S.userCam) { cloud.view.yaw = k * (0.74 + 0.2 * Math.sin((t - 3) * 0.36)); cloud.view.pitch = k * 0.34; }
        cloud.showGrid = k > 0.12; cloud.showGhost = k > 0.3;
        cloud.render();

        // callouts
        const dpr = mount.width / mount.clientWidth, W = mount.clientWidth, H = mount.clientHeight;
        const zp = Z[a].d[ai] + (Z[b].d[ai] - Z[a].d[ai]) * mx, zg = gt.d[ai];
        const pj = (z) => { const p = cloud.project([ray[0] * z, -ray[1] * z, -z]); return p && [p[0] / dpr, p[1] / dpr]; };
        const pc = pj(1e-6), pp = pj(zp), pg = pj(zg), show = clamp01((k - 0.5) / 0.3) * (pc && pp && pg ? 1 : 0);
        labels.style.opacity = svg.style.opacity = show;
        if (show > 0) {
          const far = pj(Math.max(zp, zg) * 1.08), ok = Math.max(zp / zg, zg / zp) < 1.25;
          lineTo(G.ray, pc, far); lineTo(G.gap, pp, pg);
          G.gap.style.stroke = ok ? 'var(--good)' : 'var(--bad)';
          const place = (node, p, dx, dy, ln, c) => {
            const x = Math.max(76, Math.min(W - 76 - S.hudW, p[0] + dx)), y = Math.max(96, Math.min(H - 96, p[1] + dy));
            node.style.left = x + 'px'; node.style.top = y + 'px'; lineTo(ln, p, [x, y]); dot(c, p);
          };
          const sideSign = pg[0] >= pp[0] ? 1 : -1;
          place(L.cam, pc, -26, -34, G.l1, G.d1);
          place(L.pred, pp, -sideSign * 86, -44, G.l2, G.d2);
          place(L.gt, pg, sideSign * 92, -58, G.l3, G.d3);
          put1('lp', L.pred, `${state === 0 ? 'query only' : state === 'sel' ? 'ContextDepth' : 'candidate ' + state} <b>${m(zp)} m</b>`);
          put1('lg', L.gt, `ground truth <b>${m(zg)} m</b>`);
          put1('lgap', L.gap, `${signed((zp - zg) / zg * 100, 0)}%`);
          L.gap.style.left = (pp[0] + pg[0]) / 2 + 'px'; L.gap.style.top = (pp[1] + pg[1]) / 2 + 26 + 'px';
          L.gap.style.color = ok ? 'var(--good)' : 'var(--bad)';
        }
        cloud.labels.forEach((Lb, i) => {
          const p = cloud.project(Lb.p), n = gridLabs[i];
          if (!n) return;
          if (!p || k < 0.5) { n.style.display = 'none'; return; }
          n.style.display = ''; n.style.left = p[0] / dpr + 'px'; n.style.top = p[1] / dpr + 'px';
        });
      } else { labels.style.opacity = svg.style.opacity = 0; }

      // caption + read-out
      const d1 = (x) => pct(x.d1), caps = [
        'The frozen model sees only the query pixels.',
        'Outpainted surroundings give three scale hypotheses for the same pixels.',
        'A retrieved reference shows which way the model errs, and picks one candidate.',
        `Same pixels, same weights: δ<sub>1</sub> ${d1(mm.direct)} → ${d1(mm.selector)}.`];
      put1('cap', $('#st-caption'), `<b>${s.title}.</b> ${two ? 'Drag the divider to compare. Hover to read depth in metres.' : caps[ph]}`);
      const cur = state === 0 ? mm.direct : state === 'sel' ? mm.selector : mm.cand[state - 1];
      if (two) {
        put1('med', $('#st-med'), `${m(mm.direct.med)} → ${m(mm.selector.med)} m`);
        put1('d1', $('#st-d1'), `${d1(mm.direct)} → ${d1(mm.selector)}`);
        put1('ar', $('#st-ar'), `${mm.direct.ar.toFixed(3)} → ${mm.selector.ar.toFixed(3)}`);
        put1('st', $('#st-state'), 'query only → ContextDepth');
      } else {
        put1('med', $('#st-med'), `${m(meds[a] + (meds[b] - meds[a]) * mx)} m`);
        put1('d1', $('#st-d1'), d1(cur)); put1('ar', $('#st-ar'), cur.ar.toFixed(3));
        put1('st', $('#st-state'), state === 0 ? 'query only' : state === 'sel' ? `selected · candidate ${sl.choice + 1}` : `candidate ${state}`);
      }
      $('.g-pred', root).style.left = S.d.gp(two ? mm.selector.med : meds[a] + (meds[b] - meds[a]) * mx);
      if (S.last.dstate !== (state === 0 ? 'q' : 'c')) { S.last.dstate = state === 0 ? 'q' : 'c'; root.dataset.state = S.last.dstate; }
      cls('root', root, 'ctx', ph >= 1); cls('root', root, 'sel', t >= 7);
      cls('root', root, 'compact', two);

      // picture-in-picture
      const shown = ph === 0 ? sl.choice : t < 5 ? 0 : t < 6 ? 1 : t < 7.8 ? 2 : sl.choice, kf = S.d.kfit, out = ph >= 1;
      if (S.last.pip !== shown + '|' + out) {
        S.last.pip = shown + '|' + out;
        $$('#st-pip-zoom img').forEach((im, i) => im.classList.toggle('on', i === shown));
        $('#st-pip-zoom').style.transform = `scale(${out ? kf : 1})`;
        $('#st-pip-roi').style.inset = out ? `${(1 - kf) / 2 * 100}%` : '0%';
      }
      put1('ptag', $('#st-pip-tag'), ph === 0 ? 'query only' : t < 7.8 && !two ? `candidate ${shown + 1} of 3` : `selected · candidate ${sl.choice + 1}`);
      $$('.st-cand', root).forEach((row, i) => {
        cls('c' + i, row, 'on', two || t >= 4 + i); cls('c' + i, row, 'cur', !two && state === i + 1);
        cls('c' + i, row, 'badge', two || t >= 7); cls('c' + i, row, 'sel', (two || t >= 7.8) && i === sl.choice);
      });

      // timeline
      head.style.left = (t / DUR * 100) + '%';
      segs.forEach((e, i) => cls('s' + i, e, 'on', i === ph));
      const bar = $(`button[data-v="${S.scene}"] i`, strip); if (bar) bar.style.width = (two ? 0 : t / DUR * 100) + '%';
    }

    let prev = performance.now();
    (function loop(now) {
      requestAnimationFrame(loop);
      const dt = Math.min(0.05, (now - prev) / 1000); prev = now;
      if (!S.d || !S.visible || document.hidden) return;
      if (S.mode === '3d' && S.playing && !scrub) {
        S.t += dt;
        if (S.t >= DUR) {
          if (!S.touched && order.length > 1) {
            S.t = DUR - 0.001;
            if (!S.busy) { S.busy = true; load(order[(order.indexOf(S.scene) + 1) % order.length], S.model).catch(console.error).then(() => { S.busy = false; }); }
          } else S.t = 0;
        }
      }
      render(S.t);
    })(prev);

    onVisible(root, (v) => { if (v && !S.visible && !S.touched && !reduceMotion) S.t = 0; S.visible = v; }, { threshold: 0.05 });
    const relayout = () => {
      const hud = $('#st-hud');
      const over = overlayHud();
      S.hudW = over ? hud.offsetWidth + 24 : 0;
      root.style.marginBottom = over ? '' : hud.offsetHeight + 14 + 'px';
      if (S.mode !== '3d') fit2d();
    };
    window.addEventListener('resize', relayout);
    applyMode(); relayout();
    return load(S.scene, S.model).then(relayout);
  }

  /* ───────────────────────── Method visuals ───────────────────────── */
  async function initMethod() {
    const s = SCENES.dock || Object.values(SCENES)[0], mk = 'da3', mm = s.models[mk], c = s.canvas;
    const ar = `${c.w} / ${c.h}`;

    // 1. outpainted candidates cycling around untouched query pixels
    const v1 = $('#viz-outpaint'); v1.classList.add('viz-out'); v1.style.aspectRatio = ar;
    const imgs = [0, 1, 2].map(i => Object.assign(new Image(), { src: url(s, `cand${i}.jpg`), alt: `Outpainted candidate ${i + 1}`, loading: 'lazy' }));
    imgs.forEach(i => v1.appendChild(i));
    const box = el('div', 'roi', '<span>query pixels pasted back</span>');
    box.style.cssText = `left:${c.x / c.w * 100}%;top:${c.y / c.h * 100}%;width:${c.rw / c.w * 100}%;height:${c.rh / c.h * 100}%`;
    const chip = el('span', 'chip');
    v1.append(box, chip);
    let k = 0;
    const cycle = () => { imgs.forEach((im, i) => im.classList.toggle('on', i === k)); chip.textContent = `candidate ${k + 1} of 3`; k = (k + 1) % 3; };
    cycle();
    let on1 = false; onVisible(v1, (v) => { on1 = v; });
    setInterval(() => { if (on1 && !reduceMotion && !document.hidden) cycle(); }, 2100);

    // 2. one frozen model, several scale hypotheses for the same pixels
    const v2 = $('#viz-predict'); v2.classList.add('viz-pred'); v2.style.aspectRatio = ar;
    const names = [mk + '_direct', mk + '_c0', mk + '_c1', mk + '_c2'];
    const maps = await Promise.all(names.map(n => depthOf(s, n)));
    maps.forEach((d, i) => {
      const cell = el('div', 'cell' + (i === 0 ? ' q' : ''));
      cell.appendChild(CD.colorizeDepth(d, s.vis, LUT));
      const med = i === 0 ? mm.direct.med : mm.cand[i - 1].med;
      cell.appendChild(el('span', '', (i === 0 ? 'query only' : `candidate ${i}`) + ` · ${m(med)} m`));
      v2.appendChild(cell);
    });

    // 3. the reference's directional bias picks the candidate
    const v3 = $('#viz-select'); v3.classList.add('viz-sel'); v3.style.aspectRatio = ar;
    const sel = mm.sel, over = sel.e_y > 0;
    const ref = el('div', 'ref');
    ref.appendChild(Object.assign(new Image(), { src: url(s, 'ref.jpg'), alt: 'Retrieved reference image', loading: 'lazy' }));
    ref.appendChild(el('span', 'chip', 'retrieved reference'));
    const bias = el('div', 'bias', `<b>e<sub>y</sub> = ${signed(sel.e_y, 3)}</b><br>model ${over ? 'over' : 'under'}estimates here`);
    bias.dataset.pop = 'pop-bias'; bias.tabIndex = 0; bias.setAttribute('role', 'button');
    ref.appendChild(bias);
    const list = el('div', 'list');
    list.dataset.pop = 'pop-unknown'; list.tabIndex = 0; list.setAttribute('role', 'button');
    const maxAbs = Math.max(0.012, ...sel.responses.map(Math.abs));
    sel.responses.forEach((r, i) => {
      const row = el('div', 'row' + (i === sel.choice ? ' sel' : sel.eligible[i] ? ' elig' : ''));
      const w = Math.abs(r) / maxAbs * 50;
      row.innerHTML = `<span>c${i + 1}</span><span class="bar"><i style="left:${r >= 0 ? 50 : 50 - w}%;width:${Math.max(w, 1.5)}%"></i></span>` +
        `<span class="mk">${i === sel.choice ? 'selected' : i === sel.anchor ? 'anchor' : sel.eligible[i] ? 'eligible' : 'rejected'}</span>`;
      row.title = `Δ = ${signed(r, 3)} relative to the anchor`;
      list.appendChild(row);
    });
    const note = el('div', '', `Δ<sub>i</sub> vs anchor · keep ${over ? 'Δ < 0' : 'Δ > 0'}`);
    note.style.cssText = 'font:500 10.5px/1.3 var(--mono);color:var(--ink-3);padding:2px 2px 0';
    list.appendChild(note);
    v3.append(ref, list);
  }

  /* ───────────────────────── Results chart ───────────────────────── */
  function initChart() {
    const host = $('#chart'), LO = 45, HI = 90, st = { input: 'crop', cam: 'unknown' };
    const pos = (v) => ((v - LO) / (HI - LO) * 100) + '%';
    const rows = {};
    R.models.forEach((md) => {
      const row = el('div', 'row-ch');
      row.innerHTML = `<div class="name">${md.short || md.name}${md.note ? '<sup>†</sup>' : ''}</div><div class="track">` +
        [50, 60, 70, 80, 90].map(v => `<i class="grid" style="left:${pos(v)}"></i>`).join('') +
        `<i class="span"></i><i class="pt pt-o"></i><i class="pt pt-q"></i><i class="pt pt-s"></i><span class="lab lab-q"></span><span class="lab lab-s"></span></div><div class="gain"></div>`;
      host.appendChild(row);
      rows[md.key] = row;
    });
    const axis = el('div', 'axis');
    axis.innerHTML = `<div></div><div class="ticks">${[50, 60, 70, 80, 90].map(v => `<span style="left:${pos(v)}">${v}</span>`).join('')}</div><div class="cap">Δδ<sub>1</sub></div>`;
    host.appendChild(axis);
    const notes = {
      'crop|unknown': '<b>All six models improve.</b> Depth Anything 3 gains the most: +22.65 points (51.17 → 73.82) with frozen weights.',
      'crop|known': '<b>All six models improve.</b> With known intrinsics, selection reverses the small losses a single outpainting causes on UniDepth V2 and Depth Pro: context helps beyond focal-length estimation.',
      'full|unknown': '<b>Five of six models improve, by smaller margins.</b> Full images already show much of the scene layout (DA3: +5.61 instead of +22.65). Depth Pro<sup>†</sup> loses accuracy without intrinsics; the FOV probe below explains why.',
      'full|known': '<b>All six models improve, with diminishing returns.</b> Selection improves on a single outpainting in every setting.'
    };
    function update() {
      const d = R.main[st.cam][st.input];
      R.models.forEach((md) => {
        const [q, o, s] = d[md.key], row = rows[md.key], neg = s < q;
        const span = $('.span', row);
        span.style.left = pos(Math.min(q, s)); span.style.width = (Math.abs(s - q) / (HI - LO) * 100) + '%';
        span.classList.toggle('neg', neg);
        $('.pt-q', row).style.left = pos(q); $('.pt-o', row).style.left = pos(o);
        const ps = $('.pt-s', row); ps.style.left = pos(s); ps.classList.toggle('neg', neg);
        const lq = $('.lab-q', row), ls = $('.lab-s', row);
        lq.style.left = pos(q); lq.textContent = q.toFixed(2);
        ls.style.left = pos(s); ls.textContent = s.toFixed(2);
        const g = $('.gain', row); g.textContent = signed(s - q, 2); g.classList.toggle('neg', neg);
      });
      $('#chart-note').innerHTML = notes[st.input + '|' + st.cam];
    }
    bindSeg($('#ch-input'), (v) => { st.input = v; update(); });
    bindSeg($('#ch-cam'), (v) => { st.cam = v; update(); });
    update();
  }

  /* ───────────────────────── Tables ───────────────────────── */
  function mix(a, b, t) { return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`; }
  function shade(v, q, isBase) {
    if (isBase) return '#eeeef3';
    const d = v - q;
    if (Math.abs(d) < 2) return d >= 0 ? '#e1e9f7' : '#fff2c9';
    const t = Math.pow(Math.min(1, (Math.abs(d) - 2) / 20), 0.55);
    return d > 0 ? mix([224, 242, 223], [128, 198, 143], t) : mix([251, 225, 221], [232, 150, 150], t);
  }
  function buildTable(host, src, cols, groupLabel) {
    const blocks = [['unknown', 'crop'], ['unknown', 'full'], ['known', 'crop'], ['known', 'full']];
    const n = cols.length;
    let h = `<thead><tr><th class="l" rowspan="3">Model</th>` +
      `<th colspan="${2 * n + 1}">Without GT camera</th><th class="gap"></th><th colspan="${2 * n + 1}">With GT camera</th></tr><tr>` +
      blocks.map((b, i) => `${i === 2 ? '<th class="gap"></th>' : i % 2 ? '<th class="gap"></th>' : ''}<th colspan="${n}">${b[1] === 'crop' ? 'Crop' : 'Full image'}</th>`).join('') + `</tr><tr>` +
      blocks.map((b, i) => `${i ? '<th class="gap"></th>' : ''}` + cols.map(c => `<th>${c}</th>`).join('')).join('') + `</tr></thead><tbody>`;
    R.models.forEach((md) => {
      h += `<tr><td class="l">${md.short || md.name}${md.note ? '<sup>†</sup>' : ''}</td>`;
      blocks.forEach((b, bi) => {
        const vals = src[b[0]][b[1]][md.key], best = Math.max(...vals);
        if (bi) h += '<td class="gap"></td>';
        vals.forEach((v, i) => { h += `<td class="${v === best ? 'best' : ''}"><span style="background:${shade(v, vals[0], i === 0)}">${v.toFixed(2)}</span></td>`; });
      });
      h += '</tr>';
    });
    host.innerHTML = h + '</tbody>';
  }

  /* ───────────────────────── Probe ───────────────────────── */
  // Fixed categorical order: colour follows the model, never its rank.
  const MODEL_COLORS = { moge3: '#2a78d6', da3: '#eb6834', moge2: '#1baf7a', unidepth: '#eda100', unik3d: '#e87ba4', depth_pro: '#008300' };
  function initProbe() {
    const root = $('#pb');
    if (!root) return;
    const F = [0.667, 0.75, 0.833, 0.917, 1], NS = 'http://www.w3.org/2000/svg';
    const st = { step: 0, pos: 0, hi: null, pinned: null, pausedUntil: 0, visible: false };
    const svgEl = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
    const fAt = (p) => { const i = Math.min(3, Math.max(0, Math.floor(p))), u = Math.min(1, Math.max(0, p - i)); return [F[i] + (F[i + 1] - F[i]) * u, F[i + 1] + ((F[i + 2] == null ? 1 : F[i + 2]) - F[i + 1]) * u]; };
    const box = (f) => { const w = 400 * f, h = 300 * f; return [200 - w / 2, 150 - h / 2, w, h]; };
    const rectPath = (b) => `M${b[0]} ${b[1]}h${b[2]}v${b[3]}h${-b[2]}Z`;

    /* stage: what the model sees at this step, and where it is scored */
    const veil = $('#pb-veil'), ring = $('#pb-ring'), rin = $('#pb-in'), rinU = $('#pb-in-u'), rout = $('#pb-out');
    const tagIn = $('#pb-tag-in'), tagOut = $('#pb-tag-out');
    function drawStage(p) {
      const [fi, fo] = fAt(p), bi = box(fi), bo = box(Math.min(1, fo));
      veil.setAttribute('d', 'M0 0h400v300h-400Z' + rectPath(bo));
      ring.setAttribute('d', rectPath(bo) + rectPath(bi));
      [[rin, bi], [rinU, bi], [rout, bo]].forEach(([r, b]) => { r.setAttribute('x', b[0]); r.setAttribute('y', b[1]); r.setAttribute('width', b[2]); r.setAttribute('height', b[3]); });
      tagIn.style.left = (bi[0] + bi[2]) / 4 + '%'; tagIn.style.top = bi[1] / 3 + '%';
      tagOut.style.left = bo[0] / 4 + '%'; tagOut.style.top = bo[1] / 3 + '%';
    }
    /* pictorial equation: delta_1 with the wider input minus delta_1 with the narrower one */
    const wide = $('#pb-wide'), wideIn = $('i', wide), narrow = $('#pb-narrow');
    function drawEq(k) {
      const fi = F[k], fo = F[k + 1], W = 118;
      wide.style.width = W + 'px'; wide.style.backgroundSize = `${120 / fo}% auto`;
      wideIn.style.width = (fi / fo * 100) + '%'; wideIn.style.backgroundSize = `${120 / fi}% auto`;
      narrow.style.width = (W * fi / fo) + 'px'; narrow.style.backgroundSize = `${120 / fi}% auto`;
      $('#pb-wide-l').innerHTML = `δ<sub>1</sub> with Q<sub>${k + 1}</sub>`;
      $('#pb-narrow-l').innerHTML = `δ<sub>1</sub> with Q<sub>${k}</sub>`;
      tagIn.innerHTML = `Q<sub>${k}</sub>`; tagOut.innerHTML = `Q<sub>${k + 1}</sub>`;
    }

    /* charts */
    const W = 360, H = 214, ml = 30, mr = 8, mt = 16, mb = 26, colW = (W - ml - mr) / 4;
    const X = (i) => ml + colW * (i + 0.5);
    function chart(host, readHost, data, ghost, lo, hi, ticks, title) {
      const Y = (v) => mt + (H - mt - mb) * (1 - (v - lo) / (hi - lo));
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title });
      const band = svgEl('rect', { y: mt - 6, width: colW - 6, height: H - mt - mb + 12, rx: 8, class: 'pb-band' }, svg);
      svgEl('rect', { x: ml, y: Y(0), width: W - ml - mr, height: Y(lo) - Y(0), class: 'pb-neg' }, svg);
      ticks.forEach((v) => {
        svgEl('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: v === 0 ? 'pb-zero' : 'pb-grid' }, svg);
        svgEl('text', { x: ml - 6, y: Y(v) + 3.5, 'text-anchor': 'end', class: 'pb-tick' }, svg).textContent = v;
      });
      for (let i = 0; i < 4; i++) svgEl('text', { x: X(i), y: H - 7, 'text-anchor': 'middle', class: 'pb-tick' }, svg).textContent = `Q${i}→Q${i + 1}`;
      const path = (a) => a.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join('');
      const series = {};
      R.models.forEach((md) => {
        const g = svgEl('g', { class: 'pb-series' }, svg), col = MODEL_COLORS[md.key];
        if (ghost) svgEl('path', { d: path(ghost[md.key]), class: 'pb-ghost', stroke: col }, g);
        svgEl('path', { d: path(data[md.key]), class: 'pb-line', stroke: col }, g);
        const dots = data[md.key].map((v, i) => svgEl('circle', { cx: X(i), cy: Y(v), r: 4, fill: col, class: 'pb-dot' }, g));
        series[md.key] = { g, dots };
      });
      host.appendChild(svg);
      // legend + live readout: one entry per model, in fixed order
      const items = {};
      R.models.forEach((md) => {
        const b = el('button', 'pb-item', `<i style="background:${MODEL_COLORS[md.key]}"></i><span>${md.short || md.name}</span><b></b>`);
        b.type = 'button'; b.dataset.k = md.key;
        readHost.appendChild(b); items[md.key] = b;
      });
      // table view for assistive technology
      const tbl = el('table'), tblWrap = el('div', 'sr-only');
      tbl.innerHTML = `<caption>${title}</caption><tr><th>Model</th>${[0, 1, 2, 3].map(i => `<th>Q${i} to Q${i + 1}</th>`).join('')}</tr>` +
        R.models.map(md => `<tr><th>${md.name}</th>${data[md.key].map(v => `<td>${v.toFixed(2)}</td>`).join('')}</tr>`).join('');
      tblWrap.appendChild(tbl); host.appendChild(tblWrap);
      svg.addEventListener('pointermove', (e) => {
        const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * W;
        setStep(Math.max(0, Math.min(3, Math.floor((x - ml) / colW))), true);
      });
      return {
        update() {
          band.setAttribute('x', ml + colW * st.step + 3);
          R.models.forEach((md) => {
            const s = series[md.key], on = !st.hi || st.hi === md.key;
            s.g.classList.toggle('dim', !on); s.g.classList.toggle('hi', st.hi === md.key);
            s.dots.forEach((d, i) => d.setAttribute('r', i === st.step ? 6 : 4));
            const it = items[md.key];
            it.classList.toggle('dim', !on); it.setAttribute('aria-pressed', String(st.pinned === md.key));
            $('b', it).textContent = signed(data[md.key][st.step], 2);
          });
        }
      };
    }
    const charts = [
      chart($('#pb-c1'), $('#pb-r1'), R.probe.known, null, -1, 4, [0, 1, 2, 3, 4], 'Probe 1, context utilization: marginal δ1 gain per expansion step with the GT camera'),
      chart($('#pb-c2'), $('#pb-r2'), R.probe.unknown, R.probe.known, -3, 9, [-3, 0, 3, 6, 9], 'Probe 2, FOV sensitivity: marginal δ1 gain per expansion step with the camera estimated by the model')
    ];
    const update = () => charts.forEach(c => c.update());

    function setStep(k, user) {
      if (user) st.pausedUntil = performance.now() + 9000;
      if (k === st.step) return;
      const from = st.pos; st.step = k;
      setSeg($('#pb-steps'), k); drawEq(k); update();
      tween(0, 1, 520, (t) => { st.pos = from + (k - from) * t; drawStage(st.pos); });
    }
    bindSeg($('#pb-steps'), (v) => setStep(+v, true));
    root.addEventListener('pointerover', (e) => { const b = e.target.closest('.pb-item'); if (b && !st.pinned) { st.hi = b.dataset.k; update(); } });
    root.addEventListener('pointerout', (e) => { if (e.target.closest('.pb-item') && !st.pinned) { st.hi = null; update(); } });
    root.addEventListener('click', (e) => {
      const b = e.target.closest('.pb-item'); if (!b) return;
      st.pinned = st.pinned === b.dataset.k ? null : b.dataset.k; st.hi = st.pinned; update();
    });

    drawStage(0); drawEq(0); update();
    onVisible(root, (v) => { st.visible = v; }, { threshold: 0.25 });
    setInterval(() => {
      if (!st.visible || reduceMotion || document.hidden || performance.now() < st.pausedUntil) return;
      setStep((st.step + 1) % 4, false);
    }, 2600);
  }

  /* ───────────────────────── On-demand formulas ───────────────────────── */
  function initTerms() {
    const pop = el('div', 'pop'); pop.setAttribute('role', 'dialog'); pop.hidden = true;
    document.body.appendChild(pop);
    let cur = null, pinned = false, tOpen = 0, tClose = 0;
    function place(trigger) {
      const r = trigger.getBoundingClientRect(), vw = document.documentElement.clientWidth;
      const w = pop.offsetWidth, h = pop.offsetHeight;
      const left = Math.max(12, Math.min(vw - w - 12, r.left + r.width / 2 - w / 2));
      const below = r.top - h - 14 < 70;
      pop.classList.toggle('below', below);
      pop.style.left = left + window.scrollX + 'px';
      pop.style.top = (below ? r.bottom + 12 : r.top - h - 12) + window.scrollY + 'px';
      pop.style.setProperty('--ax', Math.max(18, Math.min(w - 18, r.left + r.width / 2 - left)) + 'px');
    }
    function open(trigger, pin) {
      clearTimeout(tClose);
      const tpl = document.getElementById(trigger.dataset.pop);
      if (!tpl) return;
      if (cur !== trigger) {
        if (cur) cur.setAttribute('aria-expanded', 'false');
        pop.replaceChildren(tpl.content.cloneNode(true));
        cur = trigger;
      }
      pinned = !!pin;
      pop.hidden = false; trigger.setAttribute('aria-expanded', 'true');
      place(trigger);
      requestAnimationFrame(() => pop.classList.add('open'));
      pop.classList.add('open');
    }
    function close() {
      clearTimeout(tOpen);
      if (!cur) return;
      cur.setAttribute('aria-expanded', 'false'); cur = null; pinned = false;
      pop.classList.remove('open'); pop.hidden = true;
    }
    const trig = (e) => e.target.closest && e.target.closest('[data-pop]');
    document.addEventListener('pointerover', (e) => {
      if (e.pointerType !== 'mouse') return;
      const t = trig(e);
      if (t) { clearTimeout(tClose); clearTimeout(tOpen); if (!(pinned && cur === t)) tOpen = setTimeout(() => open(t, pinned && cur === t), 90); }
      else if (e.target.closest && e.target.closest('.pop')) clearTimeout(tClose);
    });
    document.addEventListener('pointerout', (e) => {
      if (e.pointerType !== 'mouse' || pinned) return;
      if (trig(e) || (e.target.closest && e.target.closest('.pop'))) { clearTimeout(tOpen); clearTimeout(tClose); tClose = setTimeout(close, 180); }
    });
    document.addEventListener('click', (e) => {
      const t = trig(e);
      if (t) { if (pinned && cur === t) close(); else open(t, true); return; }
      if (cur && !(e.target.closest && e.target.closest('.pop'))) close();
    });
    document.addEventListener('focusin', (e) => { const t = trig(e); if (t && !pinned && t.matches(':focus-visible')) open(t, false); });
    document.addEventListener('focusout', (e) => { if (trig(e) && !pinned) close(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close();
      const t = trig(e);
      if (t && t.tagName !== 'BUTTON' && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); if (pinned && cur === t) close(); else open(t, true); }
    });
    window.addEventListener('resize', () => { if (cur) place(cur); });
  }

  /* ───────────────────────── Page chrome ───────────────────────── */
  function initChrome() {
    const nav = $('#nav');
    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
    onScroll(); window.addEventListener('scroll', onScroll, { passive: true });

    const io = new IntersectionObserver((es) => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08, rootMargin: '0px 0px -6% 0px' });
    $$('[data-reveal]').forEach(n => io.observe(n));

    const links = $$('.nav-links a'), map = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
    const spy = new IntersectionObserver((es) => es.forEach(e => {
      if (e.isIntersecting) { links.forEach(a => a.classList.remove('active')); const a = map.get(e.target.id); if (a) a.classList.add('active'); }
    }), { rootMargin: '-45% 0px -50% 0px' });
    map.forEach((a, id) => { const sec = document.getElementById(id); if (sec) spy.observe(sec); });

    $$('.count').forEach((n) => {
      const to = +n.dataset.to, dec = +n.dataset.dec || 0, pre = n.dataset.prefix || '';
      let done = false;
      onVisible(n, (v) => { if (v && !done) { done = true; tween(0, to, 1400, (x) => { n.textContent = pre + x.toFixed(dec); }); } }, { threshold: 0.6 });
    });

    const copier = (btn, source) => btn.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(source.textContent); }
      catch (e) { const r = document.createRange(); r.selectNodeContents(source); const s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand('copy'); s.removeAllRanges(); }
      btn.textContent = 'Copied'; btn.classList.add('done');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1600);
    });
    copier($('#copy-bib'), $('#bib-text'));

    // Agent skill (present in the public build only): one prompt per tab.
    const prompts = $('#skill-prompts');
    if (prompts) {
      const data = JSON.parse(prompts.textContent), text = $('#skill-text'), note = $('#skill-note'), tabs = $$('.skill-tabs button');
      const show = (key) => {
        text.textContent = data[key].prompt; note.textContent = data[key].note;
        tabs.forEach(b => b.setAttribute('aria-selected', String(b.dataset.skill === key)));
      };
      tabs.forEach(b => b.addEventListener('click', () => show(b.dataset.skill)));
      show('setup');
      copier($('#copy-skill'), text);
    }
  }

  /* ───────────────────────── Boot ───────────────────────── */
  async function boot() {
    initChrome();
    initWall();
    initChart();
    buildTable($('#tbl-main'), R.main, ['Query only', 'Outpaint', 'Outpaint + Select']);
    buildTable($('#tbl-banks'), R.banks, ['Q', 'O', 'M', 'C', 'D']);
    initProbe();
    initTerms();
    const data = window.CD_SCENES;
    if (!data) { console.error('static/data/scenes.js is missing'); return; }
    LUT = data.lut_spectral;
    data.scenes.forEach(s => { SCENES[s.id] = s; });
    window.CD_SCENE_ORDER = data.scenes.map(s => s.id);
    initStage().catch(console.error);
    let methodDone = false;
    onVisible($('#method'), (v) => { if (v && !methodDone) { methodDone = true; initMethod().catch(console.error); } }, { rootMargin: '300px 0px' });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
