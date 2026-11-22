/* ContextDepth project page: metric depth decoding, colour mapping and per-pixel metrics.
 *
 * Depth maps are shipped as PNGs holding 12-bit log-depth:
 *   R = high 8 bits, G = low 4 bits (x17), B = validity mask.
 * depth = exp(ln(lo) + q / 4095 * (ln(hi) - ln(lo))) with [lo, hi] = scene.enc (metres).
 * Decoding in the browser keeps one shared metric colour scale per scene and lets
 * the page read real values in metres under the cursor.
 */
(function (root) {
  'use strict';
  const CD = root.CD = root.CD || {};
  const cache = new Map();

  CD.loadImage = function (url) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.decoding = 'async';
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('Failed to load ' + url));
      im.src = url;
    });
  };

  CD.loadDepth = function (url, enc, key) {
    key = key || url;
    if (!cache.has(key)) {
      cache.set(key, (async () => {
        const im = await CD.loadImage(url);
        const w = im.naturalWidth, h = im.naturalHeight;
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(im, 0, 0);
        const px = ctx.getImageData(0, 0, w, h).data;
        const d = new Float32Array(w * h), valid = new Uint8Array(w * h);
        const l0 = Math.log(enc[0]), l1 = Math.log(enc[1]) - l0;
        for (let i = 0, j = 0; i < d.length; i++, j += 4) {
          const q = (px[j] << 4) | Math.round(px[j + 1] / 17);
          d[i] = Math.exp(l0 + (q / 4095) * l1);
          valid[i] = px[j + 2] > 127 ? 1 : 0;
        }
        return { w, h, d, valid };
      })());
    }
    return cache.get(key);
  };

  /* Each scene ships its depth maps and its query image as data: URLs in static/scenes/<id>/maps.js.
   * A <script> tag loads from disk where fetch() cannot, and pixels of a data: URL image can be read
   * back even when the page itself was opened from a local file. */
  const bundles = new Map();
  CD.loadMaps = function (id, base) {
    if (root.CD_MAPS && root.CD_MAPS[id]) return Promise.resolve(root.CD_MAPS[id]);
    if (!bundles.has(id)) {
      bundles.set(id, new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = (base || '') + 'static/scenes/' + id + '/maps.js';
        s.onload = () => (root.CD_MAPS && root.CD_MAPS[id]) ? resolve(root.CD_MAPS[id]) : reject(new Error('maps.js did not register scene ' + id));
        s.onerror = () => reject(new Error('Failed to load depth maps of scene ' + id));
        document.head.appendChild(s);
      }));
    }
    return bundles.get(id);
  };
  /* One depth map of a scene, decoded to metres. Reference maps (ref_*) use the reference's own range. */
  CD.sceneDepth = function (scene, name, base) {
    return CD.loadMaps(scene.id, base).then((mp) => CD.loadDepth(mp[name], name.indexOf('ref_') === 0 ? scene.ref.enc : scene.enc, scene.id + '/' + name));
  };

  /* RGBA pixels of an image resampled to w x h (used to colour point clouds). */
  CD.imagePixels = function (im, w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(im, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h).data;
  };

  function canvasFor(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  const INVALID = [236, 236, 242];

  /* Depth -> colour on a log scale shared by every map of a scene. */
  CD.colorizeDepth = function (map, vis, lut, useMask) {
    const c = canvasFor(map.w, map.h), ctx = c.getContext('2d');
    const im = ctx.createImageData(map.w, map.h), o = im.data;
    const l0 = Math.log(vis[0]), k = 255 / (Math.log(vis[1]) - l0);
    for (let i = 0, j = 0; i < map.d.length; i++, j += 4) {
      if (useMask && !map.valid[i]) { o[j] = INVALID[0]; o[j + 1] = INVALID[1]; o[j + 2] = INVALID[2]; o[j + 3] = 255; continue; }
      let t = (Math.log(map.d[i]) - l0) * k;
      t = t < 0 ? 0 : t > 255 ? 255 : t | 0;
      const col = lut[t];
      o[j] = col[0]; o[j + 1] = col[1]; o[j + 2] = col[2]; o[j + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return c;
  };

  // Sequential ramp for absolute relative error: white (0) -> amber -> red -> deep crimson (>= max).
  const ERR_STOPS = [[0, [255, 255, 255]], [0.12, [255, 240, 214]], [0.32, [253, 196, 125]], [0.55, [244, 124, 79]],
                     [0.78, [214, 60, 74]], [1, [120, 18, 62]]];
  function ramp(stops, t) {
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    for (let s = 1; s < stops.length; s++) {
      if (t <= stops[s][0]) {
        const a = stops[s - 1], b = stops[s], u = (t - a[0]) / (b[0] - a[0]);
        return [a[1][0] + (b[1][0] - a[1][0]) * u, a[1][1] + (b[1][1] - a[1][1]) * u, a[1][2] + (b[1][2] - a[1][2]) * u];
      }
    }
    return stops[stops.length - 1][1];
  }
  CD.errorColor = (t) => ramp(ERR_STOPS, t);

  /* |pred - gt| / gt, clipped at max. Pixels without ground truth are grey. */
  CD.colorizeError = function (pred, gt, max) {
    max = max || 0.5;
    const c = canvasFor(gt.w, gt.h), ctx = c.getContext('2d');
    const im = ctx.createImageData(gt.w, gt.h), o = im.data;
    for (let i = 0, j = 0; i < gt.d.length; i++, j += 4) {
      let col = INVALID;
      if (gt.valid[i]) col = ramp(ERR_STOPS, Math.abs(pred.d[i] - gt.d[i]) / gt.d[i] / max);
      o[j] = col[0]; o[j + 1] = col[1]; o[j + 2] = col[2]; o[j + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return c;
  };

  /* Change in per-pixel relative error, as in the paper's qualitative figure:
   * green = error reduced, pink = error increased, square-root scale clipped at +-clip. */
  CD.colorizeGain = function (before, after, gt, clip) {
    clip = clip || 0.2;
    const G = [11, 97, 47], P = [220, 100, 118];
    const c = canvasFor(gt.w, gt.h), ctx = c.getContext('2d');
    const im = ctx.createImageData(gt.w, gt.h), o = im.data;
    for (let i = 0, j = 0; i < gt.d.length; i++, j += 4) {
      let col = INVALID;
      if (gt.valid[i]) {
        const g = gt.d[i];
        const s = (Math.abs(before.d[i] - g) - Math.abs(after.d[i] - g)) / g;
        const t = Math.sqrt(Math.min(1, Math.abs(s) / clip)), e = s >= 0 ? G : P;
        col = [255 + (e[0] - 255) * t, 255 + (e[1] - 255) * t, 255 + (e[2] - 255) * t];
      }
      o[j] = col[0]; o[j + 1] = col[1]; o[j + 2] = col[2]; o[j + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return c;
  };

  /* delta_1 accuracy, AbsRel and median depth over valid ground-truth pixels (no alignment). */
  CD.metrics = function (pred, gt) {
    let n = 0, inl = 0, ar = 0;
    const vals = [];
    for (let i = 0; i < gt.d.length; i++) {
      if (!gt.valid[i]) continue;
      const p = pred.d[i], g = gt.d[i];
      n++; ar += Math.abs(p - g) / g;
      if (Math.max(p / g, g / p) < 1.25) inl++;
      vals.push(p);
    }
    vals.sort((a, b) => a - b);
    return { d1: inl / n, ar: ar / n, med: vals[vals.length >> 1], n };
  };

  CD.fmtMeters = function (v) {
    return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
  };

  /* "Nice" tick values on a log axis between lo and hi (metres). */
  CD.depthTicks = function (lo, hi, maxTicks) {
    const cands = [0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200];
    let ticks = cands.filter(v => v >= lo * 1.02 && v <= hi * 0.98);
    maxTicks = maxTicks || 6;
    while (ticks.length > maxTicks) ticks = ticks.filter((_, i) => i % 2 === 0);
    return ticks;
  };

  /* CSS gradient string for a LUT (used by colour bars). */
  CD.lutGradient = function (lut, dir) {
    const stops = [];
    for (let i = 0; i <= 16; i++) {
      const c = lut[Math.min(255, Math.round(i * 255 / 16))];
      stops.push(`rgb(${c[0]},${c[1]},${c[2]}) ${(i * 100 / 16).toFixed(1)}%`);
    }
    return `linear-gradient(${dir || 'to right'}, ${stops.join(', ')})`;
  };
  CD.errorGradient = function (dir) {
    const stops = [];
    for (let i = 0; i <= 12; i++) {
      const c = ramp(ERR_STOPS, i / 12);
      stops.push(`rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0}) ${(i * 100 / 12).toFixed(1)}%`);
    }
    return `linear-gradient(${dir || 'to right'}, ${stops.join(', ')})`;
  };
})(window);
