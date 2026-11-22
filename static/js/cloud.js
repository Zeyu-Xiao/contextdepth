/* ContextDepth project page: a small dependency-free WebGL point-cloud viewer.
 *
 * Every pixel is un-projected along its camera ray with the ground-truth intrinsics,
 * so two depth maps of the same image differ only by how far each point sits along
 * its ray. `mix` morphs the cloud between depth A (query only) and depth B (ContextDepth);
 * ground truth is drawn as a sparse dark lattice the cloud should land on.
 */
(function (root) {
  'use strict';
  const CD = root.CD = root.CD || {};

  const mat = {
    persp(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    },
    lookAt(e, t, up) {
      let zx = e[0] - t[0], zy = e[1] - t[1], zz = e[2] - t[2];
      let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
      let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
      l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
      const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
      return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
        -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
    },
    mul(a, b) {
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
      return o;
    }
  };

  const VS = `
    attribute vec2 aRay; attribute float aZa; attribute float aZb; attribute vec3 aCol;
    uniform mat4 uMVP; uniform float uMix, uPt, uZmax, uTintAmt; uniform vec3 uTint;
    varying vec3 vCol; varying float vSize;
    void main() {
      float z = mix(aZa, aZb, uMix);
      gl_Position = uMVP * vec4(aRay.x * z, -aRay.y * z, -z, 1.0);
      gl_PointSize = clamp(uPt * z / max(gl_Position.w, 1e-4), 1.0, 16.0);
      if (z > uZmax || z <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      vCol = mix(aCol, uTint, uTintAmt); vSize = gl_PointSize;
    }`;
  const FS = `
    precision mediump float; varying vec3 vCol; varying float vSize;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      if (vSize > 4.5 && dot(d, d) > 0.25) discard;   // small points stay square so they tile without gaps
      gl_FragColor = vec4(vCol, 1.0);
    }`;
  const LVS = `attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMVP; varying vec3 vCol;
    void main() { gl_Position = uMVP * vec4(aPos, 1.0); vCol = aCol; }`;
  const LFS = `precision mediump float; varying vec3 vCol; void main() { gl_FragColor = vec4(vCol, 1.0); }`;

  function program(gl, vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  function niceStep(span, maxLines) {
    const steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100];
    for (const s of steps) if (span / s <= maxLines) return s;
    return 200;
  }

  class Cloud {
    constructor(canvas, opts) {
      opts = opts || {};
      this.canvas = canvas;
      this.fovy = (opts.fov || 36) * Math.PI / 180;
      this.view = { yaw: 0.7, pitch: 0.3, dist: 1, target: [0, 0, -1] };
      this.mix = 0; this.showGhost = true; this.showGrid = true;
      this.labels = [];
      let gl = null;
      try { gl = canvas.getContext('webgl', { antialias: true, alpha: false, preserveDrawingBuffer: true }); } catch (e) { /* no WebGL */ }
      this.gl = gl;
      if (!gl) { this.failed = true; return; }
      this.pp = program(gl, VS, FS);
      this.lp = program(gl, LVS, LFS);
      this.buf = { main: gl.createBuffer(), col: gl.createBuffer(), ghost: gl.createBuffer(), lines: gl.createBuffer() };
      gl.enable(gl.DEPTH_TEST);
      gl.clearColor(1, 1, 1, 1);
    }

    /* scene: { w, h, K:[fx,fy,cx,cy], rgba, zA:{d}, zB:{d}, gt:{d,valid}|null, zc (metres), zmax } */
    setScene(s) {
      if (this.failed) return;
      const gl = this.gl, w = s.w, h = s.h, [fx, fy, cx, cy] = s.K;
      const stride = s.stride || 1;
      const n = Math.ceil(w / stride) * Math.ceil(h / stride);
      const geo = new Float32Array(n * 4), col = new Uint8Array(n * 3);
      let k = 0;
      for (let v = 0; v < h; v += stride) for (let u = 0; u < w; u += stride) {
        const i = v * w + u;
        geo[k * 4] = (u + 0.5 - cx) / fx; geo[k * 4 + 1] = (v + 0.5 - cy) / fy;
        geo[k * 4 + 2] = s.zA.d[i]; geo[k * 4 + 3] = s.zB.d[i];
        col[k * 3] = s.rgba[i * 4]; col[k * 3 + 1] = s.rgba[i * 4 + 1]; col[k * 3 + 2] = s.rgba[i * 4 + 2];
        k++;
      }
      this.n = k; this.geo = geo;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.main); gl.bufferData(gl.ARRAY_BUFFER, geo, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.col); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);

      // Ground truth as a sparse lattice.
      this.ng = 0;
      if (s.gt) {
        const gs = Math.max(2, Math.round(w / 150));
        const g = [];
        for (let v = (gs >> 1); v < h; v += gs) for (let u = (gs >> 1); u < w; u += gs) {
          const i = v * w + u;
          if (!s.gt.valid[i]) continue;
          g.push((u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, s.gt.d[i], s.gt.d[i]);
        }
        this.ng = g.length / 4;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.ghost); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(g), gl.STATIC_DRAW);
      }
      this.fx = fx; this.stride = stride; this.zmax = s.zmax; this.zc = s.zc;

      // Metric floor grid + capture-camera frustum.
      const hx = (w / 2) / fx, hy = (h / 2) / fy;
      const floorY = -(hy * s.zc) * 1.02;                       // GL y of the lower image edge at the reference depth
      const zEnd = Math.min(s.zmax, s.zc * 2.1);
      const step = niceStep(zEnd, 7);
      const xExt = hx * zEnd * 1.05;
      const L = [], grid = [0.82, 0.82, 0.88], gridStrong = [0.62, 0.62, 0.72], acc = [0.357, 0.278, 0.878];
      const line = (a, b, c) => L.push(a[0], a[1], a[2], c[0], c[1], c[2], b[0], b[1], b[2], c[0], c[1], c[2]);
      this.labels = [];
      for (let z = step; z <= zEnd + 1e-6; z += step) {
        const xw = Math.min(xExt, hx * z * 1.35 + step * 0.4);
        line([-xw, floorY, -z], [xw, floorY, -z], gridStrong);
        this.labels.push({ p: [-xw, floorY, -z], text: (step < 1 ? z.toFixed(step < 0.1 ? 2 : (step * 10) % 1 ? 2 : 1) : z.toFixed(0)) + ' m' });
      }
      line([0, floorY, 0], [-hx * zEnd * 1.35, floorY, -zEnd], grid);
      line([0, floorY, 0], [hx * zEnd * 1.35, floorY, -zEnd], grid);
      line([0, floorY, 0], [0, floorY, -zEnd], grid);
      const fz = Math.min(s.zc * 0.16, step * 0.8), c4 = [[-hx, hy], [hx, hy], [hx, -hy], [-hx, -hy]].map(([x, y]) => [x * fz, y * fz, -fz]);
      for (let i = 0; i < 4; i++) { line([0, 0, 0], c4[i], acc); line(c4[i], c4[(i + 1) % 4], acc); }
      this.nl = L.length / 6;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.lines); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(L), gl.STATIC_DRAW);
      this.cameraLabel = { p: [0, hy * fz * 1.6, 0], text: 'camera' };

      this.view.target = [0, floorY * 0.25, -s.zc * 0.82];
      this.view.dist = s.zc * (s.distScale || 1.75);
    }

    /* Swap the two depth maps the cloud morphs between (same rays, same colours; stride 1 only). */
    setZ(zA, zB) {
      if (this.failed || !this.geo || this.stride !== 1) return;
      const g = this.geo, n = this.n, gl = this.gl;
      for (let k = 0; k < n; k++) { g[k * 4 + 2] = zA.d[k]; g[k * 4 + 3] = zB.d[k]; }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.main); gl.bufferData(gl.ARRAY_BUFFER, g, gl.DYNAMIC_DRAW);
    }

    resize() {
      const c = this.canvas, dpr = Math.min(root.devicePixelRatio || 1, this.dprMax || 2);
      const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
      if (w && h && (c.width !== w || c.height !== h)) { c.width = w; c.height = h; }
    }

    mvp() {
      const v = this.view, cp = Math.cos(v.pitch), t = v.target;
      const eye = [t[0] + v.dist * Math.sin(v.yaw) * cp, t[1] + v.dist * Math.sin(v.pitch), t[2] + v.dist * Math.cos(v.yaw) * cp];
      const near = Math.max(0.01, v.dist * 0.02), far = v.dist * 60;
      const m = mat.mul(mat.persp(this.fovy, this.canvas.width / this.canvas.height, near, far), mat.lookAt(eye, t, [0, 1, 0]));
      const s = this.shiftX || 0;   // lens shift: keeps the scene centred in the part of the canvas not covered by overlays
      if (s) { m[0] += s * m[3]; m[4] += s * m[7]; m[8] += s * m[11]; m[12] += s * m[15]; }
      return m;
    }

    render() {
      if (this.failed || !this.n) return;
      const gl = this.gl, W = this.canvas.width, H = this.canvas.height;
      gl.viewport(0, 0, W, H);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const mvp = this._mvp = this.mvp();
      const focal = (H / 2) / Math.tan(this.fovy / 2);

      if (this.showGrid) {
        gl.useProgram(this.lp);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.lp, 'uMVP'), false, mvp);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.lines);
        const p = gl.getAttribLocation(this.lp, 'aPos'), c = gl.getAttribLocation(this.lp, 'aCol');
        gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p, 3, gl.FLOAT, false, 24, 0);
        gl.enableVertexAttribArray(c); gl.vertexAttribPointer(c, 3, gl.FLOAT, false, 24, 12);
        gl.drawArrays(gl.LINES, 0, this.nl);
        gl.disableVertexAttribArray(p); gl.disableVertexAttribArray(c);
      }

      gl.useProgram(this.pp);
      const U = (n) => gl.getUniformLocation(this.pp, n), A = (n) => gl.getAttribLocation(this.pp, n);
      gl.uniformMatrix4fv(U('uMVP'), false, mvp);
      gl.uniform1f(U('uZmax'), this.zmax);
      const aRay = A('aRay'), aZa = A('aZa'), aZb = A('aZb'), aCol = A('aCol');
      gl.enableVertexAttribArray(aRay); gl.enableVertexAttribArray(aZa); gl.enableVertexAttribArray(aZb);

      // colour cloud
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.main);
      gl.vertexAttribPointer(aRay, 2, gl.FLOAT, false, 16, 0);
      gl.vertexAttribPointer(aZa, 1, gl.FLOAT, false, 16, 8);
      gl.vertexAttribPointer(aZb, 1, gl.FLOAT, false, 16, 12);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.col);
      gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.uniform1f(U('uMix'), this.mix);
      gl.uniform1f(U('uPt'), this.stride * focal / this.fx * 1.45);
      gl.uniform1f(U('uTintAmt'), 0); gl.uniform3f(U('uTint'), 0, 0, 0);
      gl.drawArrays(gl.POINTS, 0, this.n);
      gl.disableVertexAttribArray(aCol);

      // ground-truth lattice
      if (this.showGhost && this.ng) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.ghost);
        gl.vertexAttribPointer(aRay, 2, gl.FLOAT, false, 16, 0);
        gl.vertexAttribPointer(aZa, 1, gl.FLOAT, false, 16, 8);
        gl.vertexAttribPointer(aZb, 1, gl.FLOAT, false, 16, 12);
        gl.vertexAttrib3f(aCol, 0, 0, 0);
        gl.uniform1f(U('uMix'), 0);
        gl.uniform1f(U('uPt'), focal / this.fx * 1.15);
        gl.uniform1f(U('uTintAmt'), 1); gl.uniform3f(U('uTint'), 0.07, 0.07, 0.13);
        gl.drawArrays(gl.POINTS, 0, this.ng);
      }
      gl.disableVertexAttribArray(aRay); gl.disableVertexAttribArray(aZa); gl.disableVertexAttribArray(aZb);
    }

    /* Project a point (viewer coordinates) to canvas pixels; null when behind the camera. */
    project(p) {
      const m = this._mvp || this.mvp();
      const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
      const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
      if (w <= 0) return null;
      return [(x / w * 0.5 + 0.5) * this.canvas.width, (1 - (y / w * 0.5 + 0.5)) * this.canvas.height];
    }
  }

  CD.Cloud = Cloud;
})(window);
