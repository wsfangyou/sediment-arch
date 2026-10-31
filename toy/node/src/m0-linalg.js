// ============================================================
// M0 线性代数工具（纯 JS，无外部依赖，全部真实计算）
// ============================================================

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function makeRandn(seed) {
  const u = mulberry32(seed);
  return () => { // Box–Muller
    let s = 0, c = 0;
    do { s = u() * 2; c = u() * 2; } while (s >= 1 || s === 0);
    return Math.sqrt(-2 * Math.log(s)) * Math.cos(2 * Math.PI * c);
  };
}

export const zeros = (r, c) => Array.from({ length: r }, () => new Float64Array(c));
export const rows = (m) => m.length, cols = (m) => m[0].length;

export function matmul(A, B) {
  const n = rows(A), k = cols(A), m = cols(B);
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i], Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p]; if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < m; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}
export function transpose(A) {
  const C = zeros(cols(A), rows(A));
  for (let i = 0; i < rows(A); i++) for (let j = 0; j < cols(A); j++) C[j][i] = A[i][j];
  return C;
}
export function addM(A, B) {
  const C = zeros(rows(A), cols(A));
  for (let i = 0; i < rows(A); i++) for (let j = 0; j < cols(A); j++) C[i][j] = A[i][j] + B[i][j];
  return C;
}
export function scaleM(A, s) {
  return A.map(r => Float64Array.from(r, v => v * s));
}
export function eye(n) {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}
export function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
export function norm(a) { return Math.sqrt(dot(a, a)); }
export function cosine(a, b) { const na = norm(a), nb = norm(b); return na * nb === 0 ? 0 : dot(a, b) / (na * nb); }
export function unit(a) { const n = norm(a); return Float64Array.from(a, v => (n ? v / n : 0)); }

// ---- 对称正定求解（Cholesky）——岭回归正规方程用 ----
export function cholesky(A) {
  const n = rows(A), L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) { if (s <= 1e-12) s = 1e-12; L[i][j] = Math.sqrt(s); }
      else L[i][j] = s / L[j][j];
    }
  }
  return L;
}
export function cholSolve(L, b) {
  const n = L.length; const y = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}
/** 求 X 使 A X ≈ B（按列解，A 对称正定） */
export function cholSolveMat(L, B) {
  const m = cols(B), X = zeros(L.length, m);
  for (let j = 0; j < m; j++) {
    const b = new Float64Array(rows(B));
    for (let i = 0; i < rows(B); i++) b[i] = B[i][j];
    const x = cholSolve(L, b);
    for (let i = 0; i < x.length; i++) X[i][j] = x[i];
  }
  return X;
}

/** 岭回归：求 W 使 X W ≈ Y，W = (XᵀX+λI)⁻¹XᵀY。X: n×d, Y: n×e → W: d×e */
export function ridge(X, Y, lambda) {
  const Xt = transpose(X);
  const XtX = matmul(Xt, X);
  const reg = addM(XtX, scaleM(eye(XtX.length), lambda));
  const L = cholesky(reg);
  const XtY = matmul(Xt, Y);
  return cholSolveMat(L, XtY);
}

/** 正交基（修正 Gram–Schmidt），用于参考骨干的规范帧 */
export function gramSchmidt(vecs) {
  const out = [];
  for (const v0 of vecs) {
    let v = Float64Array.from(v0);
    for (const q of out) { const c = dot(v, q); for (let i = 0; i < v.length; i++) v[i] -= c * q[i]; }
    const nrm = norm(v);
    if (nrm < 1e-8) continue;
    out.push(Float64Array.from(v, x => x / nrm));
  }
  return out;
}

// ---- 量化（MSP 槽位 Q8，报告 §6.1 口径） ----
export function quantInt8(vec) {
  let mx = 0; for (const v of vec) mx = Math.max(mx, Math.abs(v));
  const scale = mx / 127 || 1e-9;
  const q = new Int8Array(vec.length);
  for (let i = 0; i < vec.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round(vec[i] / scale)));
  return { q, scale };
}
export function dequantInt8({ q, scale }) {
  return Float64Array.from(q, v => v * scale);
}
