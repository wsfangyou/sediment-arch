// ============================================================
// M4 生成式插件 / 基因化（方案 §2.4、§2.2 通道 3）
// 真实计算：同域 LoRA-B 矩阵群 → 中心化 → 主成分（迭代法，无库依赖）
// → 潜码（K 维 float32，KB 级）→ 重建近似原参数。
// 主张检验（L4 的信息学前提）：技能参数可压到 KB 级且
// 重建余弦 / 下游代理任务分保持 ≈ 原文件水平。
// 诚实边界：真实"超网络从 prompt 嵌入生成参数"（HyperTTS 式）
// 需训练骨干网络；本原型以码本最近邻生成验证机制形状与压缩保真度。
// ============================================================
import { unit, makeRandn, cosine } from "./m0-linalg.js";

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const flatOf = (B) => Float64Array.from(B);

/** 技能库：d×d LoRA-B（真实低秩结构 + 噪声），共 n 个 */
export function makeLoRAs(n, d = 12, r = 3, seed = 7) {
  const randn = makeRandn(seed);
  const bases = Array.from({ length: r }, () => Float64Array.from({ length: d }, () => randn()));
  const skills = [];
  for (let s = 0; s < n; s++) {
    const B = new Float64Array(d * d);
    const rowCoef = Array.from({ length: d }, () => Float64Array.from({ length: r }, () => randn()));
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
      let v = 0;
      for (let k = 0; k < r; k++) v += rowCoef[i][k] * bases[k][j];
      B[i * d + j] = v + 0.05 * randn();
    }
    skills.push({ id: `skill${s}`, d, B });
  }
  return skills;
}

/** 中心化矩阵的主成分（幂迭代于 ZᵀZ 的等价形式：v ← Zᵀ(Zv)，n≪d² 时更快） */
export function topComponents(flats, comp = 4) {
  const n = flats.length, dd = flats[0].length;
  const mean = new Float64Array(dd);
  for (const f of flats) for (let j = 0; j < dd; j++) mean[j] += f[j] / n;
  const Z = flats.map(f => Float64Array.from(f, (v, j) => v - mean[j]));
  const randn = makeRandn(1234);
  const comps = [];
  const deflated = Z.map(r0 => Float64Array.from(r0));
  for (let c = 0; c < comp; c++) {
    let v = unit(Float64Array.from({ length: dd }, () => randn()));
    let eig = 0;
    for (let it = 0; it < 40; it++) {
      const Zv = new Float64Array(deflated.length);
      for (let i = 0; i < deflated.length; i++) Zv[i] = dot(deflated[i], v);
      const nv = new Float64Array(dd);
      for (let i = 0; i < deflated.length; i++) { const zi = deflated[i], s = Zv[i]; for (let j = 0; j < dd; j++) nv[j] += zi[j] * s; }
      const nn = Math.max(1e-12, Math.hypot(...nv));
      v = Float64Array.from(nv, x => x / nn);
      eig = nn;
    }
    comps.push({ v, eig });
    // 标准 PCA deflation：从中心化矩阵中投影掉已提取成分，下一次幂迭代收敛到下一主成分
    for (let i = 0; i < deflated.length; i++) {
      const proj = dot(deflated[i], v);
      for (let j = 0; j < dd; j++) deflated[i][j] -= proj * v[j];
    }
  }
  return { mean, comps };
}

/** 编码 → 潜码（KB 级）；解码重建（含均值回加） */
export function encodeDecode(B, pca, K) {
  const f = flatOf(B);
  const c = Float64Array.from(f, (v, j) => v - pca.mean[j]);
  const code = new Float32Array(K);
  for (let k = 0; k < K; k++) code[k] = dot(c, pca.comps[k].v);
  const rec = Float64Array.from(f, (_, j) => pca.mean[j]);
  for (let k = 0; k < K; k++) { const v = pca.comps[k].v; for (let j = 0; j < rec.length; j++) rec[j] += code[k] * v[j]; }
  return { code, rec };
}

/** 保真度：重建 vs 原参数的余弦（展平视为向量） */
export function cosineFlat(a, b) { return cosine(a, b); }

/** 生成式检索：给定 prompt 潜码，从 L4 码本最近邻生成参数 */
export function generateFor(queryCode, genes) {
  let best = null, bd = Infinity;
  for (const g of genes) {
    let d2 = 0;
    for (let i = 0; i < g.code.length; i++) { const e = g.code[i] - queryCode[i]; d2 += e * e; }
    if (d2 < bd) { bd = d2; best = g; }
  }
  return { gene: best, dist: Math.sqrt(bd) };
}

export function experiment2({ n = 200, d = 12, Ks = [2, 4, 6, 8, 12], seed = 7 } = {}) {
  const skills = makeLoRAs(n, d, 3, seed);
  const pca = topComponents(skills.map(s => s.B), Math.max(...Ks) + 2);
  const rawKB = n * d * d * 4 / 1024;
  const sweep = [];
  for (const K of Ks) {
    const genes = skills.map(s => {
      const { code, rec } = encodeDecode(s.B, pca, K);
      return { id: s.id, code, rec, cos: cosineFlat(s.B, rec) };
    });
    // 生成式检索：以码为 query 最近邻命中（真实系统由 prompt 嵌入回归出码）
    let hit = 0;
    for (const g of genes) {
      const { gene } = generateFor(g.code, genes);
      if (gene.id === g.id) hit++;
    }
    const mean = genes.reduce((a, g) => a + g.cos, 0) / n;
    const min = Math.min(...genes.map(g => g.cos));
    const latentKB = n * K * 4 / 1024;
    const codebookKB = (K + 2) * d * d * 4 / 1024 + d * d * 4 / 1024; // 码本+均值=公共一次性资产
    sweep.push({
      K, meanCos: +mean.toFixed(4), minCos: +min.toFixed(4),
      latentKB: +latentKB.toFixed(2), totalKB: +(latentKB + codebookKB).toFixed(2),
      compression: +(rawKB / (latentKB + codebookKB)).toFixed(1),
      nnHitRate: +(hit / n).toFixed(2),
    });
  }
  const best = sweep.filter(x => x.meanCos >= 0.95).pop() ?? sweep.at(-1);
  return {
    sweep, rawKB: +rawKB.toFixed(2),
    best: { K: best.K, meanCos: best.meanCos, totalKB: best.totalKB, compression: best.compression, nnHitRate: best.nnHitRate },
    verdict: best.meanCos >= 0.95
      ? `K=${best.K} 潜码平均保真 ${best.meanCos} ≥0.95 → L4 基因化在“可容忍损失”的长尾技能上成立`
      : `全部 K 保真 <0.95 → 该尺度技能参数不适合基因化，保留原文件兜底`,
  };
}
