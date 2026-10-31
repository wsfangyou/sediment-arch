// ============================================================
// M3 MSP（Memory State Packet）——方案 §2.3 的真实实现
//
// 语义（实验 3 要检验的可证伪主张本身）：
//   · 槽内容 = 规范帧（ISA 定义的坐标空间）里的 key/value，Q8 量化，
//     与骨干无关 —— "记忆状态而非权重切片"的实质。
//   · 骨干↔规范帧唯一要协商的对象 = 标定 head（岭回归线性探针）。
//   · 第三方发布：面向公开冻结的参考骨干训练，产出规范帧槽包。
//   · 使用端挂载：仅用自己骨干的锚点重估 head，槽一个字节不动。
//
// 四臂对照（方案 §7 实验 3，存亡假设）：
//   (a)  联合上限：知识方直接面向目标骨干 B 训练（行业现状的隐含前提），
//        槽 key 存 B 空间坐标，查询零标定误差 —— 质量上限参照。
//   (b)  ISA 转译：槽存规范帧坐标，查询经锚点探针 head_B 进入规范帧。
//        与 (a) 的全部差距 = 锚点标定的估计误差 —— 本实验的核心量。
//   (b0) 朴素迁移：有槽包但完全不做标定（查询原始骨干坐标直查规范槽）
//        —— 对照组，预期接近随机水平。
//   drift(k)：公共锚点数量 k < d 时 (b) 臂的退化曲线。
// 判定：acc(b) ≥ 95%×acc(a) → "ABI 可最小化"在本机制上升格。
// 诚实边界：玩具尺度下骨干间关系被建模为"线性映射+标注噪声"；
// 真实 Transformer 骨干间关系非线性，本实验检验机制形状而非外推结论。
// ============================================================
import {
  zeros, matmul, ridge, cosine, unit, quantInt8, dequantInt8,
  makeRandn, mulberry32,
} from "./m0-linalg.js";

export const MSP_SPEC = { d: 32, M: 200, lambda: 1e-2, quantBits: 8 };

/** 骨干视角：canonicalVecs 经映射 G 进入该骨干表示空间 */
export const substrateView = (vecs, G) => vecs.map(v => Float64Array.from(matmul([v], G)[0]));

function randMat(d, randn, diag = 1.5, off = 0.4) {
  const A = zeros(d, d);
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) A[i][j] = randn() * (i === j ? diag : off);
  return A;
}

/**
 * 知识域生成（簇结构）：同簇事实共享 base，成员 = base + 扰动。
 * 每簇前 4 条进槽（train），后 4 条做评测（eval）——检索信号
 * 来自簇内相似性；跨簇近正交 → 错标定 = 错簇 = 质量崩塌，
 * 各臂差异只来自【坐标空间是否对齐】这一个变量。
 */
export function makeClusteredFacts(randn, d, nClusters, per, P) {
  const out = [];
  for (let c = 0; c < nClusters; c++) {
    const base = unit(Float64Array.from({ length: d }, () => randn()));
    for (let j = 0; j < per; j++) {
      const k = unit(Float64Array.from(base, (v, i) => v + 0.32 * randn()));
      const v = unit(Float64Array.from(matmul([k], P)[0]));
      out.push({ k, v, cluster: c, slot: j });
    }
  }
  return out;
}
export function probe(anchorSpace, anchorCanon, lambda = MSP_SPEC.lambda) {
  return ridge(anchorSpace.map(v => Float64Array.from(v)), anchorCanon.map(v => Float64Array.from(v)), lambda);
}

export class MSP {
  constructor(spec = MSP_SPEC) {
    this.spec = spec;
    this.keys = [];   // 存储空间的 key（Q8）
    this.values = []; // 规范帧的 value（Q8）
  }
  store(pairs) {
    for (const { k, v } of pairs.slice(0, this.spec.M)) {
      this.keys.push(quantInt8(unit(k)));
      this.values.push(quantInt8(unit(v)));
    }
    return this;
  }
  retrieve(queryVec, head, topk = 1) {
    const q = head ? Float64Array.from(matmul([queryVec], head)[0]) : queryVec;
    const qn = unit(q);
    return this.keys
      .map((kq, i) => ({ i, s: cosine(qn, dequantInt8(kq)), v: dequantInt8(this.values[i]) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, topk);
  }
  /** 评测对列表：{q, truth}；head=null 表示查询与槽同空间 */
  evalPairs(pairs, head) {
    let acc = 0;
    for (const { q, truth } of pairs) {
      const got = this.retrieve(q, head, 1)[0].v;
      acc += Math.max(0, cosine(got, unit(truth)));
    }
    return acc / Math.max(pairs.length, 1);
  }
  packetBytes() {
    return this.keys.length * this.spec.d * (this.spec.quantBits / 8) * 2
      + 256 /* 清单: 基座指纹/域标签/冲突声明/量化规格 */;
  }
}

export function experiment3({ d = MSP_SPEC.d, M = MSP_SPEC.M, anchorNoise = 0.05, seed = 42 } = {}) {
  const randn = makeRandn(seed);
  const GA = randMat(d, randn), GB = randMat(d, randn), P = randMat(d, randn, 1.8);
  const nClusters = 50, per = 8; // 50×8=400 条：每簇前 4 进槽（=M200），后 4 评测
  const canon = makeClusteredFacts(randn, d, nClusters, per, P);
  const qB_all = substrateView(canon.map(f => f.k), GB); // 骨干 B 坐标
  const trainIdxs = canon.map((f, i) => f.slot < 4 ? i : -1).filter(i => i >= 0);
  const evalIdxList = canon.map((f, i) => f.slot >= 4 ? i : -1).filter(i => i >= 0);
  const trainPairs = trainIdxs.map(i => canon[i]);
  const evalQ = evalIdxList.map(i => qB_all[i]);
  const evalTruth = evalIdxList.map(i => canon[i].v);
  const pairsFor = () => evalQ.map((q, t) => ({ q, truth: evalTruth[t] }));

  // ---- 臂 (a) 联合上限：知识方直接面向 B 训练（槽 key 存 B 空间坐标），查询零标定 ----
  const mspA = new MSP(MSP_SPEC);
  mspA.store(trainIdxs.map(i => ({ k: qB_all[i], v: canon[i].v })));
  const accA = mspA.evalPairs(pairsFor(), null);

  // ---- 臂 (b) ISA 转译：第三方面向参考骨干发布规范帧槽包；B 用带噪锚点重估 head ----
  const mspB = new MSP(MSP_SPEC);
  mspB.store(trainPairs.map(f => ({ k: f.k, v: f.v }))); // 槽 = 规范帧坐标，与骨干无关
  const nAnchor = d;
  const canonAnchors = Array.from({ length: nAnchor }, () => unit(Float64Array.from({ length: d }, () => randn())));
  const anchorsB_noisy = substrateView(canonAnchors, GB).map(v => Float64Array.from(v, x => x + anchorNoise * randn()));
  const headB = probe(anchorsB_noisy, canonAnchors);
  const accB = mspB.evalPairs(pairsFor(), headB);

  // ---- 臂 (b0) 朴素迁移：完全不做标定（B 坐标直查规范槽） ----
  const accB0 = mspB.evalPairs(pairsFor(), null);

  // ---- 漂移曲线：锚点数 k < d → 标定欠定 ----
  const drift = [];
  for (const k of [4, 8, 12, 16, 24, nAnchor]) {
    const head = probe(anchorsB_noisy.slice(0, k), canonAnchors.slice(0, k));
    drift.push({ anchors: k, dim: d, acc: mspB.evalPairs(pairsFor(), head) });
  }

  const ratio = accB / accA;
  return {
    d, M: trainPairs.length, evalN: evalQ.length, anchorNoise,
    accA, accB, accB0, ratio,
    drift: drift.map(x => ({ ...x, acc: +x.acc.toFixed(3) })),
    packetKB: +(mspB.packetBytes() / 1024).toFixed(1),
    verdict: ratio >= 0.95 ? "升格：ISA 转译 ≥ 95%×联合上限（玩具尺度）"
      : ratio > accB0 + 0.15 ? "部分成立：转译显著优于不转译，但未达 95% 线"
      : "收缩：标定不足以桥接，退化为同骨干格式",
  };
}
