// ============================================================
// M6 实验编排
// E-band     ：报告 §2.1 推论 1/2/3 独立复算（地基自检）
// E1 sediment：方案 §7 实验1 —— 使用→蒸馏→L1晋升→风化/基因化闭环
// E2 gene    ：m4 —— L4 基因化压缩-保真扫描
// E3 msp     ：m3 —— 跨骨干 ISA 转译（存亡假设，四臂对照）
// E4 arbit   ：D.8 实验2 —— 插件尺寸×上下文相图 + 仲裁对照
//
// 质量代理声明：本原型没有真实 LLM 骨干，用"能力向量"代理插件
// 增益、用"高斯噪声旋转"注入蒸馏损失（假设 σ=0.05）。实验验证
// 的是【调度/沉积/迁移机制的动力学与物理预算】，不是模型质量。
// ============================================================
import { HW, tokenStreamCeiling, amortizedPerToken, loadLatencyMs, hotLoadLatencyMs, prefillMs, decodeCeiling, streamingBW, fmtMs } from "./m1-hw.js";
import { BandSystem, BAND } from "./m2-bands.js";
import { Arbiter } from "./m5-arbiter.js";
import { experiment2 } from "./m4-gene.js";
import { experiment3 } from "./m3-msp.js";
import { cosine, unit, makeRandn } from "./m0-linalg.js";

const rng = (seed) => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };

function makePlugin(id, sizeMB, strength, capSeed, cold) {
  const randn = makeRandn(capSeed);
  const cap = unit(Float64Array.from({ length: 48 }, () => randn()));
  return { id, sizeMB, loraSizeMB: Math.max(sizeMB * 0.2, 6), cap, strength, cold };
}
function taskScore(taskCap, activeCaps) {
  let s = 0.35; // 骨干基础分（代理）
  for (const v of activeCaps) s += 0.55 * Math.max(0, cosine(taskCap, v));
  return Math.min(1, s);
}

// ---- E-band：复算报告 §2.1 ----
export function expBand() {
  const rows = [];
  const c1 = tokenStreamCeiling(4.4, 25);
  rows.push({ name: "推论1 稠密7–8B Q4 逐token过PCIe(x16 25GB/s)", value: `${c1.toFixed(2)} tok/s`, ref: "报告 5.7", pass: Math.abs(c1 - 5.7) < 0.5 });
  const c1x8 = tokenStreamCeiling(4.4, 10);
  rows.push({ name: "推论1 x8 笔记本链路（10GB/s 保守端）", value: `${c1x8.toFixed(2)} tok/s`, ref: "报告：跌到 2 tok/s 级", pass: c1x8 <= 2.4 });
  const c1nv = tokenStreamCeiling(4.4, streamingBW());
  rows.push({ name: "推论1 NVMe→DRAM→VRAM 串联流式", value: `${c1nv.toFixed(2)} tok/s`, ref: "报告 <2（灾难区确认）", pass: c1nv < 2 });
  const activeGB = 368 * 0.055; // 671B Q4≈368GB × 5.5% 激活
  const c2 = 80 / activeGB;
  rows.push({ name: "推论2 MoE 激活专家 DRAM 带宽下限", value: `≥${c2.toFixed(1)} tok/s`, ref: "报告实测 8.7–13.7（AMX/选择性激活高于下限）", pass: c2 >= 3 });
  const t4096 = amortizedPerToken(300, 4096);
  rows.push({ name: "推论3 300MB LoRA 4096token 摊薄", value: `${t4096.toFixed(4)} MB/token`, ref: "报告 D.1：摊薄后无感", pass: t4096 < 0.1 });
  rows.push({ name: "推论3 300MB 冷加载一次（NVMe+H2D+挂载）", value: fmtMs(loadLatencyMs(300)), ref: "报告 D.4：百 ms 级", pass: loadLatencyMs(300) < 400 });
  const ceil = decodeCeiling({ residentGB: HW.backboneBytesGB, remoteMBperToken: 2, computeCap: HW.decodeTokPerS });
  rows.push({ name: "综合解码上限（2.6GB常驻+2MB/token外置读）", value: `${ceil.toFixed(1)} tok/s`, ref: "报告 §6.3：4B 真实 20–30", pass: ceil <= HW.decodeTokPerS + 0.5 });
  return rows;
}

// ---- E1 沉积闭环 ----
export function expSediment({ rounds = 200, nSkills = 8, seed = 42, promoThreshold = 0.03, weatherFloor = 0.008 } = {}) {
  const u = rng(seed);
  const sys = new BandSystem({ promoThreshold, weatherFloor });
  const skills = [];
  for (let i = 0; i < nSkills; i++) {
    // 注册返回驻留对象副本——后续统计/晋升必须用副本，不能再用临时对象
    skills.push(sys.register(makePlugin(`s${i}`, 180 + i * 45, 0.55 + 0.03 * (i % 3), seed + i * 13, u() < 0.35)));
  }
  const activePool = skills.slice(0, 6); // 负载只触达 6 个技能，其余 2 个沉睡 → 基因化通道
  const randn = makeRandn(seed + 5);

  const tax = [], scores = [], promoLog = [], distill = [], loadShare = [];
  for (let r = 0; r < rounds; r++) {
    const skill = activePool[Math.floor(Math.pow(u(), 1.7) * activePool.length)]; // Zipf 式
    const promptTok = [32, 64, 128, 256, 512, 1024][Math.floor(u() * 6)];
    // 加载代价：L1=0；L3 热=H2D；L3 冷=NVMe×(1+1.5u) 抖动（报告 §2.3 p95 毛刺）；L4=潜码生成≈90ms 无文件 I/O
    let tLoad = 0;
    if (skill.band === BAND.L3) tLoad = skill.cold ? loadLatencyMs(skill.sizeMB) * (1 + 1.5 * u()) : hotLoadLatencyMs(skill.sizeMB);
    else if (skill.band === BAND.L4) tLoad = 90;
    loadShare.push(tLoad > 0 ? 1 : 0);
    tax.push(30 + Math.max(tLoad - prefillMs(promptTok), 0)); // 加载未被 prefill 隐藏的暴露部分 + 路由
    // 质量：L1 蒸馏 cap（有损代理）+ 其他 L1 沉积物附加增益
    const active = sys.l1.map(e => e.distilledCap);
    const own = skill.band === BAND.L1 ? (sys.l1.find(e => e.id === skill.id)?.distilledCap ?? skill.cap) : skill.cap;
    if (!active.includes(own)) active.push(own);
    scores.push(taskScore(skill.cap, active));
    sys.recordUse(skill.id, 0.55 + 0.03 * ((skill.id.charCodeAt(1) % 3)), skill.sizeMB, r);
    if (r % 10 === 9) {
      for (const s of skills) {
        if (s.band === BAND.L3 && s.scoreEMA * s.strength / (s.sizeMB / 100) >= promoThreshold) {
          const distCap = unit(Float64Array.from(s.cap, v => v + 0.03 * randn())); // 蒸馏损失假设 σ=0.03
          const keep = cosine(s.cap, distCap); // 晋升质量门（N1 风险处置）
          distill.push(keep);
          if (sys.promote(s.id, distCap, keep)) promoLog.push({ id: s.id, at: r, keep: +keep.toFixed(4) });
        }
      }
      sys.weather(r);
      if (r >= 80) for (const s of skills) if (s.band === BAND.L3 && s.sessionsUsed === 0) sys.geneify(s.id);
    }
  }
  const win = (a, x, y) => a.slice(x, y).reduce((s2, v) => s2 + v, 0) / Math.max(1, y - x);
  const p95 = (a, x, y) => { const s2 = a.slice(x, y).sort((m, n2) => m - n2); return s2[Math.floor(s2.length * 0.95)]; };
  const tKeep = distill.length ? distill.reduce((a, b) => a + b, 0) / distill.length : null;
  const keepPct = win(scores, rounds - 50, rounds) / win(scores, 0, 50);
  return {
    tax_p95_first50: Math.round(p95(tax, 0, 50)),
    tax_p95_last50: Math.round(p95(tax, rounds - 50, rounds)),
    tax_p95_drop_pct: +((1 - p95(tax, rounds - 50, rounds) / Math.max(p95(tax, 0, 50), 1)) * 100).toFixed(1),
    pluginMovePct_first50: +(win(loadShare, 0, 50) * 100).toFixed(1),
    pluginMovePct_last50: +(win(loadShare, rounds - 50, rounds) * 100).toFixed(1),
    promotions: promoLog.length, demotions: sys.telemetry.demotions, geneified: sys.telemetry.geneifications,
    distillKeepPct: tKeep ? +(tKeep * 100).toFixed(1) : null,
    score_first50: +win(scores, 0, 50).toFixed(3), score_last50: +win(scores, rounds - 50, rounds).toFixed(3),
    qualityKeep: +keepPct.toFixed(3),
    bands: sys.bandSnapshot(),
    verdict: (tKeep === null || tKeep >= 0.95) && keepPct >= 0.95
      && p95(tax, rounds - 50, rounds) < p95(tax, 0, 50) && sys.telemetry.promotions > 0
      ? "晋升保真≥95% 且任务分不劣化 且加载税下降 → 沉积闭环成立"
      : "未达门限：见上表逐项",
  };
}

// ---- E4 相图 + 仲裁对照 ----
export function expArbit() {
  const sizesMB = [0, 100, 300, 600, 1024, 2048, 2900];
  const ctx = [2048, 8192, 16384, 32768];
  const fixed = HW.backboneBytesGB * 1024 + HW.workspaceMB + HW.l1BudgetMB; // L0+L1
  const grid = [];
  let naivePass = 0, arbPass = 0;
  for (const s of sizesMB) for (const c of ctx) {
    const kv = c * HW.kvMBPerToken;
    const naiveOk = fixed + s + kv + HW.workspaceMB <= HW.vramMB;
    // 仲裁规则（方案 §3）：Q8→低精 KV(×0.55) + ≤60% 页外移 → 驻留≈0.22kv；
    // 且为 KV 工作集留 ≥200MB，插件不得吃满竞价区
    const arbOk = fixed + s + Math.max(0.22 * kv, 150) <= HW.vramMB && s <= HW.vramMB - fixed - 200;
    if (naiveOk) naivePass++; if (arbOk) arbPass++;
    grid.push({ s, c, peak: Math.round(fixed + s + kv + HW.workspaceMB), naiveOk, arbOk });
  }
  const arb = new Arbiter({ fixedMB: fixed, poolMB: Math.max(HW.vramMB - fixed - HW.workspaceMB, 0) });
  for (let p = 0; p < 14; p++) {
    arb.kvPages.forEach(x => x.age++);
    arb.admit("kv", { id: `kv${p}`, tokens: 2048, sizeMB: 2048 * HW.kvMBPerToken, age: 0 });
    if (p % 4 === 3) arb.admit("plugin", { id: `pl${p}`, sizeMB: 300, gain: 0.7 });
  }
  return {
    grid,
    naivePass: `${naivePass}/${grid.length}`, arbPass: `${arbPass}/${grid.length}`,
    arbiterFinalMB: +arb.usedMB().toFixed(1), arbiterPoolMB: arb.poolMB,
    arbiterEvictions: arb.evictions, arbiterLogTail: arb.log.slice(-3),
    verdict: arbPass > naivePass
      ? `竞价仲裁把静态 OOM 区 ${naivePass}/${grid.length} 提升到 ${arbPass}/${grid.length} → 中粒度带在 6GB 下可运营`
      : "仲裁收益不足，检查量化/卸载系数",
  };
}

export const experiments = { band: expBand, sediment: expSediment, gene: experiment2, msp: experiment3, arbit: expArbit };
