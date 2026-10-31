// ============================================================
// M2 五层驻留带（方案 §2.1）
// L0 岩基带(常驻显存) / L1 热沉积带(常驻显存, 自组织)
// L2 会话带(显存↔钉选内存) / L3 频带库(钉选内存) / L4 潜码带(NVMe, KB 级)
//
// 三条上行通道（方案 §2.2）：
//   通道2 热蒸馏 L3→L1：byteFreqScore ≥ 阈值 且 L1 预算容得下（末位淘汰）
//   通道3 基因化 L3→L4：从未产生使用记录 → 压成 KB 级潜码
//   风化   L1→L3：长期低分 → 降回频带库（原插件仍在，可回滚）
// ============================================================
import { HW } from "./m1-hw.js";

export const BAND = { L0: "L0", L1: "L1", L2: "L2", L3: "L3", L4: "L4" };

export class BandSystem {
  constructor(opts = {}) {
    this.plugins = new Map();
    this.l1 = [];                                  // {id, sizeMB, score, distilledCap}
    this.l1BudgetMB = opts.l1BudgetMB ?? HW.l1BudgetMB;
    this.promoThreshold = opts.promoThreshold ?? 0.03;
    this.weatherFloor = opts.weatherFloor ?? 0.008;
    this.telemetry = { promotions: 0, demotions: 0, geneifications: 0 };
  }

  register(p) {
    const item = { ...p, band: BAND.L3, scoreEMA: 0, sessionsUsed: 0, lastUsedRound: -1 };
    this.plugins.set(p.id, item);
    return item;
  }

  /** 使用记录：EMA 频率 × 强度 / 体积（方案 §2.2 通道2 的 byteFreqScore） */
  recordUse(id, intensity, sizeMB, round, ema = 0.25) {
    const p = this.plugins.get(id);
    if (!p) return;
    p.sessionsUsed++;
    p.lastUsedRound = round;
    const inst = intensity / Math.max(sizeMB / 100, 0.1);
    p.scoreEMA = p.scoreEMA === 0 ? inst : (1 - ema) * p.scoreEMA + ema * inst;
  }

  /** 热蒸馏晋升：返回是否成功（L1 预算不足按末位淘汰腾位） */
  promote(id, distilledCap, keepFidelity) {
    const p = this.plugins.get(id);
    if (!p || p.band !== BAND.L3 || p.scoreEMA < this.promoThreshold) return false;
    if (keepFidelity < 0.95) return false;            // 晋升质量门（方案 N1 风险处置）
    const size = p.loraSizeMB;
    while (this.l1UsedMB() + size > this.l1BudgetMB && this.l1.length > 0) {
      this.l1.sort((a, b) => b.score - a.score);
      const victim = this.l1.pop();
      const vp = this.plugins.get(victim.id);
      vp.band = BAND.L3;
      vp.sizeMB = vp.origSizeMB ?? vp.sizeMB;
      this.telemetry.demotions++;
    }
    if (this.l1UsedMB() + size > this.l1BudgetMB) return false; // 比整个预算还大 → 不晋升
    this.l1.push({ id, sizeMB: size, score: p.scoreEMA, distilledCap });
    p.band = BAND.L1;
    p.origSizeMB = p.sizeMB;
    p.sizeMB = size;                 // 驻留足迹 = 迷你 LoRA 蒸馏产物体积
    this.telemetry.promotions++;
    return true;
  }

  /** 风化：久未使用且分数跌破底线 → 降回 L3 */
  weather(round) {
    this.l1 = this.l1.filter(e => {
      const p = this.plugins.get(e.id);
      const stale = round - p.lastUsedRound > 60 && p.scoreEMA < this.weatherFloor;
      if (stale) { p.band = BAND.L3; p.sizeMB = p.origSizeMB ?? p.sizeMB; this.telemetry.demotions++; }
      return !stale;
    });
  }

  /** 基因化：零使用的沉睡插件 → L4 潜码（磁盘足迹 KB 级） */
  geneify(id) {
    const p = this.plugins.get(id);
    if (!p || p.band !== BAND.L3 || p.sessionsUsed > 0) return false;
    p.band = BAND.L4; p.sizeMB = 0.004;
    this.telemetry.geneifications++;
    return true;
  }

  l1UsedMB() { return this.l1.reduce((a, e) => a + e.sizeMB, 0); }
  counts() { const c = { L1: 0, L2: 0, L3: 0, L4: 0 }; for (const p of this.plugins.values()) c[p.band]++; return c; }
  bandSnapshot() {
    const mbs = { L3: 0, L4: 0 };
    for (const p of this.plugins.values()) if (mbs[p.band] !== undefined) mbs[p.band] += p.sizeMB;
    return { L1MB: +this.l1UsedMB().toFixed(0), L1BudgetMB: this.l1BudgetMB, L3MB: +mbs.L3.toFixed(0), L4KB: +(mbs.L4 * 1024).toFixed(1), counts: this.counts() };
  }
}
