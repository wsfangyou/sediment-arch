// ============================================================
// M5 统一驻留仲裁（方案 §3）+ 端到端一次请求时序（方案 §4）
// 竞价区租户 = KV 页（报价按轮龄指数衰减×长度）+ 激活插件
//（报价 = 剩余会话预期增益/字节）。周期动作限幅：每周期驱逐 1 名。
// ============================================================
import { HW, loadLatencyMs, hotLoadLatencyMs, prefillMs, decodeCeiling } from "./m1-hw.js";

export class Arbiter {
  constructor({ fixedMB, poolMB }) {
    this.fixedMB = fixedMB;
    this.poolMB = poolMB;
    this.kvPages = [];
    this.plugins = [];
    this.log = [];
    this.evictions = { kv: 0, plugin: 0 };
  }
  usedMB() {
    return this.kvPages.reduce((a, p) => a + p.sizeMB, 0) + this.plugins.reduce((a, p) => a + p.sizeMB, 0);
  }
  bidKV(p) { return Math.exp(-p.age / 8) * p.tokens * 0.02; }   // 边际价值代理（轮龄衰减）
  bidPlugin(p) { return (p.gain || 0.5) / (p.sizeMB / 100); }
  admit(kind, item) {
    if (kind === "kv") this.kvPages.push(item); else if (item.sizeMB > 0) this.plugins.push(item);
    let guard = 0;
    while (this.usedMB() > this.poolMB && guard++ < 60) {
      const cands = [
        ...this.kvPages.map(p => ({ ref: p, kind: "kv", bid: this.bidKV(p) })),
        ...this.plugins.map(p => ({ ref: p, kind: "plugin", bid: this.bidPlugin(p) })),
      ].sort((a, b) => a.bid - b.bid);
      if (!cands.length) break;
      const v = cands[0];
      if (v.kind === "kv") { this.kvPages = this.kvPages.filter(x => x !== v.ref); v.ref.offloaded = true; this.evictions.kv++; }
      else { this.plugins = this.plugins.filter(x => x !== v.ref); this.evictions.plugin++; }
      this.log.push({ step: this.log.length + 1, evicted: v.kind, id: v.ref.id, bid: +v.bid.toFixed(3), usedAfterMB: +this.usedMB().toFixed(1) });
    }
    return this.usedMB() <= this.poolMB;
  }
}

/**
 * 一次请求的时序分解（方案 §4）：路由 → 加载(并行 prefill) → 生成。
 * 返回加载是否被隐藏、峰值显存、是否越界、解码上限（M1 模型）。
 */
export function runTurn({ promptTokens, genTokens, pluginMB = 0, pluginBand = "L3-cold", l0l1MB, arbiter }) {
  const tRoute = 30;
  let tLoad = 0;
  if (pluginMB > 0 && pluginBand !== "L1") {
    tLoad = pluginBand === "L3-hot" ? hotLoadLatencyMs(pluginMB) : loadLatencyMs(pluginMB);
  }
  const tPrefill = prefillMs(promptTokens);
  const ttft = tRoute + Math.max(0, tLoad - tPrefill) + tPrefill;
  const kvMB = (promptTokens + genTokens) * HW.kvMBPerToken;
  const peakMB = l0l1MB + pluginMB + kvMB; // workspace 已含在 l0l1MB 口径外，单独加
  const peakTotal = peakMB + HW.workspaceMB;
  const fits = peakTotal <= HW.vramMB;
  if (arbiter) {
    arbiter.kvPages.forEach(p => p.age++);
    if (pluginMB > 0) arbiter.admit("plugin", { id: "plug", sizeMB: pluginMB, gain: 0.6 });
    arbiter.admit("kv", { id: `kv-${arbiter.log.length}`, tokens: promptTokens + genTokens, sizeMB: kvMB, age: 0 });
  }
  const decode = decodeCeiling({ residentGB: (l0l1MB + pluginMB) / 1024, remoteMBperToken: 0, computeCap: HW.decodeTokPerS });
  return {
    tRoute, tLoad: Math.round(tLoad), tPrefill: Math.round(tPrefill),
    ttft: Math.round(ttft), loadHidden: tLoad <= tPrefill,
    kvMB: +kvMB.toFixed(1), peakMB: +peakTotal.toFixed(1), fits,
    decodeTokS: +decode.toFixed(1),
  };
}
