#!/usr/bin/env node
// 沉积架构原型入口：node src/index.js [band|sediment|gene|msp|arbit|demo|all]
import { experiments, expSediment, expArbit, expBand } from "./m6-experiments.js";
import { runTurn, Arbiter } from "./m5-arbiter.js";
import { HW } from "./m1-hw.js";

const title = (s) => console.log(`\n═══ ${s} ${"═".repeat(Math.max(0, 46 - s.length))}`);

function band() {
  title("E-band · 报告 §2.1 三条推论独立复算（地基自检）");
  for (const r of expBand()) console.log(`${r.pass ? "✅" : "❌"} ${r.name.padEnd(34)} ${r.value}  ｜参照 ${r.ref}`);
}

function sediment() {
  title("E1 · 沉积闭环（方案 §7 实验1：使用→蒸馏→晋升→风化/基因化）");
  const r = expSediment({ rounds: 200 });
  console.log(JSON.stringify(r, null, 2));
}

function gene() {
  title("E2 · 生成式插件 / L4 基因化保真度（方案 §7 实验2，代理尺度）");
  console.log(JSON.stringify(experiments.gene({}), null, 2));
}

function msp() {
  title("E3 · MSP 跨骨干 ISA 转译（方案 §7 实验3 · 存亡假设）");
  const r = experiments.msp({});
  console.log(`维度 d=${r.d} | 槽 ${r.M} | 评测 ${r.evalN} 条 | 锚点噪声 ${r.anchorNoise} | 记忆包 ${r.packetKB} KB`);
  console.log(`(a) 联合上限      ${r.accA.toFixed(3)}`);
  console.log(`(b) ISA 转译      ${r.accB.toFixed(3)}   ｜ (b)/(a) = ${(r.ratio * 100).toFixed(1)}%`);
  console.log(`(b0) 朴素迁移     ${r.accB0.toFixed(3)}   ← 对照：不做转译应≈随机(0.1)`);
  console.log(`漂移曲线: ${r.drift.map(x => `k=${x.anchors}:${x.acc}`).join("  ")}`);
  console.log(`判定: ${r.verdict}`);
}

function arbit() {
  title("E4 · 插件尺寸×上下文相图 + 竞价仲裁对照（方案 §3 / D.8）");
  const r = expArbit();
  console.log(`静态通过: ${r.naivePass} ｜ 仲裁通过: ${r.arbPass}`);
  console.log(`仲裁器终态: 占用 ${r.arbiterFinalMB}/${r.arbiterPoolMB} MB, 驱逐 ${JSON.stringify(r.arbiterEvictions)}`);
  console.log("尾部驱逐日志:", r.arbiterLogTail);
  console.log(`判定: ${r.verdict}`);
}

function demo() {
  title("DEMO · 一次请求的端到端时序（方案 §4 表格的实算）");
  const fixed = HW.backboneBytesGB * 1024 + HW.workspaceMB + HW.l1BudgetMB;
  const arb = new Arbiter({ fixedMB: fixed, poolMB: Math.max(HW.vramMB - fixed, 0) });
  for (const [name, pluginMB, b0, g0, band0] of [
    ["空闲态仅骨干+L1", 0, 256, 512, "L1"],
    ["加载 300MB 冷插件", 300, 512, 2048, "L3-cold"],
    ["加载 300MB 热插件(内存驻留)", 300, 1024, 4096, "L3-hot"],
    ["2GB 大脑插件（预期违反峰值预算）", 2048, 1024, 4096, "L3-cold"],
  ]) {
    const t = runTurn({ promptTokens: b0, genTokens: g0, pluginMB, pluginBand: band0, l0l1MB: HW.backboneBytesGB * 1024 + HW.workspaceMB + HW.l1BudgetMB, arbiter: arb });
    console.log(`${name.padEnd(28)} TTFT ${String(t.ttft).padStart(5)} ms (加载${t.loadHidden ? " 已隐藏 " : ` 暴露${String(t.tLoad).padStart(4)}ms`}) 峰值 ${String(t.peakMB).padStart(6)} MB ${t.fits ? "✅" : "❌ OOM"} 解码≈${t.decodeTokS} tok/s`);
  }
}

const map = { band, sediment, gene, msp, arbit, demo };
const which = process.argv[2] ?? "all";
if (which === "all") { for (const f of Object.values(map)) f(); }
else if (map[which]) map[which]();
else { console.error(`未知实验: ${which}（可选 ${Object.keys(map).join("|")}）`); process.exit(1); }
