// ============================================================
// M1 硬件与带宽数学模型
// 数据来源：《低显存LLM插件化架构可行性评估报告》§2.1 带宽表
// 全部数值为报告标称/实测区间的中值或指定值，可被用户覆盖。
// ============================================================

export const HW = {
  vramMB: 6144,            // RTX 4050 Laptop 6 GB
  ramGB: 32,
  // 有效带宽（GB/s）——报告 §2.1
  bwVRAM: 90,              // 6GB 档 GDDR6 有效 ≈60–120，取中值
  bwDRAM: 80,              // DDR5 双通道 ≈70–90
  bwPCIe: 12,              // PCIe 4.0 x8 笔记本常见 ≈10–14（x16 可到 22–28）
  bwNVMe: 6,               // Gen4 ≈5–7
  pciePinEff: 0.88,        // 已 pin 内存块 H2D 效率 85–90%，取 88%
  // 骨干与 KV（报告 §6.1 / §6.3）
  backboneBytesGB: 2.6,    // 4B-Q4_K_M ≈ 2.2–2.8
  kvMBPerToken: 0.13,      // Q8 KV ≈ 0.1–0.15 MB/token
  prefillTokPerS: 800,     // 〔推算〕4B-GPU prefill 算力主导，500–1500 tok/s 量级（报告只给解码区间）
  decodeTokPerS: 22,       // 报告 §6.3：常驻 4B-Q4 真实 ≈20–30，取 22
  // 工作区与固定开销（报告 D.2）
  workspaceMB: 150,
  l1BudgetMB: 1000,        // L1 热沉积带预算（方案 §2.1）
};

// ---- 核心数学：推论 1 / 推论 2 / 推论 3 的实现 ----

/** 推论 1：逐 token 全量搬运的天花板（报告：4.4GB÷25GB/s≈5.7） */
export function tokenStreamCeiling(paramsGB, linkGbps = HW.bwPCIe) {
  return linkGbps / paramsGB;
}

/** 推论 3 / R 方案摊薄：S_plugin / T，单位 MB/token（报告 D.1 口径） */
export function amortizedPerToken(pluginMB, genTokens) {
  return pluginMB / genTokens;
}

/** 会话级一次性加载耗时（ms）：NVMe 读 + pinned H2D + 挂载 */
export function loadLatencyMs(pluginMB, fromNVMe = true) {
  const tNV = fromNVMe ? (pluginMB / 1024 / HW.bwNVMe) * 1000 : 0;
  const tH2D = (pluginMB / 1024 / (HW.bwPCIe * HW.pciePinEff)) * 1000;
  const tAttach = 20; // 挂载/合并常数（报告 D.1：几十~几百 ms 量级）
  return tNV + tH2D + tAttach;
}

/** 预取（仅钉选内存驻留）H2D 耗时 */
export function hotLoadLatencyMs(pluginMB) {
  return (pluginMB / 1024 / (HW.bwPCIe * HW.pciePinEff)) * 1000 + 15;
}

/** prefill 时长（ms），用于判断加载能否被隐藏 */
export function prefillMs(promptTokens) {
  return (promptTokens / HW.prefillTokPerS) * 1000;
}

/** 解码吞吐：每 token 必读常驻参数（VRAM 带宽）+ 外置读取（链路带宽），叠加算力约束（报告 §2.1 公式） */
export function decodeCeiling({ residentGB, remoteMBperToken = 0, computeCap }) {
  const tVram = residentGB / HW.bwVRAM;                      // GB ÷ GB/s = s/token
  const tLink = (remoteMBperToken / 1024) / (HW.bwPCIe * HW.pciePinEff);
  const tok = 1 / (tVram + tLink);
  return computeCap ? Math.min(tok, computeCap) : tok;
}

/** 冷路径稳流式加载的等效带宽（NVMe→DRAM→VRAM 串联，保守取调和） */
export function streamingBW(nvme = HW.bwNVMe, pcie = HW.bwPCIe) {
  return 1 / (1 / nvme + 1 / pcie);
}

export function fmtMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;
}
