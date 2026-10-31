# -*- coding: utf-8 -*-
"""训练→验收 自动串联：等 l0_base 导出完成后自动跑四项职责验收。

设计理由：预训练约 12 小时，不应让人守着。本脚本轮询导出产物，
出现后依次执行 accept_l0.py 的四项验收，全程日志落盘 chain.log。
"""
import os, sys, time, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FINAL = os.path.join(ROOT, "l0_base", "model.safetensors")
CFG = os.path.join(ROOT, "l0_base", "config.json")
LOG = os.path.join(ROOT, "l0_acceptance", "chain.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
PY = r"D:\Doubao-works\训练前置\train_env\Scripts\python.exe"


def log(*a):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("串联器启动，等待训练导出 …")
log(f"监听: {FINAL}")
t0 = time.time()
POLL = 120
last_beat = 0
while True:
    if os.path.exists(FINAL) and os.path.exists(CFG):
        sz = os.path.getsize(FINAL)
        # 连续两次大小不变才认为写完（防止读到半截文件）
        time.sleep(20)
        if os.path.getsize(FINAL) == sz and sz > 1e9:
            log(f"导出完成 {sz/1e9:.2f} GB，等待 {time.time()-t0:.0f}s")
            break
    el = time.time() - t0
    if el - last_beat > 1800:
        log(f"仍在等待 … 已 {el/3600:.1f} h")
        last_beat = el
    if el > 20 * 3600:
        log("超过 20 小时仍未导出，放弃（训练可能异常，请检查 proc 日志）")
        sys.exit(1)
    time.sleep(POLL)

log("开始四项职责验收 …")
env = dict(os.environ)
env["D2_STEPS"] = "300"
r = subprocess.run([PY, os.path.join(HERE, "accept_l0.py")], cwd=HERE, env=env)
log(f"验收进程退出码 {r.returncode}")

p = os.path.join(ROOT, "l0_acceptance", "l0_acceptance.json")
if os.path.exists(p):
    d = json.load(open(p, encoding="utf-8"))
    log(f"通过职责: {d['summary']['duties_passed']} ({d['summary']['n_passed']}/4)")
    log(f"判定: {d['summary']['verdict']}")
else:
    log("未生成验收 JSON")
log("串联器结束")
