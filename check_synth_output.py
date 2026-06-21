# -*- coding: utf-8 -*-
"""比對 synth pipeline 輸出 (imagesoutput/*.json) 與 synth/answer.txt 標準答案。"""
import os, sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.join(os.path.dirname(__file__), "synth")
OUT = os.path.join(BASE, "imagesoutput")
ANS = os.path.join(BASE, "answer.txt")

# 解析答案
exp = {}
with open(ANS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, rest = line.split(" ", 1)
        exp[name] = {"v": "有垂直" in rest, "h": "有平行" in rest, "c": "(交叉)" in rest}

n = 0
cls_counts = {}
routed = 0                     # 分類為埋深 -> 有跑尺規
acc = {"v": [0, 0], "h": [0, 0], "c": [0, 0]}
cross_cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
dup = susp = 0
missing_ruler = 0

for name, e in exp.items():
    jp = os.path.join(OUT, name + ".json")
    if not os.path.exists(jp):
        continue
    n += 1
    with open(jp, encoding="utf-8") as f:
        r = json.load(f)

    cls_counts[r.get("class")] = cls_counts.get(r.get("class"), 0) + 1
    if r.get("is_duplicate"):
        dup += 1
    if "suspected_duplicate" in r:
        susp += 1

    rr = r.get("ruler_result")
    if not rr or "detections" not in rr:
        missing_ruler += 1
        continue
    routed += 1
    d = rr["detections"]
    gv = d.get("vertical_count", 0) > 0
    gh = d.get("horizontal_count", 0) > 0
    gc = d.get("is_crossed", False)
    for k, ok in [("v", gv == e["v"]), ("h", gh == e["h"]), ("c", gc == e["c"])]:
        acc[k][0] += ok
        acc[k][1] += 1
    if e["c"] and gc: cross_cm["TP"] += 1
    elif e["c"]: cross_cm["FN"] += 1
    elif gc: cross_cm["FP"] += 1
    else: cross_cm["TN"] += 1

print(f"總比對: {n} 張")
print(f"\n[分類結果分布]")
for k, v in sorted(cls_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({100*v/n:.1f}%)")
print(f"  -> 成功路由到尺規流程: {routed}/{n} ({100*routed/n:.1f}%)")
if missing_ruler:
    print(f"  -> 未跑尺規(分類非埋深): {missing_ruler}")

print(f"\n[尺規偵測準確率 (僅計有跑尺規的 {routed} 張)]")
for k, label in [("v", "垂直"), ("h", "平行"), ("c", "交叉")]:
    c, t = acc[k]
    if t:
        print(f"  {label}: {c}/{t} = {100*c/t:.1f}%")
print(f"  交叉混淆: {cross_cm}")
if cross_cm['FP'] + cross_cm['TN']:
    fp_rate = 100*cross_cm['FP']/(cross_cm['FP']+cross_cm['TN'])
    print(f"  交叉誤報率 (沒碰到卻判交叉): {fp_rate:.1f}%")

print(f"\n[去重]")
print(f"  SHA-256 完全重複: {dup}")
print(f"  ResNet 疑似重複: {susp}  (合成圖共用少量素材，預期偏高)")
