# -*- coding: utf-8 -*-
"""
test_models.py — 不需要真實照片，驗證 5 個 YOLO 模型：
  1. 能否載入
  2. 模型自報的類別名稱，是否與 vision_pipeline 程式碼裡寫死的對照表吻合
  3. 用合成圖跑一次推論，確認流程不會當掉

用法:  python test_models.py
"""
import sys
import numpy as np
from ultralytics import YOLO

import vision_pipelinev3resnet as vp

OK = "[OK]  "
BAD = "[BAD] "
WARN = "[??]  "

# 每個模型「程式碼期望」的類別集合
EXPECTED = {
    "classify":       set(vp.CLASS_TO_TYPE.keys()),          # 埋深/水準點/座標
    "ruler":          {"vertical", "horizontal"},            # run_ruler 第407-409行（小寫比對）
    "coordfmt":       set(vp.COORD_YOLO_CLASS_MAP.keys()),   # local_NEH/simple_NEZ/...
    "ruler_classify": set(vp.RULER_CLASSIFY_VALID),          # 管線/人手孔/一般
    # bench 是偵測模型（裁切數字框），類別名稱不影響流程，只檢查能否載入＋推論
    "bench":          None,
}

problems = []

print("=" * 60)
print(f"DEVICE = {vp.DEVICE}")
print("=" * 60)

for key, expected in EXPECTED.items():
    path = vp.paths[key]
    print(f"\n--- {key}  ({path}) ---")
    try:
        model = YOLO(path)
    except Exception as e:
        print(f"{BAD}模型載入失敗: {e}")
        problems.append(f"{key}: 載入失敗 {e}")
        continue

    names = set(model.names.values())
    task = getattr(model, "task", "?")
    print(f"      task={task}  類別={sorted(model.names.values())}")

    if expected is not None:
        # ruler 是用小寫比對，統一轉小寫處理
        if key == "ruler":
            got = {n.lower() for n in names}
            exp = {n.lower() for n in expected}
        else:
            got, exp = names, expected

        missing = exp - got          # 程式期望、但模型沒有的（會導致永遠匹配不到）
        extra = got - exp            # 模型有、但程式沒處理的（可能漏判）
        if missing:
            print(f"{BAD}程式期望卻找不到的類別: {sorted(missing)}")
            problems.append(f"{key}: 缺少類別 {sorted(missing)}")
        elif extra:
            print(f"{WARN}模型多出程式未處理的類別: {sorted(extra)}（確認是否需要處理）")
        else:
            print(f"{OK}類別名稱與程式碼完全吻合")

    # 用合成圖跑一次推論，確認流程不當掉
    try:
        dummy = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
        r = model.predict(source=dummy, device=vp.DEVICE, verbose=False)[0]
        if task == "classify":
            top = r.names[r.probs.top1]
            print(f"{OK}合成圖推論成功 -> top1={top} ({float(r.probs.top1conf):.3f})")
        else:
            print(f"{OK}合成圖推論成功 -> 偵測框數={len(r.boxes)}")
    except Exception as e:
        print(f"{BAD}合成圖推論失敗: {e}")
        problems.append(f"{key}: 推論失敗 {e}")

print("\n" + "=" * 60)
if problems:
    print(f"發現 {len(problems)} 個問題:")
    for p in problems:
        print("  - " + p)
    sys.exit(1)
else:
    print("全部通過：5 個模型皆可載入、類別吻合、推論正常。")
