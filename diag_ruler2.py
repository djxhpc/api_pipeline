# -*- coding: utf-8 -*-
"""診斷 test 資料夾 6 張尺規圖：列出 V/H 框信心值，及通過閥值後的 V-H 配對 IoU/重疊。"""
import os, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import vision_pipelinev3resnet as vp

TEST_DIR = os.path.join(os.path.dirname(__file__), "test")
ANS = {  # 檔名 -> (期望平行, 期望交叉)
    "(1).jpg": (True, True), "(2).jpe": (True, True), "(2).jpg": (True, False),
    "(3).jpg": (True, False), "(6).jpg": (False, False), "(10).jpg": (True, True),
}

model = vp._get_model("ruler")
disk = {f.strip(): f for f in os.listdir(TEST_DIR)}

for name, (exp_h, exp_cross) in ANS.items():
    fp = os.path.join(TEST_DIR, disk[name])
    img = vp._read_image_bgr(fp)
    preds = model.predict(source=img, imgsz=1280, conf=0.001, iou=0.5,
                          augment=True, device=vp.DEVICE, verbose=False)[0]
    V, H = [], []
    for b in (preds.boxes or []):
        c = float(b.conf[0]); cls = model.names[int(b.cls[0])].lower()
        (V if cls == "vertical" else H).append((c, b.xyxy[0].tolist()))
    V.sort(reverse=True); H.sort(reverse=True)

    print(f"\n=== {name}  期望: 平行={exp_h}, 交叉={exp_cross} ===")
    print("  V conf:", [round(c, 3) for c, _ in V[:8]])
    print("  H conf:", [round(c, 3) for c, _ in H[:8]])

    # 現行閥值通過的框
    Vp = [box for c, box in V if c >= vp.VERTICAL_CONF_THRESHOLD]
    Hp = [box for c, box in H if c >= vp.HORIZONTAL_CONF_THRESHOLD]
    print(f"  過閥值(V>={vp.VERTICAL_CONF_THRESHOLD},H>={vp.HORIZONTAL_CONF_THRESHOLD}): V={len(Vp)}, H={len(Hp)}")

    # 列出每個通過的 H 框：信心、與最近 V 框的 IoU
    print("  通過的 H 框 [conf 與最高V框的IoU]:")
    for c, box in [(c, box) for c, box in H if c >= vp.HORIZONTAL_CONF_THRESHOLD]:
        best = 0.0
        for vb in Vp:
            _, iou = vp._boxes_intersect(vb, box)
            best = max(best, iou)
        print(f"     conf={c:.3f}  maxIoU_with_V={best:.3f}")
