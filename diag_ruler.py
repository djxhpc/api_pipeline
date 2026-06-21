# -*- coding: utf-8 -*-
"""
diag_ruler.py — 診斷尺規偵測：把模型「原始」輸出的所有框（不套用閥值）印出來，
                判斷某張圖到底是「閥值卡掉」還是「模型根本沒抓到」。

用法:
    python diag_ruler.py <影像檔或資料夾>

輸出每張圖：所有 vertical / horizontal 框的信心值（由高到低），
並標示在現行閥值（V=0.12, H=0.02）下會不會被採用。
"""
import sys
import os
import vision_pipelinev3resnet as vp

V_TH = vp.VERTICAL_CONF_THRESHOLD
H_TH = vp.HORIZONTAL_CONF_THRESHOLD
RAW_CONF = 0.001   # 幾乎不過濾，看模型真實能力

IMG_EXT = vp._IMG_EXT


def diag_one(path, model):
    try:
        img_bgr = vp._read_image_bgr(path)
    except Exception as e:
        print(f"  [讀檔失敗] {e}")
        return

    preds = model.predict(
        source=img_bgr, imgsz=1280, conf=RAW_CONF, iou=0.5,
        augment=True, device=vp.DEVICE, verbose=False
    )[0]

    rows = []
    for box in (preds.boxes or []):
        conf = float(box.conf[0])
        try:
            cls = model.names[int(box.cls[0])].lower()
        except Exception:
            cls = str(box.cls[0])
        rows.append((cls, conf))
    rows.sort(key=lambda r: r[1], reverse=True)

    v_all = [c for cls, c in rows if cls == "vertical"]
    h_all = [c for cls, c in rows if cls == "horizontal"]
    v_pass = [c for c in v_all if c >= V_TH]
    h_pass = [c for c in h_all if c >= H_TH]

    print(f"\n=== {os.path.basename(path)} ===")
    if not rows:
        print(f"  模型輸出 0 個框（連 conf>={RAW_CONF} 都沒有）→ 純 recall 問題，調閥值無效")
    else:
        print(f"  全部框（conf>={RAW_CONF}）:")
        for cls, c in rows[:15]:
            print(f"    {cls:11s} {c:.4f}")
        if len(rows) > 15:
            print(f"    ...（共 {len(rows)} 個框）")

    # 現行判定
    verdict = "正常" if v_pass else "請重新拍攝"
    print(f"  --- 現行閥值 V>={V_TH} / H>={H_TH} ---")
    print(f"  vertical:   抓到 {len(v_all)} 個, 過閥值 {len(v_pass)} 個" +
          (f"  最高={v_all[0]:.4f}" if v_all else ""))
    print(f"  horizontal: 抓到 {len(h_all)} 個, 過閥值 {len(h_pass)} 個" +
          (f"  最高={h_all[0]:.4f}" if h_all else ""))
    print(f"  >> 現行判定: {verdict}")

    # 建議
    if not v_pass and v_all:
        print(f"  >> [可救] vertical 最高僅 {v_all[0]:.4f}，把 VERTICAL_CONF_THRESHOLD "
              f"降到 <={v_all[0]:.3f} 即可抓到（注意誤報）")
    elif not v_all:
        print(f"  >> [調閥值無效] 模型對 vertical 完全沒框，屬 recall 問題")


def main():
    if len(sys.argv) < 2:
        print("用法: python diag_ruler.py <影像檔或資料夾>")
        sys.exit(1)
    target = sys.argv[1]
    model = vp._get_model("ruler")

    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if f.lower().endswith(IMG_EXT)]
    elif os.path.isfile(target):
        files = [target]
    else:
        print(f"找不到路徑: {target}")
        sys.exit(1)

    for f in files:
        diag_one(f, model)


if __name__ == "__main__":
    main()
