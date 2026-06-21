# -*- coding: utf-8 -*-
"""
synth_ruler.py — 尺規合成測試資料產生器 ＋ 評估器

流程:
  1. harvest : 用 ruler YOLO 從你現有照片裁出「高信心」的垂直/水平標尺 -> 素材庫
  2. gen     : 程式生成漸層背景，隨機貼上標尺，幾何上自動標註：
                 - 有垂直無平行(無交叉 沒碰到)
                 - 有垂直有平行(無交叉 沒碰到)
                 - 有垂直有平行(交叉)
               輸出 images/ 與 answer.txt（格式同你的 test/answer.txt）
  3. eval    : 對生成圖跑 run_ruler，比對標準答案，算垂直/平行/交叉準確率

用法:
  python synth_ruler.py gen  --out ./synth --n 100
  python synth_ruler.py eval --out ./synth

注意: 前景是真實標尺、背景是合成漸層（屬「分布外」），偵測率是參考下限，
      非真實工地準確率；此工具最大價值是「大量、可控、自動標註」地壓測。
"""
import os
import sys
import argparse
import random
import numpy as np
import cv2
from PIL import Image
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import vision_pipelinev3resnet as vp

DEFAULT_SRC = [
    r"D:\work2\0526ocr\testorror\尺規",
    r"D:\work2\0526ocr\埋深",
    r"D:\work2\api_pipeline\test",
]

# 場景 -> answer.txt 描述
SCENARIOS = {
    "v_only":  "尺規 偵測有垂直無平行(無交叉 沒碰到)",
    "vh_apart": "尺規 偵測有垂直有平行(無交叉 沒碰到)",
    "vh_cross": "尺規 偵測有垂直有平行(交叉)",
}


# ───────────────────────── harvest ─────────────────────────
def harvest(src_dirs, lib_dir, conf=0.45, max_per_class=300):
    model = vp._get_model("ruler")
    vdir = os.path.join(lib_dir, "vertical")
    hdir = os.path.join(lib_dir, "horizontal")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(hdir, exist_ok=True)
    cnt = {"vertical": 0, "horizontal": 0}

    for d in src_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(vp._IMG_EXT):
                continue
            fp = os.path.join(d, fn)
            try:
                img = vp._read_image_bgr(fp)
            except Exception:
                continue
            preds = model.predict(source=img, imgsz=1280, conf=conf, iou=0.5,
                                  augment=False, device=vp.DEVICE, verbose=False)[0]
            for b in (preds.boxes or []):
                cls = model.names[int(b.cls[0])].lower()
                if cls not in ("vertical", "horizontal"):
                    continue
                if cnt[cls] >= max_per_class:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                crop = img[max(0, y1):y2, max(0, x1):x2]
                ch, cw = crop.shape[:2]
                if ch < 40 or cw < 12:
                    continue
                # 形狀檢查：垂直要瘦高、水平要矮寬
                if cls == "vertical" and ch < cw * 1.5:
                    continue
                if cls == "horizontal" and cw < ch * 1.5:
                    continue
                out = os.path.join(vdir if cls == "vertical" else hdir,
                                   f"{cls}_{cnt[cls]:04d}.png")
                cv2.imwrite(out, crop)
                cnt[cls] += 1
    print(f"[harvest] vertical={cnt['vertical']}, horizontal={cnt['horizontal']} -> {lib_dir}")
    return cnt


def _load_lib(lib_dir):
    def load(sub):
        d = os.path.join(lib_dir, sub)
        files = [os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d) else []
        return [cv2.imread(f) for f in files if cv2.imread(f) is not None]
    return load("vertical"), load("horizontal")


# ───────────────────────── compose ─────────────────────────
def _background(w, h):
    c1 = np.array([random.randint(60, 200) for _ in range(3)], np.float32)
    c2 = np.array([random.randint(60, 200) for _ in range(3)], np.float32)
    t = np.linspace(0, 1, h)[:, None, None]
    bg = (c1 * (1 - t) + c2 * t)  # 垂直漸層
    bg = np.repeat(bg, w, axis=1)
    bg += np.random.normal(0, 6, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _alpha(h, w, border=10):
    border = max(1, min(border, w // 4, h // 4))
    a = np.ones((h, w), np.float32)
    for i in range(border):
        v = (i + 1) / border
        a[i, :] = np.minimum(a[i, :], v)
        a[h - 1 - i, :] = np.minimum(a[h - 1 - i, :], v)
        a[:, i] = np.minimum(a[:, i], v)
        a[:, w - 1 - i] = np.minimum(a[:, w - 1 - i], v)
    return a


def _paste(bg, fg, x, y):
    H, W = bg.shape[:2]
    h, w = fg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = fg[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = _alpha(sub.shape[0], sub.shape[1])[..., None]
    bg[y0:y1, x0:x1] = (a * sub + (1 - a) * bg[y0:y1, x0:x1]).astype(np.uint8)


def _scale_to(crop, target, dim):
    """把 crop 縮放成 (dim='h' 高 / 'w' 寬) = target，保持比例。"""
    ch, cw = crop.shape[:2]
    s = target / (ch if dim == "h" else cw)
    nw, nh = max(1, int(cw * s)), max(1, int(ch * s))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def compose_one(verts, horis, scenario):
    W = random.randint(560, 820)
    H = random.randint(560, 820)
    bg = _background(W, H)

    # 垂直標尺
    v = _scale_to(random.choice(verts), int(H * random.uniform(0.5, 0.8)), "h")
    vw, vh = v.shape[1], v.shape[0]
    vx = random.randint(int(W * 0.15), int(W * 0.85) - vw) if W * 0.85 - vw > W * 0.15 else (W - vw) // 2
    vy = random.randint(0, max(0, H - vh))
    vbox = (vx, vy, vx + vw, vy + vh)
    vcx = vx + vw // 2

    h_box = None
    if scenario != "v_only":
        hh_img = _scale_to(random.choice(horis), int(W * random.uniform(0.4, 0.7)), "w")
        hw, hh = hh_img.shape[1], hh_img.shape[0]
        if scenario == "vh_cross":
            # 水平橫跨垂直：x 範圍涵蓋垂直中心、y 落在垂直身上
            hx = int(vcx - hw * random.uniform(0.3, 0.7))
            hcy = int(vy + vh * random.uniform(0.25, 0.75))
            hy = hcy - hh // 2
        else:  # vh_apart：放在垂直框下方或上方，保證不重疊
            hx = random.randint(0, max(0, W - hw))
            below = H - vbox[3] - hh
            above = vbox[1] - hh
            if below > 4:
                hy = random.randint(vbox[3] + 2, H - hh)
            elif above > 4:
                hy = random.randint(0, vbox[1] - hh - 2)
            else:  # 沒空間就移到側邊（x 不重疊）
                hy = random.randint(0, max(0, H - hh))
                hx = 0 if vcx > W // 2 else W - hw
        h_box = (hx, hy, hx + hw, hy + hh)
        _paste(bg, hh_img, hx, hy)

    _paste(bg, v, vx, vy)  # 垂直最後貼（交叉時壓在水平之上，視覺較自然）

    # 幾何驗證 crossing（以防 apart 意外重疊）
    crossed = bool(h_box and _overlap(vbox, h_box))
    if scenario == "vh_apart" and crossed:
        scenario = "vh_cross"  # 萬一重疊就改標籤，保持答案正確
    if scenario == "vh_cross" and not crossed:
        scenario = "vh_apart"
    return bg, scenario


# ───────────────────────── modes ─────────────────────────
def do_gen(args):
    lib = os.path.join(args.out, "lib")
    if args.reharvest or not os.path.isdir(os.path.join(lib, "vertical")):
        harvest(args.src, lib, conf=args.conf)
    verts, horis = _load_lib(lib)
    if not verts:
        print("[錯誤] 素材庫沒有垂直標尺，無法生成。請確認 --src 有尺規照片。")
        sys.exit(1)
    if not horis:
        print("[警告] 沒有水平標尺素材，只能生成 v_only 場景。")

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    pool = ["v_only", "vh_apart", "vh_cross"] if horis else ["v_only"]

    lines = []
    for i in range(args.n):
        scen = random.choice(pool)
        bg, scen = compose_one(verts, horis, scen)
        name = f"synth_{i:04d}.jpg"
        cv2.imwrite(os.path.join(img_dir, name), bg)
        lines.append(f"{name} {SCENARIOS[scen]}")
    with open(os.path.join(args.out, "answer.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[gen] 產生 {args.n} 張 -> {img_dir}\n[gen] 答案 -> {os.path.join(args.out, 'answer.txt')}")


def do_eval(args):
    img_dir = os.path.join(args.out, "images")
    ans = os.path.join(args.out, "answer.txt")
    rec = {"v": [0, 0], "h": [0, 0], "cross": [0, 0]}  # [correct, total]
    cross_conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    with open(ans, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, rest = line.split(" ", 1)
            exp_v = "有垂直" in rest
            exp_h = "有平行" in rest
            exp_c = "(交叉)" in rest
            fp = os.path.join(img_dir, name)
            d = vp.run_ruler(fp).get("detections", {})
            got_v = d.get("vertical_count", 0) > 0
            got_h = d.get("horizontal_count", 0) > 0
            got_c = d.get("is_crossed", False)
            for k, ok in [("v", got_v == exp_v), ("h", got_h == exp_h), ("cross", got_c == exp_c)]:
                rec[k][0] += ok
                rec[k][1] += 1
            if exp_c and got_c: cross_conf["TP"] += 1
            elif exp_c and not got_c: cross_conf["FN"] += 1
            elif not exp_c and got_c: cross_conf["FP"] += 1
            else: cross_conf["TN"] += 1

    print("\n===== 評估結果 =====")
    for k, label in [("v", "垂直偵測"), ("h", "平行偵測"), ("cross", "交叉判定")]:
        c, t = rec[k]
        print(f"  {label}: {c}/{t} = {100*c/t:.1f}%" if t else f"  {label}: 無資料")
    print(f"  交叉混淆: {cross_conf}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--out", default="./synth")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--src", nargs="*", default=DEFAULT_SRC)
    g.add_argument("--conf", type=float, default=0.45)
    g.add_argument("--reharvest", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--out", default="./synth")

    args = p.parse_args()
    if args.mode == "gen":
        do_gen(args)
    else:
        do_eval(args)


if __name__ == "__main__":
    main()
