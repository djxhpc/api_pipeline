# -*- coding: utf-8 -*-
"""
field_diag.py — 「公司內部跑、產出不含影像的診斷紀錄」

目的: 影像不能外流時，在公司內對真實照片跑此工具，產生純文字/數字的紀錄，
      把紀錄檔交給工程師(或 AI)就能據此調參數、修程式，全程不需要看到原圖。

隱私設計:
  - 尺規/分類紀錄: 全是信心值與幾何數字，無影像內容。
  - 座標 OCR 文字: 預設「遮數字」(數字→#)，保留標籤與格式；--coord-values 才輸出真值。
  - 檔名: --hash-names 可改成雜湊，避免洩漏案件編號。

用法:
  python field_diag.py --src <真實照片資料夾> [--answer answer.txt] [--out field_out]
                       [--hash-names] [--coord-values]

輸出 (out 目錄):
  - records.jsonl   每張一行的詳細紀錄（不含影像）
  - summary.txt     彙總：準確率、混淆矩陣、信心值分布  ← 這份就足夠我幫你改程式
"""
import os, sys, json, re, argparse, hashlib
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import vision_pipelinev3resnet as vp

CAT_MAP = {"一等水準點": "benchmark", "測量讀數": "coord", "尺規": "ruler"}


def mask_digits(s):
    return re.sub(r"\d", "#", s or "")


def parse_answer(path):
    exp = {}
    if not path or not os.path.exists(path):
        return exp
    disk = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'(.+?\.(?:jpe|jpeg|jpg|png|bmp|webp|tiff))\s+(.*)', line, re.I)
            if not m:
                continue
            name, rest = m.group(1).strip(), m.group(2).strip()
            e = {"raw": rest}
            for cat, t in CAT_MAP.items():
                if rest.startswith(cat):
                    e["type"] = t
                    break
            if e.get("type") == "ruler":
                e["v"] = "有垂直" in rest
                e["h"] = "有平行" in rest
                e["c"] = "(交叉)" in rest
            elif e.get("type") == "benchmark":
                mn = re.search(r'(\d{3,4})', rest)
                e["num"] = mn.group(1) if mn else None
            elif e.get("type") == "coord":
                nums = re.findall(r'-?\d+\.\d+', rest)
                if len(nums) >= 3:
                    e["n"], e["e"], e["z"] = nums[:3]
            exp[name] = e
    return exp


def ruler_raw(fp):
    """擷取尺規模型所有框的信心(不套閥值)，供閥值診斷。"""
    model = vp._get_model("ruler")
    img = vp._read_image_bgr(fp)
    preds = model.predict(source=img, imgsz=1280, conf=0.001, iou=0.5,
                          augment=True, device=vp.DEVICE, verbose=False)[0]
    V, H = [], []
    for b in (preds.boxes or []):
        c = float(b.conf[0]); cls = model.names[int(b.cls[0])].lower()
        (V if cls == "vertical" else H).append(round(c, 4))
    V.sort(reverse=True); H.sort(reverse=True)
    return V, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--answer", default=None)
    ap.add_argument("--out", default="./field_out")
    ap.add_argument("--hash-names", action="store_true")
    ap.add_argument("--coord-values", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    exp = parse_answer(args.answer)
    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(vp._IMG_EXT))

    rec_f = open(os.path.join(args.out, "records.jsonl"), "w", encoding="utf-8")
    # 彙總統計
    acc = {"class": [0, 0], "v": [0, 0], "h": [0, 0], "c": [0, 0], "num": [0, 0], "coord": [0, 0]}
    cross_cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    v_fail_top, h_fail_top = [], []   # 失敗圖的最高框信心 → 看是否「卡在閥值下」

    for fn in files:
        fp = os.path.join(args.src, fn)
        e = exp.get(fn.strip(), {})
        name = hashlib.sha1(fn.encode()).hexdigest()[:12] if args.hash_names else fn
        rec = {"name": name}

        try:
            cls, conf = vp.classify_image(fp)
        except Exception as ex:
            rec["error"] = f"classify: {ex}"
            rec_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        rec["class"] = cls
        rec["class_conf"] = conf
        if "type" in e:
            exp_cls = {v: k for k, v in CAT_MAP.items()}[e["type"]]
            # 反推: type->中文類別不直接等於 class，改比對 sub_type
            ok = (vp.CLASS_TO_TYPE.get(cls) == e["type"])
            rec["class_ok"] = ok
            acc["class"][0] += ok; acc["class"][1] += 1

        sub = vp.CLASS_TO_TYPE.get(cls)
        rec["routed_type"] = sub

        if sub == "ruler":
            V, H = ruler_raw(fp)
            vp_pass = [c for c in V if c >= vp.VERTICAL_CONF_THRESHOLD]
            hp_pass = [c for c in H if c >= vp.HORIZONTAL_CONF_THRESHOLD]
            rr = vp.run_ruler(fp)
            d = rr.get("detections", {})
            rec["ruler"] = {
                "v_conf_top": V[:10], "h_conf_top": H[:10],
                "v_pass": len(vp_pass), "h_pass": len(hp_pass),
                "best_iou": d.get("best_iou"), "is_crossed": d.get("is_crossed"),
                "status": rr.get("status"),
            }
            if e.get("type") == "ruler":
                gv, gh, gc = len(vp_pass) > 0, len(hp_pass) > 0, d.get("is_crossed", False)
                rec["ruler"]["exp"] = {"v": e["v"], "h": e["h"], "c": e["c"]}
                for k, ok in [("v", gv == e["v"]), ("h", gh == e["h"]), ("c", gc == e["c"])]:
                    acc[k][0] += ok; acc[k][1] += 1
                if e["c"] and gc: cross_cm["TP"] += 1
                elif e["c"]: cross_cm["FN"] += 1
                elif gc: cross_cm["FP"] += 1
                else: cross_cm["TN"] += 1
                # 失敗時記錄最高框信心，判斷是否閥值問題
                if gv != e["v"]:
                    v_fail_top.append(V[0] if V else 0.0)
                if gh != e["h"]:
                    h_fail_top.append(H[0] if H else 0.0)

        elif sub == "benchmark":
            br = vp.run_benchmark(fp)
            rec["benchmark"] = {"number": br.get("number"), "ocr_conf": br.get("ocr_conf"),
                                "needs_review": br.get("needs_review")}
            if e.get("type") == "benchmark":
                ok = (br.get("number") == e.get("num"))
                rec["benchmark"]["exp"] = e.get("num")
                rec["benchmark"]["ok"] = ok
                acc["num"][0] += ok; acc["num"][1] += 1

        elif sub == "coord":
            cr = vp.run_coord(fp)
            engine = vp._get_ocr_engine()
            raw = vp._ocr_with_paddle(fp, engine)
            rec["coord"] = {
                "coord_class": cr.get("coord_class"),
                "needs_review": cr.get("needs_review"),
                "n_len": len(str(cr.get("N")).split(".")[0].lstrip("-")) if cr.get("N") != "N/A" else 0,
                "e_len": len(str(cr.get("E")).split(".")[0].lstrip("-")) if cr.get("E") != "N/A" else 0,
                "z_has_dot": "." in str(cr.get("H_Z")),
                "ocr_text": raw if args.coord_values else mask_digits(raw),  # 預設遮數字
            }
            if args.coord_values:
                rec["coord"]["N"], rec["coord"]["E"], rec["coord"]["H_Z"] = cr.get("N"), cr.get("E"), cr.get("H_Z")
            if e.get("type") == "coord":
                ok = (cr.get("N") == e.get("n") and cr.get("E") == e.get("e") and cr.get("H_Z") == e.get("z"))
                rec["coord"]["ok"] = ok
                acc["coord"][0] += ok; acc["coord"][1] += 1

        rec_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rec_f.close()

    # ── summary.txt ──
    def hist(vals):
        b = {"0-.02": 0, ".02-.12": 0, ".12-.3": 0, ".3-.5": 0, ".5+": 0}
        for v in vals:
            if v < 0.02: b["0-.02"] += 1
            elif v < 0.12: b[".02-.12"] += 1
            elif v < 0.3: b[".12-.3"] += 1
            elif v < 0.5: b[".3-.5"] += 1
            else: b[".5+"] += 1
        return b

    lines = [f"總圖數: {len(files)}", ""]
    lines.append("[準確率] (有答案才計)")
    for k, label in [("class", "分類路由"), ("v", "垂直"), ("h", "平行"), ("c", "交叉"),
                     ("num", "水準點數字"), ("coord", "座標NEZ")]:
        c, t = acc[k]
        if t:
            lines.append(f"  {label}: {c}/{t} = {100*c/t:.1f}%")
    if sum(cross_cm.values()):
        lines.append(f"  交叉混淆: {cross_cm}")
        if cross_cm['FP'] + cross_cm['TN']:
            lines.append(f"  交叉誤報率: {100*cross_cm['FP']/(cross_cm['FP']+cross_cm['TN']):.1f}%")
    lines.append("")
    lines.append("[失敗診斷] 偵測失敗圖的『最高框信心』分布 (看是否卡在閥值下)")
    lines.append(f"  目前閥值 V>={vp.VERTICAL_CONF_THRESHOLD}, H>={vp.HORIZONTAL_CONF_THRESHOLD}")
    lines.append(f"  垂直失敗 {len(v_fail_top)} 張, 最高框信心分布: {hist(v_fail_top)}")
    lines.append(f"  水平失敗 {len(h_fail_top)} 張, 最高框信心分布: {hist(h_fail_top)}")
    lines.append("  -> 若失敗多落在 .02-.12 / .12-.3，代表調閥值有救；落在 0-.02 代表模型沒抓到")

    summary = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)
    print(f"\n[完成] -> {args.out}/records.jsonl, {args.out}/summary.txt")


if __name__ == "__main__":
    main()
