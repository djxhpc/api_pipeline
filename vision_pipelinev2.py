
# -*- coding: utf-8 -*-
"""
vision_pipeline.py — 四模型整合影像辨識管道（核心函式庫）

模型:
    classify  : 影像分類 best.pt   -> 判斷 "埋深" / "水準點" / "座標"
    ruler     : 尺規bestv3.pt      -> 埋深類，尺規交叉辨識
    bench     : 一等水準點best.pt  -> 水準點類，輔助裁切後 OCR 4位數字
    coordfmt  : 判斷格式分類best2.pt -> 座標類，判斷座標文字格式 (1-9)，再用對應 regex 解析 N/E/H

CLASS_TO_TYPE = {
    "埋深":   "ruler",
    "水準點": "benchmark",
    "座標":   "coord",
}
（影像分類模型已無 "其他" 類別）

== GPU / CPU 自動偵測 ==

    模組載入時會自動偵測 CUDA 是否可用：
      - 可用 -> DEVICE = "cuda"，所有 YOLO 模型推論使用 GPU
      - 不可用或偵測失敗 -> 自動回退 DEVICE = "cpu"
    並會在 import 時印出偵測結果，例如：
      [GPU 偵測] CUDA 可用 -> 使用 GPU (cuda)
      [GPU 偵測] CUDA 不可用 -> 回退使用 CPU

    RapidOCR (onnxruntime) 會嘗試使用 CUDAExecutionProvider，失敗則自動回退 CPU。

== 作為函式庫使用 ==

    from vision_pipeline import process_single_image, process_folder_once

    # 處理單張影像
    result = process_single_image("/path/to/photo.jpg", output_dir="/path/to/output")

    # 處理整個資料夾（只處理尚未產生 json 結果的新照片）
    new_count = process_folder_once("/path/to/folder", output_dir="/path/to/output")

開啟二階段 pHash 疑似重複偵測（預設關閉）:

    import vision_pipeline as vp
    vp.PHASH_SUSPECTED_ENABLED = True
    vp.PHASH_SUSPECTED_THRESHOLD = 8   # 可選，預設 8

== 作為 CLI 使用 ==

    python vision_pipeline.py --process <file_or_folder> --output <output_dir> [--phash] [--phash-threshold N]

輸出:
    - <output_dir>/<原檔名>.json        : 單張影像的完整辨識結果
    - <output_dir>/hashes_db.json       : SHA-256（含 pHash16，若啟用）去重資料庫
    - <output_dir>/suspected_duplicates.json : 疑似重複清單（僅 PHASH_SUSPECTED_ENABLED 開啟時產生）
"""

import os
import sys
import json
import re
import hashlib
import argparse
import unicodedata
import numpy as np
import cv2
import imagehash
from PIL import Image, ImageOps


# =============================================
# GPU / CPU 自動偵測
# =============================================
# DEVICE 用於 YOLO 模型推論（"cuda" 或 "cpu"）。
# ORT_PROVIDERS 用於 RapidOCR (onnxruntime) 推論。
DEVICE = "cuda"
ORT_PROVIDERS = ["CPUExecutionProvider"]

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU 偵測] CUDA 可用 -> 使用 GPU ({DEVICE}: {gpu_name})")
    else:
        DEVICE = "cpu"
        print("[GPU 偵測] CUDA 不可用 -> 回退使用 CPU")
except Exception as _e:
    DEVICE = "cpu"
    print(f"[GPU 偵測] 偵測失敗 ({_e}) -> 回退使用 CPU")

# RapidOCR (onnxruntime) 的 execution provider：嘗試 CUDA，失敗則回退 CPU
try:
    import onnxruntime as _ort
    _available_providers = _ort.get_available_providers()
    if DEVICE == "cuda" and "CUDAExecutionProvider" in _available_providers:
        ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[GPU 偵測] RapidOCR(onnxruntime) -> 使用 CUDAExecutionProvider")
    else:
        ORT_PROVIDERS = ["CPUExecutionProvider"]
        if DEVICE == "cuda":
            print("[GPU 偵測] RapidOCR(onnxruntime) 無 CUDAExecutionProvider -> 回退 CPU")
        else:
            print("[GPU 偵測] RapidOCR(onnxruntime) -> 使用 CPUExecutionProvider")
except Exception as _e:
    ORT_PROVIDERS = ["CPUExecutionProvider"]
    print(f"[GPU 偵測] onnxruntime 偵測失敗 ({_e}) -> RapidOCR 回退 CPU")


# =============================================
# 設定區
# =============================================
paths = {
    "classify": "./models/影像分類bestv2.pt",
    "ruler":    "./models/尺規bestv3.pt",
    "coordfmt": "./models/判斷格式分類best2.pt",
    "bench":    "./models/一等水準點best.pt",
    "ruler_classify": "./models/尺規管圓孔best.pt",  # <== 新增這個模型
}



CLASS_TO_TYPE = {
    "埋深":   "ruler",
    "水準點": "benchmark",
    "座標":   "coord",
}

# ── 影像名稱前綴與分類對應表（優先使用，沒有符合則走 classify 模型）──
# 格式: {"前綴名稱": "分類類型"}
# 分類類型可選: "ruler" (尺規/埋深)、"benchmark" (水準點)、"coord" (座標)
# 範例: {"PIP01": "ruler", "PIP02": "benchmark", "PIP03": "coord"}
IMAGE_PREFIX_CLASS_MAP = {
    "SD11": "benchmark",
    "SD12": "benchmark",

    "PIP05": "coord",
    "PIP20": "coord",
    "PIP31": "coord",
    "HOL04": "coord",
    "POL04": "coord",
    "OTH03": "coord",
    "ATTI01": "coord",
    "SD13": "coord",

    "PIP02": "ruler",
    "POL02": "ruler",
    "HOL02": "ruler",
}

CLASSIFY_CONF_THRESH = 0.0  # 分類信心低於此值仍照分類結果走（沒有 fallback，已無"其他"類別）

# ── 尺規 (ruler) 判斷參數 ─────────────────────────
HORIZONTAL_CONF_THRESHOLD = 0.02
VERTICAL_CONF_THRESHOLD   = 0.12

RULER_CLASSIFY_CONF_THRESH = 0.98
# ── 水準點 (benchmark) 參數 ───────────────────────
BENCHMARK_YOLO_CONF    = 0.25
BENCHMARK_YOLO_PADDING = 0.3

# ── 座標 (coord) 參數 ─────────────────────────────
COORD_YOLO_CONF = 0.1
COORD_YOLO_CLASS_MAP = {
    "local_NEH":     1,
    "simple_NEZ":    2,
    "multi_NEZ":     3,
    "local_surface": 4,
    "north_east":    5,
    "bottom_left":   6,
    "mixed_NEZ":     7,
    "nez_only":      8,
}

_IMG_EXT = ('.jpg', '.jpeg', '.jpe', '.png', '.bmp', '.webp', '.tiff')

# ── 二階段 pHash 疑似重複偵測（預設關閉，手動開啟）─────────
# False = 停用（只做 SHA-256 精確去重）
# True  = 啟用，對「新照片」額外計算 pHash(16) 並與既有不重複照片比對，
#         若 Hamming 距離 <= PHASH_SUSPECTED_THRESHOLD，視為疑似重複，
#         結果寫入 suspected_duplicates.json，但仍照常跑分類流程。
PHASH_SUSPECTED_ENABLED   = True
PHASH_SUSPECTED_THRESHOLD = 7  # pHash(16) Hamming 距離 <= 此值視為疑似重複（建議 5-10）


# =============================================
# 共用工具
# =============================================

def _get_file_sha256(filepath, chunk_size=8192):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception:
        return None


def _load_hashes_db(output_dir):
    db_path = os.path.join(output_dir, "hashes_db.json")
    if os.path.exists(db_path):
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_hashes_db(output_dir, db):
    db_path = os.path.join(output_dir, "hashes_db.json")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=4)


def _get_phash16(filepath):
    """計算 pHash(16)，回傳 hex 字串；失敗回傳 None。"""
    try:
        return str(imagehash.phash(Image.open(filepath), hash_size=16))
    except Exception:
        return None


# def _load_suspected_db(output_dir):
#     path = os.path.join(output_dir, "suspected_duplicates.json")
#     if os.path.exists(path):
#         with open(path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     return {}


# def _save_suspected_db(output_dir, db):
#     path = os.path.join(output_dir, "suspected_duplicates.json")
#     with open(path, 'w', encoding='utf-8') as f:
#         json.dump(db, f, ensure_ascii=False, indent=4)


import imagehash

def _check_phash_suspected(filepath, filename, output_dir, db):
    """
    計算目前影像的 pHash16，與 hashes_db 中其他「非重複」照片的既有 pHash16 比對。
    若 Hamming 距離 <= PHASH_SUSPECTED_THRESHOLD，回傳 (similar_to, distance, phash_str)；
    否則回傳 (None, None, phash_str)。
    呼叫端負責將 phash_str 寫回 db[filename]["phash16"]。
    """
    ph_str = _get_phash16(filepath)
    if ph_str is None:
        return None, None, None

    ph_obj = imagehash.hex_to_hash(ph_str)
    best_dist, best_match = PHASH_SUSPECTED_THRESHOLD + 1, None

    for fn, info in db.items():
        if fn == filename:
            continue
        if info.get("is_duplicate"):
            continue
        existing_ph = info.get("phash16")
        if not existing_ph:
            continue
            
        # 【關鍵修正】：使用 int() 將 numpy.int64 轉為原生 Python int
        dist = int(ph_obj - imagehash.hex_to_hash(existing_ph))
        
        if dist <= PHASH_SUSPECTED_THRESHOLD and dist < best_dist:
            best_dist, best_match = dist, fn

    return best_match, (best_dist if best_match else None), ph_str


def _read_image_bgr(filepath):
    """讀取影像並做 EXIF 方向校正，回傳 BGR numpy array。"""
    pil = Image.open(filepath)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# =============================================
# 模型載入（延遲載入＋快取）
# =============================================
_MODEL_CACHE = {}


def _get_model(key):
    """延遲載入並快取 YOLO 模型，依 DEVICE 載入到 GPU 或 CPU。"""
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO as YOLOModel
        model_path = paths[key]
        print(f"[模型載入] {key} -> {model_path} (device={DEVICE})")
        model = YOLOModel(model_path)
        try:
            model.to(DEVICE)
        except Exception as e:
            print(f"  [警告] {key} 模型載入到 {DEVICE} 失敗 ({e})，改用 CPU")
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _get_ocr_engine():
    """延遲載入並快取 RapidOCR，依 ORT_PROVIDERS 嘗試使用 GPU。"""
    if "ocr" not in _MODEL_CACHE:
        from rapidocr_onnxruntime import RapidOCR
        print(f"[模型載入] RapidOCR (providers={ORT_PROVIDERS})")
        try:
            _MODEL_CACHE["ocr"] = RapidOCR(providers=ORT_PROVIDERS)
        except TypeError:
            # 舊版 rapidocr_onnxruntime 不支援 providers 參數，回退預設初始化
            print("  [警告] 此版本 RapidOCR 不支援 providers 參數，使用預設設定")
            _MODEL_CACHE["ocr"] = RapidOCR()
    return _MODEL_CACHE["ocr"]


# =============================================
# Step A: 影像分類 (classify)
# =============================================

def _classify_by_prefix(filename):
    """
    根據檔案名稱前綴檢查是否有對應的分類。
    回傳 (class_name, confidence, is_matched) 或 (None, None, False)
    """
    basename = os.path.splitext(filename)[0]  # 移除副檔名
    for prefix, class_type in IMAGE_PREFIX_CLASS_MAP.items():
        if basename.startswith(prefix):
            # 根據 class_type 找出對應的中文類別名稱
            for cls_name, cls_type in CLASS_TO_TYPE.items():
                if cls_type == class_type:
                    return cls_name, 1.0, True
    return None, None, False


def classify_image(filepath):
    """
    用 classify 模型對單張影像分類。
    回傳 (class_name, confidence)。
    class_name ∈ {"埋深", "水準點", "座標"}
    """
    model = _get_model("classify")
    res = model(filepath, device=DEVICE, verbose=False)[0]
    conf = float(res.probs.top1conf)
    cls_name = res.names[res.probs.top1]
    return cls_name, round(conf, 4)


# =============================================
# Step B-1: 尺規 (ruler) — 埋深類
# =============================================

def _boxes_intersect(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return False, 0.0
    inter_area = iw * ih
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)
    union = area_a + area_b - inter_area
    return True, inter_area / union


def _classify_crossing(vertical_boxes, horizontal_boxes):
    if not vertical_boxes:
        return "None Detected", False, 0.0
    if not horizontal_boxes:
        return "Normal (Crossed)", False, 0.0
    best_iou, is_crossed = 0.0, False
    for vb in vertical_boxes:
        for hb in horizontal_boxes:
            intersect, iou = _boxes_intersect(vb, hb)
            if iou > best_iou:
                best_iou = iou
            if intersect:
                is_crossed = True
    return "Normal (Crossed)", is_crossed, best_iou


def run_ruler(filepath):
    """
    尺規辨識（單張版，沿用 pipeline_0525 的 run_combined_ruler 判斷邏輯）。
    回傳結果 dict。
    """
    model = _get_model("ruler")

    try:
        img_bgr = _read_image_bgr(filepath)
    except Exception as e:
        return {"status": "Error", "reason": str(e)}

    preds = model.predict(
        source=img_bgr,
        imgsz=1280,
        conf=min(HORIZONTAL_CONF_THRESHOLD, VERTICAL_CONF_THRESHOLD),
        iou=0.5,
        augment=True,
        device=DEVICE,
        verbose=False
    )[0]

    vertical_boxes, horizontal_boxes = [], []
    for box in (preds.boxes or []):
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        try:
            cls_name = model.names[int(box.cls[0])].lower()
        except Exception:
            cls_name = str(box.cls[0])

        if cls_name == "vertical" and conf >= VERTICAL_CONF_THRESHOLD:
            vertical_boxes.append(bbox)
        elif cls_name == "horizontal" and conf >= HORIZONTAL_CONF_THRESHOLD:
            horizontal_boxes.append(bbox)

    status, is_crossed, best_iou = _classify_crossing(vertical_boxes, horizontal_boxes)

    return {
        "status": status,
        "detections": {
            "vertical_count": len(vertical_boxes),
            "horizontal_count": len(horizontal_boxes),
            "is_crossed": is_crossed,
            "best_iou": round(best_iou, 4)
        }
    }

def run_ruler_classify(filepath):
    """
    尺規次分類（單張版）：判斷是管線、人手孔等。
    若模型預測信心度低於 RULER_CLASSIFY_CONF_THRESH 或發生錯誤，則回傳 "一般"。
    """
    try:
        model = _get_model("ruler_classify")
        res = model(filepath, device=DEVICE, verbose=False)[0]
        conf = float(res.probs.top1conf)
        cls_name = res.names[res.probs.top1]
        
        # 若信心度達到標準，回傳模型判斷的類別 (例如 "管子")；否則回傳 "一般"
        if conf >= RULER_CLASSIFY_CONF_THRESH:
            return cls_name
        else:
            return "一般"
    except Exception as e:
        print(f"  [警告] ruler_classify 模型發生錯誤: {e}")
        return "一般"

# =============================================
# Step B-2: 水準點 (benchmark) — 水準點類
# =============================================

def _yolo_crop_regions(img_array, model, conf_thresh, padding):
    preds = model.predict(source=img_array, device=DEVICE, verbose=False)[0]
    h_img, w_img = img_array.shape[:2]
    crops = []
    for box in sorted(preds.boxes, key=lambda b: float(b.conf[0]), reverse=True):
        if float(box.conf[0]) < conf_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * padding), int(bh * padding)
        x1c, y1c = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2c, y2c = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)
        crop = Image.fromarray(img_array[y1c:y2c, x1c:x2c])
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC)
        crops.append(crop)
    return crops


def _pick_four(result):
    if not result:
        return "", 0.0
    for ocr_item in result:
        text, conf = ocr_item[1], float(ocr_item[2])
        if conf > 0.4 and len(text) == 4 and text.isdigit():
            return text, round(conf, 4)
    return "", 0.0


def run_benchmark(filepath):
    """
    水準點辨識（單張版）：
    策略零 - YOLO(bench) 裁切放大 -> OCR
    策略一 - 全圖 OCR
    策略二 - 中央裁切 30% / 50% 放大 -> OCR
    """
    reader = _get_ocr_engine()
    yolo_model = _get_model("bench")

    try:
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
        arr = np.array(img)
    except Exception as e:
        return {"number": "Error", "ocr_conf": None, "needs_review": True, "error": str(e)}

    number, ocr_conf = "", 0.0

    # 策略零：YOLO 裁切
    try:
        for crop in _yolo_crop_regions(arr, yolo_model, BENCHMARK_YOLO_CONF, BENCHMARK_YOLO_PADDING):
            result, _ = reader(np.array(crop))
            number, ocr_conf = _pick_four(result)
            if number:
                break
    except Exception:
        pass

    # 策略一：全圖
    if not number:
        result, _ = reader(arr)
        number, ocr_conf = _pick_four(result)

    # 策略二：中央裁切放大
    if not number:
        w, h = img.size
        for ratio in [0.3, 0.5]:
            cw, ch = int(w * ratio), int(h * ratio)
            left, top = (w - cw) // 2, (h - ch) // 2
            cropped = img.crop((left, top, left + cw, top + ch))
            enlarged = cropped.resize((cw * 2, ch * 2), Image.Resampling.BICUBIC)
            result, _ = reader(np.array(enlarged))
            number, ocr_conf = _pick_four(result)
            if number:
                break

    if number:
        return {"number": number, "ocr_conf": ocr_conf, "needs_review": False}
    return {"number": "請重新拍攝", "ocr_conf": None, "needs_review": True}


# =============================================
# Step B-3: 座標 (coord) — 座標類
# =============================================

_NUM = r'(-?\d+\.\d+|-?\d+)'

_COORD_PATTERNS = [
    (r'北座標[:=\s]*'      + _NUM, r'東座標[:=\s]*'      + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'縱軸[:=\s]*'        + _NUM, r'橫軸[:=\s]*'        + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'地表.*?[NN][:=\s]*' + _NUM, r'地表.*?[EE][:=\s]*' + _NUM, r'地表.*?[HH][:=\s]*'   + _NUM),
    (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'本地.*?[HH][:=\s]*'   + _NUM),
    (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'高程[:=\s]*'         + _NUM),
    (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高程?[:=\s]*'        + _NUM),
    (r'北[:=\s]*'          + _NUM, r'東[:=\s]*'          + _NUM, r'高度[:=\s]*'         + _NUM),
    (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[ZZ][:=\s]*'         + _NUM),
    (r'[NN][:=\s]*'        + _NUM, r'[EE][:=\s]*'        + _NUM, r'[HH][:=\s]*'         + _NUM),
]

_CLASS_PATTERNS = {
    1: [
        (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'本地.*?.[HH][:=\s]*' + _NUM),
        (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'高程[:=\s]*'         + _NUM),
    ],
    2: [
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[ZZ][:=\s]*' + _NUM),
    ],
    3: [
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'高程[:=\s]*' + _NUM),
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[ZZ][:=\s]*' + _NUM),
        (r'縱軸[:=\s]*' + _NUM, r'橫軸[:=\s]*' + _NUM, r'高程[:=\s]*' + _NUM),
    ],
    4: [
        (r'本地.*?[NN][:=\s]*' + _NUM, r'本地.*?[EE][:=\s]*' + _NUM, r'高程[:=\s]*'         + _NUM),
        (r'地表.*?[NN][:=\s]*' + _NUM, r'地表.*?[EE][:=\s]*' + _NUM, r'地表.*?[HH][:=\s]*'   + _NUM),
    ],
    5: [
        (r'北[座坐][標标][:=\s]*' + _NUM, r'[東东][座坐][標标][:=\s]*' + _NUM, r'高[座坐][標标][:=\s]*' + _NUM),
        (r'北[座坐][標标][:=\s]*' + _NUM, r'[東东][座坐][標标][:=\s]*' + _NUM, r'高程[:=\s]*'           + _NUM),
        (r'北[:=\s]*'             + _NUM, r'東[:=\s]*'                 + _NUM, r'高度[:=\s]*'           + _NUM),
        (r'北[:=\s]*'             + _NUM, r'東[:=\s]*'                 + _NUM, r'高[:=\s]*'             + _NUM),
        (r'[Nn][:=\s]*'           + _NUM, r'[Ee][:=\s]*'              + _NUM, r'[Hh][:=\s]+'           + _NUM),
    ],
    7: [
        (r'[EE][:=\s\n]*' + _NUM + r'.*?[NN][:=\s\n]*' + _NUM + r'.*?[ZZ27z][:=\s\n]*' + _NUM),
        (r'[EE][:=\s\n]*' + _NUM, r'[NN][:=\s\n]*' + _NUM, r'[ZZ27z][:=\s\n]*' + _NUM)
    ],
    8: [
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[ZZ][:=\s]*' + _NUM),
    ],
}


def _ocr_with_paddle(image_path, engine):
    result, _ = engine(image_path)
    if not result:
        return ""
    return "\n".join(item[1] for item in result if float(item[2]) > 0.3)


def _match_coord(text, patterns):
    best = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}
    m = re.search(r'NEZ[:=\s]*' + _NUM + r',' + _NUM + r',' + _NUM, text.replace(' ', ''))
    if m:
        return {"N": m.group(1), "E": m.group(2), "H_Z": m.group(3)}
    for np_, ep_, hp_ in patterns:
        mn = re.search(np_, text, re.IGNORECASE)
        me = re.search(ep_, text, re.IGNORECASE)
        mh = re.search(hp_, text, re.IGNORECASE)
        if mn and me and mh:
            return {"N": mn.group(1), "E": me.group(1), "H_Z": mh.group(1)}
        if mn and best["N"]   == "N/A": best["N"]   = mn.group(1)
        if me and best["E"]   == "N/A": best["E"]   = me.group(1)
        if mh and best["H_Z"] == "N/A": best["H_Z"] = mh.group(1)
    return best


def _strip_dms(text):
    dms_pattern = re.compile(
        r'\d{1,3}\s*[°\*oO]\s*\d{1,2}\s*[\'′]\s*\d{1,2}(?:\.\d+)?[""′\']?\s*[NSEWnsew]?',
        re.IGNORECASE
    )
    return dms_pattern.sub('', text)


def _validate_twd97(res):
    def _id(v): return len(v.split('.')[0].lstrip('-'))
    if not (
        res["N"]   != "N/A" and _id(res["N"])   == 7 and
        res["E"]   != "N/A" and _id(res["E"])   == 6 and
        res["H_Z"] != "N/A" and '.' in res["H_Z"]
    ):
        return False
    try:
        n_val = int(res["N"].split('.')[0])
        e_val = int(res["E"].split('.')[0])
        return 2_400_000 <= n_val <= 2_800_000 and 147_000 <= e_val <= 350_000
    except (ValueError, AttributeError):
        return False


def _digit_fallback_twd97(text):
    def _is_valid_h(v):
        if '.' not in v:
            return False
        ip = v.split('.')[0].lstrip('-')
        return ip != '0' and 1 <= len(ip) <= 4

    all_m = list(re.finditer(r'-?\d+(?:\.\d+)?', text))
    cn = ce = ch = None
    n_idx = e_idx = -1

    for i, m in enumerate(all_m):
        v_str = m.group()
        clean_v = v_str.split('.')[0].lstrip('-')
        if len(clean_v) == 7 and cn is None:
            val = int(clean_v)
            if 2000000 <= val <= 3000000:
                cn = v_str
                n_idx = i
                break

    if cn:
        e_candidates = []
        for i, m in enumerate(all_m):
            if i > n_idx:
                val_str = m.group().split('.')[0].lstrip('-')
                if len(val_str) == 6:
                    val = int(val_str)
                    if 100000 <= val <= 400000:
                        e_candidates.append((i, m.group()))

        for e_i, e_v in e_candidates:
            if e_i + 1 < len(all_m) and _is_valid_h(all_m[e_i + 1].group()):
                ce = e_v
                e_idx = e_i
                ch = all_m[e_i + 1].group()
                break

        if not ce and e_candidates:
            ce = e_candidates[0][1]
            e_idx = e_candidates[0][0]

    if ce and not ch:
        for i, m in enumerate(all_m):
            if i <= e_idx:
                continue
            v = m.group()
            if _is_valid_h(v):
                ch = v
                break

    return {
        "N":   cn if cn else "N/A",
        "E":   ce if ce else "N/A",
        "H_Z": ch if ch else "N/A",
    }


def _extract_coord_bottom_left(image_path, engine, crop_ratio=0.45):
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            w, h = img.size
            y_start = int(h * (1 - crop_ratio))
            x_end = min(int(w * (crop_ratio * 1.25)), w)
            crop = img.crop((0, y_start, x_end, h))
        text = _ocr_with_paddle(np.array(crop), engine)
    except Exception:
        text = ""

    bottom_left_patterns = [
        (r'N[:=\s\n]*' + _NUM, r'E[:=\s\n]*' + _NUM, r'H[:=\s\n]*' + _NUM),
        (r'[NN][:=\s]*' + _NUM, r'[EE][:=\s]*' + _NUM, r'[HH][:=\s]*' + _NUM)
    ]
    res = _match_coord(text, bottom_left_patterns)

    if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        valid_nums = []
        for line in lines:
            m_num = re.search(_NUM, line)
            if m_num:
                valid_nums.append(m_num.group(1))
        if len(valid_nums) >= 3:
            v0, v1, v2 = valid_nums[0], valid_nums[1], valid_nums[2]
            def _id(v): return len(v.split('.')[0].lstrip('-'))
            if _id(v0) == 7 and _id(v1) == 6 and '.' in v2:
                res["N"], res["E"], res["H_Z"] = v0, v1, v2

    if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
        full_text = _ocr_with_paddle(image_path, engine)
        res = _match_coord(full_text, _COORD_PATTERNS)
        if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
            res = _match_coord(full_text, bottom_left_patterns)

    return res


def _extract_coord_local_neh(image_path, engine):
    text = _ocr_with_paddle(image_path, engine)

    n_match = re.search(r'本地.*?[NN][:=\s]*' + _NUM, text, re.IGNORECASE)
    e_match = re.search(r'本地.*?[EE][:=\s]*' + _NUM, text, re.IGNORECASE)

    res = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}
    if n_match:
        res["N"] = n_match.group(1)
    if e_match:
        res["E"] = e_match.group(1)
        text_after_e = text[e_match.end():]
        h_match = re.search(r'[HH]\s+' + _NUM, text_after_e, re.IGNORECASE)
        if not h_match:
            h_match = re.search(r'^\s*' + _NUM, text_after_e, re.MULTILINE)
        if h_match:
            res["H_Z"] = h_match.group(1)

    if res["H_Z"] == "N/A":
        h2 = re.search(r'高程[:=\s]*' + _NUM, text, re.IGNORECASE)
        if h2:
            res["H_Z"] = h2.group(1)

    if not _validate_twd97(res):
        fb = _digit_fallback_twd97(text)
        if fb["N"] != "N/A": res["N"]   = fb["N"]
        if fb["E"] != "N/A": res["E"]   = fb["E"]
        if fb["H_Z"] != "N/A": res["H_Z"] = fb["H_Z"]

    return res, text


def run_coord(filepath):
    """
    座標 OCR（單張版）。
    Step A: coordfmt 模型分類格式 (1-9)
    Step B: 依格式套用對應 regex / 特殊邏輯解析 N/E/H_Z
    Step C: 嚴格格式檢查 -> needs_review
    """
    engine = _get_ocr_engine()
    coord_yolo = _get_model("coordfmt")

    # ── Step A: 格式分類 ──
    class_id = 9
    try:
        pred = coord_yolo(filepath, device=DEVICE, verbose=False)[0]
        conf = float(pred.probs.top1conf)
        cls_name = pred.names[pred.probs.top1]
        if conf >= COORD_YOLO_CONF:
            class_id = COORD_YOLO_CLASS_MAP.get(cls_name, 9)
    except Exception:
        class_id = 9

    # ── Step B: 依格式解析 ──
    if class_id == 6:
        res = _extract_coord_bottom_left(filepath, engine)
    elif class_id == 1:
        res, full_text_c1 = _extract_coord_local_neh(filepath, engine)
        if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
            res = _match_coord(full_text_c1, _COORD_PATTERNS)
    else:
        text = _ocr_with_paddle(filepath, engine)
        text = unicodedata.normalize('NFKC', text)
        text = _strip_dms(text)

        if class_id == 2:
            nez_c2 = re.search(
                r'NEZ\s*[:\s]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)',
                text, re.IGNORECASE
            )
            res = None
            if nez_c2:
                cands = [nez_c2.group(1), nez_c2.group(2), nez_c2.group(3)]
                c2_n = c2_e = c2_z = None
                for c in cands:
                    il = len(c.split('.')[0].lstrip('-'))
                    if il == 7 and c2_n is None: c2_n = c
                    elif il == 6 and c2_e is None: c2_e = c
                    elif 1 <= il <= 4 and c2_z is None: c2_z = c
                if c2_n and c2_e and c2_z:
                    res = {"N": c2_n, "E": c2_e, "H_Z": c2_z}

            if res is None:
                cleaned_text = text.replace(" ", "")
                pattern_nez_strict = (
                    r'[NN](?::|=)?' + _NUM +
                    r'[EE](?::|=)?' + _NUM +
                    r'[ZZ27z](?::|=)?' + _NUM
                )
                m_strict = re.search(pattern_nez_strict, cleaned_text, re.IGNORECASE)
                if m_strict:
                    res = {"N": m_strict.group(1), "E": m_strict.group(2), "H_Z": m_strict.group(3)}
                else:
                    n_m = re.search(r'(?:^|\n)\s*[NN]\s*[\n:=]\s*' + _NUM, text, re.IGNORECASE)
                    e_m = re.search(r'(?:^|\n)\s*[EE]\s*[\n:=]\s*' + _NUM, text, re.IGNORECASE)
                    z_m = re.search(r'(?:^|\n)\s*[ZZ27z]\s*[\n:=]\s*' + _NUM, text, re.IGNORECASE)
                    res = {
                        "N":   n_m.group(1) if n_m else "N/A",
                        "E":   e_m.group(1) if e_m else "N/A",
                        "H_Z": z_m.group(1) if z_m else "N/A",
                    }

            if not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                    res = fb

        elif class_id == 3:
            res = _match_coord(text, _CLASS_PATTERNS[3])
            if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                res = _match_coord(text, _COORD_PATTERNS)
            if not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                    res = fb

        elif class_id == 7:
            c7_bottom_text = ""
            try:
                with Image.open(filepath) as c7_img:
                    c7_img = ImageOps.exif_transpose(c7_img).convert('RGB')
                    c7_w, c7_h = c7_img.size
                    c7_crop = c7_img.crop((0, int(c7_h * 0.78), c7_w, c7_h))
                c7_bottom_text = _ocr_with_paddle(np.array(c7_crop), engine)
            except Exception:
                pass

            c7_text = c7_bottom_text if re.search(r'NEZ', c7_bottom_text, re.IGNORECASE) else text
            res = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}

            nez_comma_match = re.search(
                r'NEZ\s*[:\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)',
                c7_text, re.IGNORECASE
            )
            if nez_comma_match:
                raw_e, raw_n, raw_z = nez_comma_match.group(1), nez_comma_match.group(2), nez_comma_match.group(3)
                candidates = [raw_e, raw_n, raw_z]
                coord_n, coord_e, coord_z = None, None, None
                for c in candidates:
                    int_len = len(c.split('.')[0].lstrip('-'))
                    if int_len == 7 and coord_n is None:
                        coord_n = c
                    elif int_len == 6 and coord_e is None:
                        coord_e = c
                    elif 1 <= int_len <= 4 and coord_z is None:
                        coord_z = c
                if coord_n and coord_e and coord_z:
                    res["N"], res["E"], res["H_Z"] = coord_n, coord_e, coord_z
                else:
                    nez_comma_match = None

            if res["N"] == "N/A" or res["E"] == "N/A" or res["H_Z"] == "N/A":
                num_matches = list(re.finditer(r'-?\d+(?:\.\d+)?', c7_text))
                coord_n, coord_e, coord_z = None, None, None

                for m in num_matches:
                    val_str = m.group()
                    int_digits = len(val_str.split('.')[0].lstrip('-'))
                    if int_digits == 6 and coord_e is None:
                        coord_e = val_str
                    elif int_digits == 7 and coord_n is None:
                        coord_n = val_str

                if coord_n and coord_e:
                    res["N"], res["E"] = coord_n, coord_e
                    lines = [line.strip() for line in c7_text.split('\n') if line.strip()]

                    z_label_pattern = re.compile(
                        r'(?:高程|正高|\b[Zz]\s*[:：=\s]|EL\s*[:：=\s])', re.IGNORECASE
                    )
                    sigma_keywords = re.compile(
                        r'\b(?:sigma|pdop|hdop|vdop|rdop|dop)\b|(?:^|\s)[oO]\s*[:：]', re.IGNORECASE
                    )

                    z_found = False
                    for line in lines:
                        if z_found:
                            break
                        if sigma_keywords.search(line):
                            continue
                        if z_label_pattern.search(line):
                            for num in re.findall(r'-?\d+(?:\.\d+)?', line):
                                if num in (coord_n, coord_e):
                                    continue
                                z_int_len = len(num.split('.')[0].lstrip('-'))
                                if 1 <= z_int_len <= 4:
                                    coord_z = num
                                    z_found = True
                                    break

                    if not z_found:
                        target_line_indices = set()
                        for li, line in enumerate(lines):
                            if coord_n in line or coord_e in line:
                                target_line_indices.add(li)

                        search_order = []
                        for li in sorted(target_line_indices):
                            search_order.append(li)
                            if li + 1 < len(lines): search_order.append(li + 1)
                            if li + 2 < len(lines): search_order.append(li + 2)
                            if li - 1 >= 0: search_order.append(li - 1)

                        seen_idx = set()
                        final_order = [i for i in search_order if not (i in seen_idx or seen_idx.add(i))]
                        candidate_lines = [(lines[i], i) for i in final_order if i < len(lines)]

                        for context_line, _line_num in candidate_lines:
                            if z_found:
                                break
                            if sigma_keywords.search(context_line):
                                continue
                            if any(k in context_line.lower() for k in ['天線', '天线', 'ant', '北', '東', '东']):
                                context_line = re.sub(
                                    r'.*?(?:天線|天线|ant|antenna)[^0-9]*?\d+(?:\.\d+)?', '',
                                    context_line, flags=re.IGNORECASE
                                )
                            for num in re.findall(r'-?\d+(?:\.\d+)?', context_line):
                                if num in (coord_n, coord_e):
                                    continue
                                try:
                                    z_val = float(num)
                                except ValueError:
                                    continue
                                if abs(z_val) < 1:
                                    continue
                                z_int_len = len(num.split('.')[0].lstrip('-'))
                                if 1 <= z_int_len <= 4:
                                    coord_z = num
                                    z_found = True
                                    break

                    if coord_z and len(coord_z.split('.')[0]) == 1:
                        z_idx = c7_text.find(coord_z)
                        if z_idx > 0 and c7_text[z_idx - 1].isdigit():
                            coord_z = c7_text[z_idx - 1] + coord_z

                    if coord_z:
                        res["H_Z"] = coord_z

            if any(v == "N/A" for v in [res["N"], res["E"]]):
                res = _match_coord(c7_text, _COORD_PATTERNS)

        elif class_id in _CLASS_PATTERNS:
            res = _match_coord(text, _CLASS_PATTERNS[class_id])
            if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                res = _match_coord(text, _COORD_PATTERNS)

            if class_id == 5 and not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                    res = fb

            if class_id == 4 and not _validate_twd97(res):
                standalone_re = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*m?\s*$')
                nums_in_order = []
                for line in text.split('\n'):
                    m = standalone_re.match(line)
                    if m:
                        nums_in_order.append(m.group(1))
                coord_n, coord_e, coord_z = None, None, None
                for v in nums_in_order:
                    int_digits = len(v.split('.')[0].lstrip('-'))
                    if int_digits == 7 and coord_n is None:
                        coord_n = v
                    elif int_digits == 6 and coord_n is not None and coord_e is None:
                        coord_e = v
                    elif coord_e is not None and coord_z is None and '.' in v:
                        coord_z = v
                if coord_n and coord_e and coord_z:
                    res = {"N": coord_n, "E": coord_e, "H_Z": coord_z}
                if not _validate_twd97(res):
                    fb = _digit_fallback_twd97(text)
                    if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                        res = fb

            if class_id not in [4, 5] and not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                    res = fb
        else:
            res = _match_coord(text, _COORD_PATTERNS)
            if not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                    res = fb

    # ── Step C: 嚴格格式檢查 ──
    def _is_valid_format(r):
        def check_value(val, expected_digits):
            try:
                if val == "N/A":
                    return False
                str_val = str(val)
                if '.' not in str_val:
                    return False
                int_part = str_val.split('.')[0].lstrip('-')
                return len(int_part) == expected_digits
            except Exception:
                return False

        n_valid = check_value(r["N"], 7)
        e_valid = check_value(r["E"], 6)
        z_valid = (r["H_Z"] != "N/A" and '.' in str(r["H_Z"]))
        return n_valid and e_valid and z_valid

    res["needs_review"] = "正常" if _is_valid_format(res) else "請重新拍攝"
    res["coord_class"] = class_id
    return res


# =============================================
# 主流程：單張影像處理
# =============================================

def process_single_image(filepath, output_dir):
    """
    對單張影像執行完整流程：
      1. SHA-256 去重比對（更新 hashes_db.json）
      2. 影像分類 (classify) -> "埋深"/"水準點"/"座標"
      3. 依 CLASS_TO_TYPE 路由到對應模型流程
      4. 寫出 <output_dir>/<檔名>.json
    回傳結果 dict。
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(filepath)

    result = {
        "filename": filename,
        # "absolute_path": os.path.abspath(filepath),
    }

    # ── 1. SHA-256 去重 ──
    sha256 = _get_file_sha256(filepath)
    db = _load_hashes_db(output_dir)

    is_duplicate, duplicate_of = False, None
    if sha256 is None:
        result["sha256"] = None
        result["is_duplicate"] = False
        result["error"] = "無法讀取檔案計算 SHA-256"
        _write_result_json(output_dir, filename, result)
        return result
    else:
        for fn, info in db.items():
            if fn != filename and info.get("sha256") == sha256:
                is_duplicate = True
                duplicate_of = fn
                break
        # 更新資料庫（不論是否重複，皆記錄此檔案的 hash）
        db[filename] = {"sha256": sha256}
        if is_duplicate:
            db[filename]["duplicate_of"] = duplicate_of
        _save_hashes_db(output_dir, db)

    result["sha256"] = sha256
    result["is_duplicate"] = is_duplicate
    result["duplicate_of"] = duplicate_of

        # ── 1b. 二階段 pHash 疑似重複偵測（手動開啟：PHASH_SUSPECTED_ENABLED = True）──
        # ── 1b. 二階段 pHash 疑似重複偵測（手動開啟：PHASH_SUSPECTED_ENABLED = True）──
    if PHASH_SUSPECTED_ENABLED and not is_duplicate:
        similar_to, distance, ph_str = _check_phash_suspected(filepath, filename, output_dir, db)
        if ph_str:
            # 1. 不論有無重複，先把 phash16 存入 db
            db[filename]["phash16"] = ph_str
            
            # 2. 如果發現疑似重複的照片，將相似資訊合併寫入同一個 db 節點
            if similar_to:
                # 確保 distance 是 int
                distance = int(distance) if distance is not None else None
                
                # 寫出到單張照片 <檔名>.json 的結果
                result["suspected_duplicate"] = {
                    "similar_to": similar_to,
                    "distance": distance,
                }
                
                # 直接合併至 hashes_db 的結構中
                db[filename]["similar_to"] = similar_to
                db[filename]["distance"] = distance
                
                print(f"  [疑似重複] {filename} ~ {similar_to} (距離={distance})")
            
            # 3. 統一儲存更新後的 hashes_db (只存一個檔案)
            _save_hashes_db(output_dir, db)

    # ── 2. 影像分類（優先檢查前綴，無符合則執行模型）──
    try:
        cls_name, cls_conf, matched_by_prefix = _classify_by_prefix(filename)
        if not matched_by_prefix:
            cls_name, cls_conf = classify_image(filepath)
        else:
            print(f"  [前綴匹配] {filename} -> {cls_name}")
    except Exception as e:
        result["class"] = "Error"
        result["class_confidence"] = None
        result["error"] = f"分類模型錯誤: {e}"
        _write_result_json(output_dir, filename, result)
        return result

    result["class"] = cls_name
    result["class_confidence"] = cls_conf

    sub_type = CLASS_TO_TYPE.get(cls_name)

    # ── 3. 路由到對應模型 ──
    try:
        if sub_type == "ruler":
            # 1. 執行原本的尺規交叉辨識
            result["ruler_result"] = run_ruler(filepath)
            # 2. 執行新的次分類模型 (人手孔/管子/一般)，並放在外層結構
            result["ruler_classify"] = run_ruler_classify(filepath)
            
        elif sub_type == "benchmark":
            result["benchmark_result"] = run_benchmark(filepath)
        elif sub_type == "coord":
            result["coord_result"] = run_coord(filepath)
        else:
            result["error"] = f"未知分類類別: {cls_name}"
    except Exception as e:
        result["error"] = f"子流程錯誤 ({sub_type}): {e}"

    # ── 4. needs_review 統整 ──
    if is_duplicate:
        result["needs_review"] = "重複"
    elif sub_type == "ruler":
        rr = result.get("ruler_result", {})
        result["needs_review"] = "正常" if rr.get("status") == "Normal (Crossed)" else "請重新拍攝"
    elif sub_type == "benchmark":
        br = result.get("benchmark_result", {})
        result["needs_review"] = "請重新拍攝" if br.get("needs_review") else "正常"
    elif sub_type == "coord":
        cr = result.get("coord_result", {})
        result["needs_review"] = cr.get("needs_review", "請重新拍攝")
    else:
        result["needs_review"] = "請重新拍攝"

    _write_result_json(output_dir, filename, result)
    return result


def _write_result_json(output_dir, filename, result):
    out_path = os.path.join(output_dir, f"{filename}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"  -> {out_path}  [class={result.get('class')}, needs_review={result.get('needs_review')}]")


# =============================================
# 批次處理
# =============================================

def process_folder_once(folder, output_dir):
    """處理資料夾內所有尚未產生 json 結果的影像（不遞迴）。"""
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(_IMG_EXT) and os.path.isfile(os.path.join(folder, f))
    )
    new_count = 0
    for fn in files:
        json_path = os.path.join(output_dir, f"{fn}.json")
        if os.path.exists(json_path):
            continue
        fp = os.path.join(folder, fn)
        print(f"[處理] {fp}")
        try:
            process_single_image(fp, output_dir)
            new_count += 1
        except Exception as e:
            print(f"  [錯誤] {fn}: {e}")
    if new_count == 0:
        print("  (無新照片)")
    return new_count


# =============================================
# CLI 入口（簡易測試用，主要建議用函式庫或 API 呼叫）
# =============================================

def main():
    global PHASH_SUSPECTED_ENABLED, PHASH_SUSPECTED_THRESHOLD

    parser = argparse.ArgumentParser(description="四模型整合影像辨識管道")
    parser.add_argument("--process", required=True, help="處理單張影像或一個資料夾（單次）")
    parser.add_argument("--output", required=True, help="輸出結果 JSON 的資料夾")
    parser.add_argument("--phash", action="store_true",
                         help="開啟二階段 pHash 疑似重複偵測（預設關閉）")
    parser.add_argument("--phash-threshold", type=int, default=None,
                         help="pHash(16) Hamming 距離門檻（預設 8，需搭配 --phash）")
    args = parser.parse_args()

    if args.phash:
        PHASH_SUSPECTED_ENABLED = True
    if args.phash_threshold is not None:
        PHASH_SUSPECTED_THRESHOLD = args.phash_threshold

    target = args.process
    if os.path.isdir(target):
        process_folder_once(target, args.output)
    elif os.path.isfile(target):
        process_single_image(target, args.output)
    else:
        print(f"[錯誤] 找不到路徑: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
