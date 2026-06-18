# -*- coding: utf-8 -*-
"""
vision_pipeline.py — 四模型整合影像辨識管道（核心函式庫）

模型:
    classify  : 影像分類 best.pt   -> 判斷 "埋深" / "水準點" / "座標"
    ruler     : 尺規bestv3.pt      -> 埋深類，尺規交叉辨識
    bench     : 一等水準點best.pt  -> 水準點類，輔助裁切後 OCR 4位數字
    coordfmt  : 判斷格式分類best2.pt -> 座標類，判斷座標文字格式 (1-9)，再用對應 regex 解析 N/E/H
    ruler_classify: 尺規管圓孔best.pt -> 尺規次分類 (管線/人手孔/一般)

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

開啟二階段 ResNet 疑似重複偵測（預設關閉）:

    import vision_pipeline as vp
    vp.RESNET_SUSPECTED_ENABLED = True
    vp.RESNET_SIMILARITY_THRESHOLD = 0.95   # 可選，預設 0.95

== 作為 CLI 使用 ==

    python vision_pipeline.py --process <file_or_folder> --output <output_dir> [--resnet] [--resnet-threshold 0.95]

輸出:
    - <output_dir>/<原檔名>.json        : 單張影像的完整辨識結果
    - <output_dir>/hashes_db.json       : 包含 SHA-256、ResNet特徵 及 疑似重複(similar_to) 的統一資料庫
"""

import os
import sys
import json
import re
import hashlib
import argparse
import unicodedata
import base64
import numpy as np
import cv2
from PIL import Image, ImageOps


# =============================================
# GPU / CPU 自動偵測
# =============================================
DEVICE = "cuda"
ORT_PROVIDERS = ["CPUExecutionProvider"]

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, ResNet18_Weights

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
    "ruler_classify": "./models/尺規管圓孔best.pt",
}

CLASS_TO_TYPE = {
    "埋深":   "ruler",
    "水準點": "benchmark",
    "座標":   "coord",
}

IMAGE_PREFIX_CLASS_MAP = {
    "SD11": "benchmark", "SD12": "benchmark",
    "PIP05": "coord", "PIP20": "coord", "PIP31": "coord",
    "HOL04": "coord", "POL04": "coord", "OTH03": "coord",
    "ATTI01": "coord", "SD13": "coord",
    "PIP02": "ruler", "POL02": "ruler", "HOL02": "ruler",
}

CLASSIFY_CONF_THRESH = 0.0

HORIZONTAL_CONF_THRESHOLD = 0.02
VERTICAL_CONF_THRESHOLD   = 0.12

RULER_CLASSIFY_CONF_THRESH = 0.98
RULER_CLASSIFY_VALID = {"管線", "人手孔", "一般"}

BENCHMARK_YOLO_CONF    = 0.25
BENCHMARK_YOLO_PADDING = 0.3

COORD_YOLO_CONF = 0.1
COORD_YOLO_CLASS_MAP = {
    "local_NEH": 1, "simple_NEZ": 2, "multi_NEZ": 3,
    "local_surface": 4, "north_east": 5, "bottom_left": 6,
    "mixed_NEZ": 7, "nez_only": 8,
}

_IMG_EXT = ('.jpg', '.jpeg', '.jpe', '.png', '.bmp', '.webp', '.tiff')

# ── 二階段 ResNet 疑似重複偵測（預設關閉，手動開啟）─────────
RESNET_SUSPECTED_ENABLED   = True
# Cosine Similarity 門檻 (0 ~ 1.0)。越高越嚴格。
# 0.95 是個好起點，可以抓出裁切、微幅旋轉、顏色變化的變造照片。
RESNET_SIMILARITY_THRESHOLD = 0.8  


# =============================================
# 共用工具 (含 ResNet 處理)
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


# --- ResNet 特徵萃取工具 ---
_resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _get_resnet_embedding(filepath):
    """
    透過 ResNet18 計算影像特徵，回傳 base64 編碼的 512 維度字串。
    使用 base64 是為了在 JSON 中儲存陣列時節省大量空間。
    """
    try:
        model = _get_model("resnet_extractor")
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
        
        tensor = _resnet_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model(tensor)
            # L2 正規化：讓後續計算 Cosine Similarity 變成單純的內積 (Dot Product)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            
        feat_np = feat.cpu().numpy().flatten().astype(np.float32)
        # 轉換為 Base64 字串存儲
        return base64.b64encode(feat_np.tobytes()).decode('utf-8')
    except Exception as e:
        print(f"  [錯誤] 計算 ResNet 特徵失敗: {e}")
        return None

def _check_resnet_suspected(filepath, filename, output_dir, db):
    """
    計算目前影像的 ResNet 向量，與 hashes_db 中既有的特徵進行比對。
    若餘弦相似度 (Cosine Similarity) >= RESNET_SIMILARITY_THRESHOLD，
    回傳 (similar_to, similarity_score, embedding_b64_str, similar_to_sha256)；
    否則回傳 (None, None, embedding_b64_str, None)。
    """
    emb_b64 = _get_resnet_embedding(filepath)
    if emb_b64 is None:
        return None, None, None, None

    # 將剛算出的 base64 解回 numpy 陣列
    current_emb = np.frombuffer(base64.b64decode(emb_b64), dtype=np.float32)
    
    best_sim = -1.0
    best_match = None

    for fn, info in db.items():
        if fn == filename:
            continue
        if info.get("is_duplicate"):
            continue
            
        exist_b64 = info.get("resnet_embedding")
        if not exist_b64:
            continue
            
        # 解碼既有影像的特徵
        exist_emb = np.frombuffer(base64.b64decode(exist_b64), dtype=np.float32)
        
        # 餘弦相似度計算 (因前面已做過 L2 Normalize，此處內積即為 Cosine Similarity)
        sim = float(np.dot(current_emb, exist_emb))
        
        if sim >= RESNET_SIMILARITY_THRESHOLD and sim > best_sim:
            best_sim = sim
            best_match = fn

    if best_match:
        similar_to_sha256 = db[best_match].get("sha256")
        # 回傳小數點後 4 位
        return best_match, round(best_sim, 4), emb_b64, similar_to_sha256
        
    return None, None, emb_b64, None


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
    """延遲載入並快取 模型 (YOLO 或 ResNet)，依 DEVICE 載入到 GPU 或 CPU。"""
    if key not in _MODEL_CACHE:
        if key == "resnet_extractor":
            # 載入 ResNet18 作為特徵萃取器 (拔掉最後的分類層)
            print(f"[模型載入] ResNet18 特徵萃取器 (device={DEVICE})")
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Identity()  # 拔除最後的全連接層，輸出 512 維度特徵
            model.to(DEVICE)
            model.eval()
            _MODEL_CACHE[key] = model
        else:
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
    if "ocr" not in _MODEL_CACHE:
        from rapidocr_onnxruntime import RapidOCR
        print(f"[模型載入] RapidOCR (providers={ORT_PROVIDERS})")
        try:
            _MODEL_CACHE["ocr"] = RapidOCR(providers=ORT_PROVIDERS)
        except TypeError:
            print("  [警告] 此版本 RapidOCR 不支援 providers 參數，使用預設設定")
            _MODEL_CACHE["ocr"] = RapidOCR()
    return _MODEL_CACHE["ocr"]


# =============================================
# Step A: 影像分類 (classify)
# =============================================

def _classify_by_prefix(filename):
    basename = os.path.splitext(filename)[0]
    for prefix, class_type in IMAGE_PREFIX_CLASS_MAP.items():
        if basename.startswith(prefix):
            for cls_name, cls_type in CLASS_TO_TYPE.items():
                if cls_type == class_type:
                    return cls_name, 1.0, True
    return None, None, False


def classify_image(filepath):
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
    try:
        model = _get_model("ruler_classify")
        res = model(filepath, device=DEVICE, verbose=False)[0]
        conf = float(res.probs.top1conf)
        cls_name = res.names[res.probs.top1]

        if conf < RULER_CLASSIFY_CONF_THRESH:
            return "一般"
        if cls_name not in RULER_CLASSIFY_VALID:
            return cls_name if cls_name else "一般"

        return cls_name

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
    reader = _get_ocr_engine()
    yolo_model = _get_model("bench")

    try:
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
        arr = np.array(img)
    except Exception as e:
        return {"number": "Error", "ocr_conf": None, "needs_review": True, "error": str(e)}

    number, ocr_conf = "", 0.0

    try:
        for crop in _yolo_crop_regions(arr, yolo_model, BENCHMARK_YOLO_CONF, BENCHMARK_YOLO_PADDING):
            result, _ = reader(np.array(crop))
            number, ocr_conf = _pick_four(result)
            if number:
                break
    except Exception:
        pass

    if not number:
        result, _ = reader(arr)
        number, ocr_conf = _pick_four(result)

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
# ... (為了簡潔，此處隱藏與你原文中相同的 Coord 長篇邏輯，請直接沿用原本的 _COORD_PATTERNS 到 _is_valid_format 等函式) ...
# 注意：這段你原本的 OCR 解析邏輯沒有任何變更，直接照貼即可。

_NUM = r'(-?\d+\.\d+|-?\d+)'
# ... (省略中間座標辨識的 Regex 邏輯，保留原樣即可) ...

# 因為字數限制，我跳過 Regex 宣告，請在你的原始碼中保留這部分
# -------------------------------------------------------------
# ...
# -------------------------------------------------------------

# （假設 run_coord 函式在此，內容與你原本完全相同）
def run_coord(filepath):
    # ...(你的原本 run_coord 內容)
    pass


# =============================================
# 主流程：單張影像處理
# =============================================

def process_single_image(filepath, output_dir):
    """
    對單張影像執行完整流程：
      1. SHA-256 去重比對（更新 hashes_db.json）
      1b. ResNet 特徵比對 (旋轉/裁切比對)
      2. 影像分類 (classify) -> "埋深"/"水準點"/"座標"
      3. 依 CLASS_TO_TYPE 路由到對應模型流程
      4. 寫出 <output_dir>/<檔名>.json
    回傳結果 dict。
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(filepath)

    result = {
        "filename": filename,
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
        
        # 確保字典有這個 key
        if filename not in db:
            db[filename] = {}
            
        db[filename]["sha256"] = sha256
        if is_duplicate:
            db[filename]["duplicate_of"] = duplicate_of
        _save_hashes_db(output_dir, db)

    result["sha256"] = sha256
    result["is_duplicate"] = is_duplicate
    result["duplicate_of"] = duplicate_of

 
    # ── 1b. 二階段 ResNet 疑似重複偵測（手動開啟）──
    if RESNET_SUSPECTED_ENABLED and not is_duplicate:
        similar_to, similarity_score, resnet_b64, similar_to_sha256 = _check_resnet_suspected(
            filepath, filename, output_dir, db
        )
        if resnet_b64:
            db[filename]["resnet_embedding"] = resnet_b64
            _save_hashes_db(output_dir, db)
            
        if similar_to:
            db[filename]["similar_to"] = similar_to
            db[filename]["similar_to_sha256"] = similar_to_sha256
            db[filename]["similarity_score"] = similarity_score

            result["suspected_duplicate"] = {
                "similar_to": similar_to,
                "similar_to_sha256": similar_to_sha256,
                "similarity_score": similarity_score,
            }
            # 印出時用百分比表示比較直觀
            print(f"  [疑似重複] {filename} ~ {similar_to} (相似度={similarity_score*100:.1f}%)")
            
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
            result["ruler_result"] = run_ruler(filepath)
            result["ruler_classify"] = run_ruler_classify(filepath)
            
        elif sub_type == "benchmark":
            result["benchmark_result"] = run_benchmark(filepath)
        # 此處需要你的完整 run_coord 函式
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
# CLI 入口
# =============================================

def main():
    global RESNET_SUSPECTED_ENABLED, RESNET_SIMILARITY_THRESHOLD

    parser = argparse.ArgumentParser(description="四模型整合影像辨識管道")
    parser.add_argument("--process", required=True, help="處理單張影像或一個資料夾（單次）")
    parser.add_argument("--output", required=True, help="輸出結果 JSON 的資料夾")
    parser.add_argument("--resnet", action="store_true",
                         help="開啟二階段 ResNet 疑似重複偵測（預設關閉）")
    parser.add_argument("--resnet-threshold", type=float, default=None,
                         help="ResNet 餘弦相似度門檻（預設 0.95，需搭配 --resnet）")
    args = parser.parse_args()

    if args.resnet:
        RESNET_SUSPECTED_ENABLED = True
    if args.resnet_threshold is not None:
        RESNET_SIMILARITY_THRESHOLD = args.resnet_threshold

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
