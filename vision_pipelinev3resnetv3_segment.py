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
# paths = {
#     "classify": "D:/嘉義照片辨識/API/models/影像分類bestv2.pt",
#     "ruler":    "D:/嘉義照片辨識/API/models/尺規bestv3.pt",
#     "coordfmt": "D:/嘉義照片辨識/API/models/判斷格式分類best2.pt",
#     "bench":    "D:/嘉義照片辨識/API/models/一等水準點bestv2.pt",
#     "ruler_classify": "D:/嘉義照片辨識/API/models/尺規管圓孔bestv3.pt",  
# }
paths = {
    "classify": "./models/影像分類bestv2.pt",
    "ruler":    "./models/尺規分割bestv3.pt",
    "coordfmt": "./models/判斷格式分類best2.pt",
    "bench":    "./models/一等水準點bestv2.pt",
    "ruler_classify": "./models/尺規管圓孔bestv3.pt",  # <== 新增這個模型
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

# =============================================
# 開關設定
# =============================================
SAVE_MASK_VIS = True          # True = 儲存 mask 視覺化圖, False = 關閉
MASK_VIS_DIR = r"C:\Users\WF_114.WFUSION\Desktop\pin\Chiayi\mix_test3\mask_vis_output"  # 視覺化圖輸出目錄
# 新增：mask 交叉判斷的嚴格度控制
MIN_MASK_OVERLAP_PIXELS = 50     # 至少要有 50 個像素重疊
MIN_MASK_OVERLAP_RATIO = 0.01   # 交集佔較小 mask 面積的比例門檻（細長物體用這個比 IoU 更準）


HORIZONTAL_CONF_THRESHOLD = 0.4
VERTICAL_CONF_THRESHOLD   = 0.6

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
RESNET_SIMILARITY_THRESHOLD = 0.9  


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
# Step B-1: 尺規 (ruler) — 埋深類 (Segmentation 版本)
# =============================================

def _masks_intersect(mask_a, mask_b):
    """
    使用 mask 像素判斷是否有實質交叉。
    細長物體（尺規、管線）交叉時 IoU 天生很低，改用
    overlap_ratio = intersection / min(area_a, area_b) 做主要判斷。
    """
    intersection = np.count_nonzero(mask_a & mask_b)

    if intersection == 0:
        return False, 0.0

    area_a = np.count_nonzero(mask_a)
    area_b = np.count_nonzero(mask_b)
    min_area = max(min(area_a, area_b), 1)
    overlap_ratio = intersection / min_area

    union = np.count_nonzero(mask_a | mask_b)
    iou = intersection / max(union, 1)

    # 像素數和 overlap_ratio 兩個條件都要符合
    is_crossed = (intersection >= MIN_MASK_OVERLAP_PIXELS) and (overlap_ratio >= MIN_MASK_OVERLAP_RATIO)

    return is_crossed, float(iou)


def _classify_crossing(vertical_masks, horizontal_masks):
    if not vertical_masks:
        return "None Detected", False, 0.0
    if not horizontal_masks:
        return "Normal (Crossed)", False, 0.0

    best_iou, is_crossed = 0.0, False
    for vm in vertical_masks:
        for hm in horizontal_masks:
            intersect, iou = _masks_intersect(vm, hm)
            if iou > best_iou:
                best_iou = iou
            if intersect:
                is_crossed = True

    return "Normal (Crossed)", is_crossed, best_iou


def _save_mask_visualization(img_bgr, masks_data, cls_names, confs, filepath):
    """
    將偵測到的 mask 以半透明方式疊加在原圖上並存檔。
    masks_data: list of torch.Tensor (H_mask x W_mask, 值域 [0,1])
    """
    os.makedirs(MASK_VIS_DIR, exist_ok=True)

    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    # 顏色對應
    color_map = {
        "vertical":   (0, 255, 0),    # 綠色
        "horizontal": (0, 0, 255),    # 紅色
    }

    for i, mask_tensor in enumerate(masks_data):
        cls_name = cls_names[i]
        conf = confs[i]
        color = color_map.get(cls_name, (255, 255, 0))

        # 轉 numpy 並 resize 到原圖大小
        mask_np = mask_tensor.cpu().numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask_resized > 0.5

        # 半透明疊加 (alpha = 0.5)
        overlay[mask_bool] = (
            overlay[mask_bool] * 0.5 + np.array(color) * 0.5
        ).astype(np.uint8)

    # 圖例
    cv2.putText(overlay, "Green: Vertical",   (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(overlay, "Red: Horizontal",   (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 存檔
    basename = os.path.basename(filepath)
    save_path = os.path.join(MASK_VIS_DIR, f"mask_{basename}")
    cv2.imwrite(save_path, overlay)
    print(f"  [Mask Vis] 已儲存: {save_path}")


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
        device=DEVICE,
        retina_masks=True,    # 提升 mask 邊緣精度
        verbose=False
    )[0]

    vertical_masks = []     # list of numpy bool array
    horizontal_masks = []

    # 視覺化用
    vis_masks_data = []
    vis_cls_names = []
    vis_confs = []

    boxes = preds.boxes
    masks = preds.masks

    if boxes is not None and masks is not None:
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            try:
                cls_name = model.names[int(box.cls[0])].lower()
            except Exception:
                cls_name = str(box.cls[0])

            # 取得 mask → numpy bool
            mask_tensor = masks.data[i]          # torch.Tensor (H_mask, W_mask)
            mask_np = mask_tensor.cpu().numpy()
            mask_bool = mask_np > 0.5

            if cls_name == "vertical" and conf >= VERTICAL_CONF_THRESHOLD:
                vertical_masks.append(mask_bool)
                vis_masks_data.append(mask_tensor)
                vis_cls_names.append(cls_name)
                vis_confs.append(conf)

            elif cls_name == "horizontal" and conf >= HORIZONTAL_CONF_THRESHOLD:
                horizontal_masks.append(mask_bool)
                vis_masks_data.append(mask_tensor)
                vis_cls_names.append(cls_name)
                vis_confs.append(conf)

    # ---- 視覺化存檔 ----
    if SAVE_MASK_VIS and vis_masks_data:
        _save_mask_visualization(
            img_bgr, vis_masks_data, vis_cls_names, vis_confs, filepath
        )

    # ---- Mask-based 交叉判斷 ----
    status, is_crossed, best_iou = _classify_crossing(
        vertical_masks, horizontal_masks
    )

    return {
        "status": status,
        "detections": {
            "vertical_count": len(vertical_masks),
            "horizontal_count": len(horizontal_masks),
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
        # ★ NEW: OCR 容錯 — 橫軸→横由/横曲, label 後卡了垃圾文字
        (r'縱軸[^\d]*?'     + _NUM, r'[横橫][軸由曲申][^\d]*?' + _NUM, r'高程[^\d]*?' + _NUM),
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
        # 前面是北/東(N/E)時，高程優先抓「高/高程/高度」，避免誤選同圖中的 H
        (r'[Nn][:=\s]*'           + _NUM, r'[Ee][:=\s]*'              + _NUM, r'高[程度]?[:=\s]*'      + _NUM),
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


def _ocr_boxes(filepath, engine):
    """
    回傳含框幾何的 OCR 結果：[{cx, cy, xl, xr, h, text, conf}, ...]。
    供「同列標籤→數值」配對使用；讀檔/OCR 失敗回傳 []。
    """
    try:
        img_bgr = _read_image_bgr(filepath)
        raw, _ = engine(img_bgr)
    except Exception:
        return []
    items = []
    for box, text, conf in (raw or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({
            "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
            "xl": min(xs), "xr": max(xs), "h": max(ys) - min(ys),
            "text": text, "conf": float(conf),
        })
    return items


def _extract_coord_row_paired(filepath, engine):
    """
    以 OCR 框的幾何位置做「同列、標籤→右側數值」配對。
    修正：在 Grid 佈局中，優先抓取水平距離(dx)最近的數值，避免抓到右側其他欄位的數值。
    """
    res = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}
    items = _ocr_boxes(filepath, engine)
    if not items:
        return res

    def num_in_row(label):
        """找與 label 同列、在其右側、且水平距離(dx)最近的數字框。"""
        ly = label["cy"]
        lh = label["h"] or 20
        lx = label["cx"]
        
        best, best_dx = None, None
        
        for it in items:
            if it is label:
                continue
            
            # 1. 必須在標籤右側 (xl > lx)
            if it["xl"] < lx:
                continue
                
            # 2. 必須在同一行 (y 座標差距在合理範圍內)
            dy = abs(it["cy"] - ly)
            if dy > lh * 1.0: # 縮小 y 容差，避免跨行
                continue
            
            # 3. 必須包含數字
            m = re.search(r'-?\d+\.\d+|-?\d+', it["text"])
            if not m:
                continue
            
            # 4. 計算水平距離 (dx)
            # 使用 it["xl"] (框左邊緣) 減去 label["cx"] (標籤中心)
            dx = it["xl"] - lx
            
            # 優先選擇水平距離最近的 (dx 最小)
            if best is None or dx < best_dx:
                best, best_dx = m.group(), dx
                
        return best

    # 匹配標籤的 Regex
    n_re = re.compile(r'北([座坐][標标])?')
    e_re = re.compile(r'[東东]([座坐][標标])?')
    h_re = re.compile(r'(正)?高(程|度|[座坐][標标])?')

    def _clean(t):
        return re.sub(r'^[\s:：]+|[\s:：]+$', '', t)

    n_label = e_label = h_label = None
    for it in items:
        t = _clean(it["text"])
        if n_label is None and n_re.fullmatch(t):
            n_label = it
        if e_label is None and e_re.fullmatch(t):
            e_label = it
        if h_label is None and h_re.fullmatch(t):
            h_label = it

    if n_label is not None:
        v = num_in_row(n_label)
        if v: res["N"] = v
    if e_label is not None:
        v = num_in_row(e_label)
        if v: res["E"] = v
    if h_label is not None:
        v = num_in_row(h_label)
        if v: res["H_Z"] = v

    return res


def _extract_coord_nez_column(filepath, engine):
    """
    class_id=8(NEZ 編輯座標畫面)常 OCR 不到 N/E/Z 單字標籤，但數值排成一欄。
    依垂直順序取 N(7 位,2-3M)、E(6 位,N 後)、Z(E 後第一個純小數，允許 0.xxx)，
    排除 B/L 經緯度(含 ° ' " 或數字後接 NSEW)。回傳抓不到為 "N/A"。
    """
    res = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}
    items = _ocr_boxes(filepath, engine)
    if not items:
        return res

    def plain_num(t):
        t = t.strip()
        if re.search(r"[°'\"′″]", t):            # DMS 符號 -> 經緯度
            return None
        if re.search(r'\d\s*[NSEWnsew]$', t):    # 數字後接方位 -> 經緯度
            return None
        m = re.fullmatch(r'-?\d+\.\d+|-?\d+', t)
        return m.group() if m else None

    nums = [v for it in sorted(items, key=lambda i: i["cy"])
            if (v := plain_num(it["text"])) is not None]

    n_i = e_i = None
    for i, v in enumerate(nums):
        ip = v.split('.')[0].lstrip('-')
        if len(ip) == 7:
            try:
                if 2_000_000 <= int(ip) <= 3_000_000:
                    res["N"], n_i = v, i
                    break
            except ValueError:
                pass

    if n_i is not None:
        for i in range(n_i + 1, len(nums)):
            ip = nums[i].split('.')[0].lstrip('-')
            if len(ip) == 6:
                try:
                    if 100_000 <= int(ip) <= 400_000:
                        res["E"], e_i = nums[i], i
                        break
                except ValueError:
                    pass

    if e_i is not None:
        for i in range(e_i + 1, len(nums)):
            v = nums[i]
            if '.' not in v:
                continue
            ip = v.split('.')[0].lstrip('-')
            if 1 <= len(ip) <= 4:                # 允許 0.xxx
                res["H_Z"] = v
                break

    return res

def _extract_coord_label_colon(text):
    """
    從文字中抓取 N: / E: / h: 後面緊跟的數值。
    適用於 OCR 結果中出現 'N: 2741234.567' 'E: 301234.567' 'h: 123.456' 格式。
    回傳 {"N","E","H_Z"}，抓不到的欄位為 "N/A"。
    """
    res = {"N": "N/A", "E": "N/A", "H_Z": "N/A"}

    # N: 後的數值（大寫 N，排除 NE 連寫等干擾）
    m_n = re.search(r'(?<![A-Za-z])N\s*[:：]\s*(-?\d+\.?\d*)', text)
    if m_n:
        res["N"] = m_n.group(1)

    # E: 後的數值
    m_e = re.search(r'(?<![A-Za-z])E\s*[:：]\s*(-?\d+\.?\d*)', text)
    if m_e:
        res["E"] = m_e.group(1)

    # h: 後的數值（小寫 h，區分大寫 H 可能代表 HRMS 等）
    # 同時支援大寫 H: 但排除 HRMS / HSTD 等
    m_h = re.search(r'(?<![A-Za-z])h\s*[:：]\s*(-?\d+\.?\d*)', text)
    if not m_h:
        # fallback: 大寫 H: 但後面不能緊跟 R/S/D/P 等（排除 HRMS, HSTD, HDOP）
        m_h = re.search(r'(?<![A-Za-z])H\s*[:：]\s*(-?\d+\.?\d*)(?!\s*[A-Za-z])', text)
    if m_h:
        res["H_Z"] = m_h.group(1)

    return res

def _merge_coord(best, new):
    """合併座標結果：只補 best 中仍為 N/A 的欄位，不覆蓋已抓到的值。"""
    for k in ("N", "E", "H_Z"):
        if best[k] == "N/A" and new.get(k, "N/A") != "N/A":
            best[k] = new[k]
    return best

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


def _strip_precision(text):
    """
    移除 RTK 定位精度資訊(RTK Fixed H:/V:/RMS)與水平精度，避免精度小數(如 H:0.025m)
    被誤抓為高程。這些值永遠不是 N/E/高程，移除對任何座標格式都安全。
    """
    text = re.sub(r'RTK[\s.:_]*Fixed[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bRMS\s*[:：]?\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'水平精度\s*[:：]?\s*-?\d+(?:\.\d+)?\s*m?', '', text)
    # 保險：殘留的 H:/V: 接「0.xxx」(精度恆為次公尺級)，不影響合法的 H:26.654
    text = re.sub(r'\b[HV]\s*[:：]\s*0\.\d+\s*m?', '', text, flags=re.IGNORECASE)
    return text


def _find_height_geom(filepath, engine, n_val, e_val):
    """
    以幾何位置找『高程/正高』標籤對應的高程值(允許 0.xxx)，避免 digit_fallback 因前導 0
    漏抓 0.274 而誤取天線/精度。優先同列右側，其次標籤正下方(同欄)。
    僅接受合理高程(整數≤4 位且不等於 N/E)，故不會配到 6 位數的橫軸值。
    只認「高程/正高」，不配天線/大地高/水平精度。找不到回 None。
    注意：影像旋轉時幾何不可靠，呼叫端應避免使用。
    """
    items = _ocr_boxes(filepath, engine)
    if not items:
        return None
    labels = [it for it in items if re.search(r'高\s*程|正\s*高', it["text"])]
    if not labels:
        return None
    lbl = labels[0]
    lh = lbl["h"] or 25

    def height_num(it):
        m = re.search(r'-?\d+\.\d+|-?\d+', it["text"])
        if not m:
            return None
        v = m.group()
        if len(v.split('.')[0].lstrip('-')) > 4:   # 座標級(>4 位) -> 非高程
            return None
        if v in (n_val, e_val):
            return None
        return v

    # 1) 同列右側
    same_row = []
    for it in items:
        if it is lbl:
            continue
        v = height_num(it)
        if v and abs(it["cy"] - lbl["cy"]) <= lh * 0.8 and it["xl"] >= lbl["cx"]:
            same_row.append((abs(it["cy"] - lbl["cy"]), it["xl"], v))
    if same_row:
        same_row.sort()
        return same_row[0][2]

    # 2) 標籤正下方(同欄，xl 接近)
    below = []
    for it in items:
        if it is lbl:
            continue
        v = height_num(it)
        if v and 0 < it["cy"] - lbl["cy"] <= lh * 2.2 and abs(it["xl"] - lbl["xl"]) <= lh * 2:
            below.append((it["cy"] - lbl["cy"], v))
    if below:
        below.sort()
        return below[0][1]

    return None


def _retry_rotations(filepath, engine):
    """
    影像被旋轉時 OCR 會錯亂。對 90/270/180 度重新 OCR，
    回傳第一個通過 TWD97 嚴格檢查的 N/E/H_Z 結果；都失敗回傳 None。
    """
    try:
        base = ImageOps.exif_transpose(Image.open(filepath)).convert('RGB')
    except Exception:
        return None
    for deg in (90, 270, 180):
        try:
            arr = cv2.cvtColor(np.array(base.rotate(deg, expand=True)), cv2.COLOR_RGB2BGR)
            raw, _ = engine(arr)
            text = "\n".join(it[1] for it in (raw or []) if float(it[2]) > 0.3)
            text = unicodedata.normalize('NFKC', text)
            text = _strip_dms(_strip_precision(text))
            r = _match_coord(text, _COORD_PATTERNS)
            if not _validate_twd97(r):
                fb = _digit_fallback_twd97(text)
                if _validate_twd97(fb):
                    r = fb
            if _validate_twd97(r):
                return r
        except Exception:
            continue
    return None


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

    def _dec_ok(v):
        # 小數位 > 5 視為兩個值被 OCR 黏在一起(如 2596899.621165503)，排除
        return '.' not in v or len(v.split('.')[1]) <= 5

    all_m = list(re.finditer(r'-?\d+(?:\.\d+)?', text))
    cn = ce = ch = None
    n_idx = e_idx = -1

    for i, m in enumerate(all_m):
        v_str = m.group()
        clean_v = v_str.split('.')[0].lstrip('-')
        if len(clean_v) == 7 and cn is None and _dec_ok(v_str):
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
                if len(val_str) == 6 and _dec_ok(m.group()):
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


def _ocr_enhanced(filepath, engine, scale=2.0):
    """
    對比強化 + 放大後重新 OCR。
    針對「灰色 / 低對比 / 小字」造成 OCR 漏字的情況（class_id 3/5/8 常見），
    以 CLAHE 增強灰階對比再放大，協助讀出原本抓不到的數值。
    回傳 OCR 文字（無法處理時回傳空字串）。
    """
    try:
        img_bgr = _read_image_bgr(filepath)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        if scale and scale != 1.0:
            enhanced_bgr = cv2.resize(
                enhanced_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        result, _ = engine(enhanced_bgr)
    except Exception:
        return ""
    if not result:
        return ""
    return "\n".join(item[1] for item in result if float(item[2]) > 0.3)


def _fill_missing_with_enhanced(res, filepath, engine, patterns):
    """
    若 res 仍有 N/A 欄位，對影像做對比強化(CLAHE+放大)後重新 OCR，
    僅用來「補足」缺漏欄位，不覆蓋已抓到的值
    （避免清晰但非 TWD97 的合法值，例如 8 位數座標，被誤洗掉）。
    coord_class 由模型判定、可能在 4/8 等類別間飄移，故此補強不綁定特定類別。
    """
    if not any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
        return res

    g_text = _ocr_enhanced(filepath, engine)
    if not g_text:
        return res
    g_text = unicodedata.normalize('NFKC', g_text)
    g_text = _strip_dms(_strip_precision(g_text))

    g_res = _match_coord(g_text, patterns)
    gfb = _digit_fallback_twd97(g_text)
    for k in ("N", "E", "H_Z"):
        if res[k] == "N/A":
            if g_res[k] != "N/A":
                res[k] = g_res[k]
            elif gfb[k] != "N/A":
                res[k] = gfb[k]
    return res


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
        text = _strip_dms(_strip_precision(text))

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
            # 1) 乾淨標籤優先（N/E/高程、縱軸/橫軸/高程、N/E/Z）。
            #    不再 merge _COORD_PATTERNS：其 best-fill 會把座標值誤塞進 H_Z 並被鎖住。
            res = _match_coord(text, _CLASS_PATTERNS[3])
            # 清掉位數不符的誤值(E 非 6 位、N 非 7 位、H_Z 為座標級>4 位)，
            # 以免被後續 merge 鎖住而蓋不掉(如 E=0.006、H_Z=193391.638)。
            for _k, _n in (("N", 7), ("E", 6)):
                if res[_k] != "N/A" and len(res[_k].split('.')[0].lstrip('-')) != _n:
                    res[_k] = "N/A"
            if res["H_Z"] != "N/A" and len(res["H_Z"].split('.')[0].lstrip('-')) > 4:
                res["H_Z"] = "N/A"
            # 2) digit_fallback 尊重位數：完整合格就整組取代，否則只補缺
            if not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if _validate_twd97(fb):
                    res = fb
                else:
                    res = _merge_coord(res, fb)
            # 3) 影像被旋轉時(高程讀不到或被 RTK H:0.025 卡住)：轉正重抓，完整合格就整組取代
            used_rotation = False
            if not _validate_twd97(res):
                rot = _retry_rotations(filepath, engine)
                if rot:
                    res = rot
                    used_rotation = True
            # 4) 未旋轉時，用「高程」標籤的幾何位置覆蓋 H_Z：避免 digit_fallback 因前導 0
            #    漏掉 0.274 而誤抓天線(1.600)。旋轉影像的幾何不可靠，故跳過。
            if not used_rotation:
                h_geom = _find_height_geom(filepath, engine, res["N"], res["E"])
                if h_geom is not None:
                    res["H_Z"] = h_geom

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

        elif class_id == 5:
            # 基底：原本的文字/數字 fallback（digit_fallback 會覆寫掉 N2329 這類無效值）
            res = _match_coord(text, _CLASS_PATTERNS[5])
            if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                res = _match_coord(text, _COORD_PATTERNS)
            if not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                    res = fb
            # ── 新增：抓 N: / E: / h: 標籤冒號後的數值 ──
            # 移除原本的 any(N/A) 判斷，改為無條件呼叫，以便執行「N/E 同時存在時優先覆蓋 H_Z」
            colon_res = _extract_coord_label_colon(text)

            # 優先邏輯：若 N: 與 E: 同時被冒號格式抓到，代表此為同一組結構化輸出，
            # 此時 h: 的數值可信度極高，直接優先覆蓋 H_Z（即使前面已抓到值）
            if colon_res["N"] != "N/A" and colon_res["E"] != "N/A" and colon_res["H_Z"] != "N/A":
                res["H_Z"] = colon_res["H_Z"]

            # 補位邏輯：其餘欄位維持「僅填補 N/A」原則，不破壞已成功解析的結果
            for k in ("N", "E", "H_Z"):
                if res[k] == "N/A" and colon_res[k] != "N/A":
                    res[k] = colon_res[k]
            # 同列幾何配對(北→N、東→E、高程/高度→H_Z)優先覆蓋：
            # 解決數值框 y 略高於標籤、用文字順序會整體位移一格的問題；
            # 並保留清晰但非 TWD97 的合法值(如 8 位數座標)；高程優先「高程/高度/正高」，
            # 不被同圖 H(HRMS/VRMS)/大地高/相位中心高/杆高 搶走。
            row_res = _extract_coord_row_paired(filepath, engine)
            for k in ("N", "E", "H_Z"):
                if row_res[k] != "N/A":
                    res[k] = row_res[k]

        elif class_id == 8:
            # 基底：NEZ 文字 pattern + digit fallback
            res = _match_coord(text, _CLASS_PATTERNS[8])
            if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                res = _match_coord(text, _COORD_PATTERNS)
            if not _validate_twd97(res):
                fb = _digit_fallback_twd97(text)
                if fb["N"] != "N/A" or fb["E"] != "N/A" or fb["H_Z"] != "N/A":
                    res = fb

            # 幾何欄位配對：N/E/Z 同一欄，Z 取 E 下方第一個純小數(允許 0.xxx)，
            # 排除 B/L 經緯度；修正 Z 被經度(L)搶走或因前導 0 漏抓的問題。
            col = _extract_coord_nez_column(filepath, engine)
            if col["H_Z"] != "N/A":
                res["H_Z"] = col["H_Z"]
            for k in ("N", "E"):
                if res[k] == "N/A" and col[k] != "N/A":
                    res[k] = col[k]

        elif class_id in _CLASS_PATTERNS:
            res = _match_coord(text, _CLASS_PATTERNS[class_id])
            if any(v == "N/A" for v in [res["N"], res["E"], res["H_Z"]]):
                res = _match_coord(text, _COORD_PATTERNS)

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

    # ── Step B+: 灰字/低對比/小字補強（不綁定 class_id）──
    # coord_class 由 coordfmt 模型判定，同張圖可能在 4/8 等類別間飄移；
    # 故在進入嚴格檢查前，對「不完整或不合格」的結果統一做一次對比強化重抓。
    if class_id not in (6,):  # bottom_left 已有自己的多段裁切流程，不重複
        res = _fill_missing_with_enhanced(res, filepath, engine, _COORD_PATTERNS)

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
