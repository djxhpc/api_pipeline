# api_pipeline

# Vision Pipeline — 安裝與使用說明

四模型整合影像辨識管道（分類 / 尺規 / 水準點 / 座標 OCR），
提供：

1. 核心函式庫(for api) `vision_pipeline.py`
2. FastAPI HTTP 服務 `api.py`
3. 原始程式(監視資料夾-每秒循環)`auto_classify_pipeline0615.py`

-----

## 1. 安裝(建議使用獨立環境)

```bash

pip install -r requirements.txt

```

將 `vision_pipeline.py` 中 `paths` 裡四個 `.pt` 模型的路徑改成實際路徑：

```python
paths = {
    "classify": r"/path/to/影像分類best.pt",
    "ruler":    r"/path/to/尺規bestv3.pt",
    "coordfmt": r"/path/to/判斷格式分類best2.pt",
    "bench":    r"/path/to/一等水準點best.pt",
}
```

-----


### CLI（測試用）

```bash
python vision_pipeline.py --process /data/photos/IMG_0001.jpg --output /data/output
python vision_pipeline.py --process /data/photos --output /data/output --phash --phash-threshold 6
```

-----

## 3. FastAPI 服務使用方式

### 啟動服務

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

啟動後可至 `http://<server-ip>:8000/docs` 查看互動式 API 文件（Swagger UI）。

### 端點說明

#### `GET /health`

健康檢查。

```bash
curl http://localhost:8000/health
```

#### `POST /process`

處理**伺服器本機已存在**的檔案或資料夾（適合管道與伺服器同一台機器，外部只是觸發處理）。

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
        "path": "/data/photos/IMG_0001.jpg",
        "output_dir": "/data/output",
        "phash": false
      }'
```

回傳該張影像的完整辨識結果 JSON。若 `path` 是資料夾，回傳 `{"processed": N, "output_dir": ...}`。

#### `POST /process_upload`

**從外部上傳影像檔案**並立即處理（適合管道與呼叫端不在同一台機器的情境）。

```bash
curl -X POST http://localhost:8000/process_upload \
  -F "file=@/local/path/IMG_0001.jpg" \
  -F "output_dir=/data/output" \
  -F "phash=false"
```

回傳該張影像的完整辨識結果 JSON。

#### `GET /result/{output_dir_b64}/{filename}`

讀取既有的結果 JSON（`output_dir` 需先用 URL-safe base64 編碼）。

```bash
python3 -c "import base64; print(base64.urlsafe_b64encode(b'/data/output').decode())"
# 例如輸出: L2RhdGEvb3V0cHV0

curl http://localhost:8000/result/L2RhdGEvb3V0cHV0/IMG_0001.jpg
```

-----

## 4. Python 呼叫 API 範例（requests）

```python
import requests

# 方式一：本機檔案，由伺服器直接讀取
resp = requests.post("http://localhost:8000/process", json={
    "path": "/data/photos/IMG_0001.jpg",
    "output_dir": "/data/output"
})
print(resp.json())

# 方式二：上傳檔案
with open("IMG_0002.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/process_upload",
        files={"file": f},
        data={"output_dir": "/data/output", "phash": "true"}
    )
print(resp.json())
```

-----


## 6. 輸出檔案說明

|檔案                                      |說明                                          |
|----------------------------------------|--------------------------------------------|
|`<output_dir>/<檔名>.json`                |單張影像完整辨識結果（分類、子流程結果、needs_review、reviewed）  |
|`<output_dir>/hashes_db.json`           |SHA-256 去重資料庫（含 pHash16，若啟用）                |
|`<output_dir>/suspected_duplicates.json`|疑似重複清單（僅 `PHASH_SUSPECTED_ENABLED=True` 時產生）|
|`<output_dir>/error_outputs.json`       |viewer.py 標註為 `X` 的影像清單                     |

-----
