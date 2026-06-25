# -*- coding: utf-8 -*-
"""
viewer.py — 影像辨識結果可視化標註程式

讀取 pipeline_watcher.py 產生的 <output_dir>/<檔名>.json 結果，
搭配原圖一起顯示，讓使用者快速標註，並輸出答案資料庫 answer.txt
"""

import os
import json
import argparse
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageOps, ImageTk


def load_records(output_dir):
    """讀取 output_dir 下所有 *.json（排除 hashes_db.json / error_outputs.json）。"""
    records = []
    for fn in sorted(os.listdir(output_dir)):
        if not fn.lower().endswith(".json"):
            continue
        if fn in ("hashes_db.json", "error_outputs.json"):
            continue
        path = os.path.join(output_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            records.append((path, data))
        except Exception as e:
            print(f"[警告] 無法讀取 {path}: {e}")
    return records


def save_record(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def append_error_output(output_dir, data):
    err_path = os.path.join(output_dir, "error_outputs.json")
    if os.path.exists(err_path):
        with open(err_path, 'r', encoding='utf-8') as f:
            err_list = json.load(f)
    else:
        err_list = []

    # 以 filename 去重，後者覆蓋前者
    err_list = [e for e in err_list if e.get("filename") != data.get("filename")]
    err_list.append(data)

    with open(err_path, 'w', encoding='utf-8') as f:
        json.dump(err_list, f, ensure_ascii=False, indent=4)


def remove_from_error_output(output_dir, filename):
    err_path = os.path.join(output_dir, "error_outputs.json")
    if not os.path.exists(err_path):
        return
    with open(err_path, 'r', encoding='utf-8') as f:
        err_list = json.load(f)
    err_list = [e for e in err_list if e.get("filename") != filename]
    with open(err_path, 'w', encoding='utf-8') as f:
        json.dump(err_list, f, ensure_ascii=False, indent=4)


def get_standard_result(data):
    """ 將辨識結果轉換為 answer.txt 用的標準輸出格式 """
    cls = data.get('class', '未知')

    if cls == '水準點':
        val = data.get('benchmark_result', {}).get('number')
        return val if val not in (None, "") else "請重新拍攝"

    elif cls == '埋深':
        val = data.get('ruler_result', {}).get('status')
        return val if val not in (None, "") else "請重新拍攝"

    elif cls == '座標':
        cr = data.get('coord_result', {})
        n = cr.get('N')
        e = cr.get('E')
        h = cr.get('H_Z')
        if None in (n,e,h):
            return "請重新拍攝"
        return f"N: {n}, E: {e}, H_Z: {h}"

    else:
        return "請重新拍攝"


def write_to_answer_txt(output_dir, filename, cls, value, modified=False, original_value=None):
    """ 寫入答案資料庫，自動覆蓋同一檔名的舊紀錄 """
    ans_path = os.path.join(output_dir, "answer.txt")

    # 建立新的行
    line = f"{filename}\t{cls}\t{value}"
    if modified:
        line += f" [X] 原:{original_value} 改:{value}"
    elif modified is None:
        line += " [X]"

    lines = []
    # 讀取舊的內容，過濾掉同一個檔名的舊紀錄
    if os.path.exists(ans_path):
        with open(ans_path, 'r', encoding='utf-8') as f:
            for l in f:
                l = l.rstrip('\n')
                if not l.startswith(filename + "\t"):
                    lines.append(l)

    # 加入新的紀錄在最後
    lines.append(line)

    # 寫回整個檔案
    with open(ans_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def format_summary(data):
    """產生右側摘要文字。"""
    lines = []
    lines.append(f"檔名: {data.get('filename', 'N/A')}")
    lines.append(f"分類: {data.get('class', 'N/A')}  (信心 {data.get('class_confidence', 'N/A')})")
    lines.append(f"重複: {'是 -> ' + str(data.get('duplicate_of')) if data.get('is_duplicate') else '否'}")
    lines.append(f"狀態: {data.get('needs_review', 'N/A')}")
    lines.append("")

    if "ruler_result" in data:
        rr = data["ruler_result"]
        lines.append("【尺規辨識】")
        lines.append(f"  status: {rr.get('status')}")
        det = rr.get("detections", {})
        if det:
            lines.append(f"  vertical 數: {det.get('vertical_count')}")
            lines.append(f"  horizontal 數: {det.get('horizontal_count')}")
            lines.append(f"  is_crossed: {det.get('is_crossed')}")
            lines.append(f"  best_iou: {det.get('best_iou')}")

    if "benchmark_result" in data:
        br = data["benchmark_result"]
        lines.append("【水準點辨識】")
        lines.append(f"  number: {br.get('number')}")
        lines.append(f"  ocr_conf: {br.get('ocr_conf')}")

    if "coord_result" in data:
        cr = data["coord_result"]
        lines.append("【座標 OCR】")
        lines.append(f"  N: {cr.get('N')}")
        lines.append(f"  E: {cr.get('E')}")
        lines.append(f"  H/Z: {cr.get('H_Z')}")
        lines.append(f"  格式類別: {cr.get('coord_class')}")
    
    if "suspected_duplicate" in data and isinstance(data["suspected_duplicate"], dict):
        dup = data["suspected_duplicate"]
        lines.append("【疑似重複相片】")
        lines.append(f"  similar_to: {dup.get('similar_to')}")
        lines.append(f"  similarity_score: {dup.get('similarity_score')}")

    if data.get("error"):
        lines.append("")
        lines.append(f"[錯誤] {data['error']}")

    if data.get("reviewed"):
        lines.append("")
        lines.append(f">>> 已標註: {data['reviewed']}")

    return "\n".join(lines)


class ViewerApp:
    def __init__(self, root, output_dir, image_root=None):
        self.root = root
        self.output_dir = output_dir
        self.image_root = image_root
        self.records = load_records(output_dir)
        self.index = 0
        self.current_data = None
        self.filename_to_path = self.build_filename_to_path_map()
        self._tk_dup_img = None
        self.dup_popup = None  # 用於記錄彈出視窗物件
        # 圖片縮放相關的屬性
        self.original_image = None  # 原始圖片
        self.display_image = None   # 當前顯示的圖片
        self.zoom_level = 1.0       # 當前縮放等級
        self.min_zoom = 0.1         # 最小縮放等級
        self.max_zoom = 10.0        # 最大縮放等級
        self.zoom_step = 0.1        # 每次滾輪的縮放步長
        
        # Canvas 顯示區域尺寸
        self.canvas_width = 700
        self.canvas_height = 600

        if self.image_root:
            print(f"[資訊] 使用手動指定的圖片根目錄: {self.image_root}")

        self.root.title("影像辨識結果檢視 / 標註")
        self.root.geometry("1100x700")

        # ── 左側：圖片預覽（使用 Canvas 實現滾動縮放） ──
        self.image_canvas = tk.Canvas(root, bg="#222222", highlightthickness=0)
        self.image_canvas.place(x=10, y=10, width=700, height=600)
        
        # 綁定滾輪事件（以滑鼠位置為中心縮放）
        self.image_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.image_canvas.bind("<Button-4>", self.on_mousewheel)   # Linux 向上
        self.image_canvas.bind("<Button-5>", self.on_mousewheel)   # Linux 向下
        
        # 綁定滑鼠拖動事件（按住左鍵拖動圖片）
        self.image_canvas.bind("<ButtonPress-1>", self.start_drag)
        self.image_canvas.bind("<B1-Motion>", self.drag_image)
        self.image_canvas.bind("<ButtonRelease-1>", self.stop_drag)

        # ── 右側：摘要文字 ──
        self.summary_text = tk.Text(root, wrap="word", font=("Microsoft JhengHei", 11))
        self.summary_text.place(x=720, y=10, width=370, height=600)

        # ── 底部：控制按鈕 ──
        self.status_label = tk.Label(root, text="", font=("Microsoft JhengHei", 11))
        self.status_label.place(x=10, y=620, width=400, height=30)

        btn_prev = ttk.Button(root, text="← 上一張", command=self.prev_record)
        btn_prev.place(x=420, y=620, width=90, height=40)

        btn_next = ttk.Button(root, text="下一張 →", command=self.next_record)
        btn_next.place(x=520, y=620, width=90, height=40)

        btn_pass = ttk.Button(root, text="O 通過 (鍵盤 O)", command=lambda: self.mark("O"))
        btn_pass.place(x=720, y=620, width=160, height=40)

        btn_fail = ttk.Button(root, text="X 修改 / 駁回 (鍵盤 X)", command=lambda: self.mark("X"))
        btn_fail.place(x=890, y=620, width=160, height=40)

        # 縮放控制按鈕
        btn_zoom_in = ttk.Button(root, text="放大 +", command=self.zoom_in)
        btn_zoom_in.place(x=10, y=660, width=80, height=30)
        
        btn_zoom_out = ttk.Button(root, text="縮小 -", command=self.zoom_out)
        btn_zoom_out.place(x=100, y=660, width=80, height=30)
        
        btn_fit = ttk.Button(root, text="原始畫面", command=self.fit_to_screen)
        btn_fit.place(x=190, y=660, width=90, height=30)
        
        
        # 顯示當前縮放等級
        self.zoom_label = tk.Label(root, text="縮放: 100%", font=("Microsoft JhengHei", 10))
        self.zoom_label.place(x=400, y=660, width=100, height=30)

        # ── 鍵盤綁定 ──
        root.bind("<Key-o>", lambda e: self.mark("O"))
        root.bind("<Key-O>", lambda e: self.mark("O"))
        root.bind("<Key-x>", lambda e: self.mark("X"))
        root.bind("<Key-X>", lambda e: self.mark("X"))
        root.bind("<Left>", lambda e: self.prev_record())
        root.bind("<Right>", lambda e: self.next_record())
        root.bind("<plus>", lambda e: self.zoom_in())
        root.bind("<equal>", lambda e: self.zoom_in())
        root.bind("<minus>", lambda e: self.zoom_out())
        root.bind("<Key-0>", lambda e: self.fit_to_screen())

        self.show_record()

    def on_mousewheel(self, event):
        """處理滑鼠滾輪事件，以滑鼠位置為中心縮放"""
        if self.original_image is None:
            return
            
        # 取得滑鼠在 Canvas 上的位置
        mouse_x = event.x
        mouse_y = event.y
        
        # 計算縮放前，滑鼠對應的圖片座標
        # 先取得目前圖片在 Canvas 上的位置
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # 計算目前圖片顯示的尺寸
        orig_w, orig_h = self.original_image.size
        disp_w = int(orig_w * self.zoom_level)
        disp_h = int(orig_h * self.zoom_level)
        
        # 計算圖片左上角位置（居中）
        img_x = (canvas_width - disp_w) // 2
        img_y = (canvas_height - disp_h) // 2
        
        # 滑鼠在圖片上的相對位置（0~1）
        rel_x = (mouse_x - img_x) / disp_w if disp_w > 0 else 0.5
        rel_y = (mouse_y - img_y) / disp_h if disp_h > 0 else 0.5
        
        # 根據滾輪方向調整縮放等級
        if event.num == 4 or event.delta > 0:
            new_zoom = self.zoom_level * 1.2  # 放大 20%
        elif event.num == 5 or event.delta < 0:
            new_zoom = self.zoom_level / 1.2  # 縮小 20%
        else:
            return
            
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            
            # 重新計算圖片顯示尺寸
            new_disp_w = int(orig_w * self.zoom_level)
            new_disp_h = int(orig_h * self.zoom_level)
            
            # 調整圖片位置，讓滑鼠下的點保持不動
            new_img_x = mouse_x - int(rel_x * new_disp_w)
            new_img_y = mouse_y - int(rel_y * new_disp_h)
            
            self.update_image_display(offset_x=new_img_x, offset_y=new_img_y)

    def zoom_in(self):
        """放大圖片"""
        if self.original_image is None:
            return
        new_zoom = min(self.max_zoom, self.zoom_level * 1.2)
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            self.update_image_display()

    def zoom_out(self):
        """縮小圖片"""
        if self.original_image is None:
            return
        new_zoom = max(self.min_zoom, self.zoom_level / 1.2)
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            self.update_image_display()

    def fit_to_screen(self):
        """適應畫面：讓圖片剛好顯示完整"""
        if self.original_image is None:
            return
        
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 10:
            canvas_width = self.canvas_width
        if canvas_height <= 10:
            canvas_height = self.canvas_height
            
        img_width, img_height = self.original_image.size
        
        # 計算縮放等級，讓圖片完整顯示在畫面中
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.zoom_level = min(scale_x, scale_y)
        
        self.update_image_display()

    def build_filename_to_path_map(self):
        """建立 filename -> absolute_path 的對照表，方便找重複相片。"""
        mapping = {}
        for _, data in self.records:
            fn = data.get("filename")
            ap = data.get("absolute_path")
            if fn and ap and os.path.exists(ap):
                mapping[fn] = ap
        return mapping

    def resolve_image_path(self, filename):
        """依 filename 找到實際圖片路徑。"""
        if not filename:
            return None

        # 1) 手動指定圖片根目錄
        if self.image_root:
            try_path = os.path.join(self.image_root, filename)
            if os.path.exists(try_path):
                return try_path

        # 2) JSON 內記錄的 absolute_path
        ap = self.filename_to_path.get(filename)
        if ap and os.path.exists(ap):
            return ap

        # 3) 保底：逐筆搜尋
        for _, data in self.records:
            if data.get("filename") == filename:
                ap = data.get("absolute_path")
                if ap and os.path.exists(ap):
                    return ap
            ap = data.get("absolute_path")
            if ap and os.path.basename(ap) == filename and os.path.exists(ap):
                return ap

        return None

    def render_duplicate_preview(self):
        """在主圖右下角顯示 suspected_duplicate 對應的縮圖，並綁定點擊事件。"""
        # 先清掉舊的縮圖與事件
        self.image_canvas.delete("dup_preview")
        self.image_canvas.tag_unbind("dup_preview", "<Button-1>")

        if not self.current_data:
            return

        dup = self.current_data.get("suspected_duplicate")
        if not isinstance(dup, dict):
            return

        similar_to = dup.get("similar_to")
        if not similar_to:
            return

        # 如果重複圖就是自己，就不顯示
        if similar_to == self.current_data.get("filename"):
            return

        dup_path = self.resolve_image_path(similar_to)
        if not dup_path:
            return

        try:
            dup_img = Image.open(dup_path)
            dup_img = ImageOps.exif_transpose(dup_img).convert("RGB")

            canvas_w = self.image_canvas.winfo_width()
            canvas_h = self.image_canvas.winfo_height()
            if canvas_w <= 10:
                canvas_w = self.canvas_width
            if canvas_h <= 10:
                canvas_h = self.canvas_height

            # 約 1/5 大小
            max_w = max(1, min(int(dup_img.width * 0.2), canvas_w // 5))
            max_h = max(1, min(int(dup_img.height * 0.2), canvas_h // 5))

            preview = dup_img.copy()
            preview.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            self._tk_dup_img = ImageTk.PhotoImage(preview)
            pw, ph = preview.size

            margin = 10
            x = canvas_w - pw - margin
            y = canvas_h - ph - margin

            # 繪製邊框 (作為點擊熱區的一部分)
            self.image_canvas.create_rectangle(
                x - 2, y - 2, x + pw + 2, y + ph + 2,
                outline="#FFD54A", width=2, tags="dup_preview", fill=""
            )

            # 繪製圖片
            img_id = self.image_canvas.create_image(
                x, y, anchor=tk.NW, image=self._tk_dup_img, tags="dup_preview"
            )
            
            # 儲存路徑到物件屬性，供點擊時使用
            self.current_dup_path = dup_path
            self.current_dup_filename = similar_to

            # 綁定點擊事件：點擊縮圖 -> 開啟大圖視窗
            self.image_canvas.tag_bind("dup_preview", "<Button-1>", lambda e: self.open_duplicate_viewer())
            
            # 滑鼠移上去變手型游標
            self.image_canvas.tag_bind("dup_preview", "<Enter>", lambda e: self.image_canvas.config(cursor="hand2"))
            self.image_canvas.tag_bind("dup_preview", "<Leave>", lambda e: self.image_canvas.config(cursor=""))

        except Exception as e:
            print(f"[警告] 無法顯示疑似重複圖片 {similar_to}: {e}")
    def open_duplicate_viewer(self):
        """開啟一個新視窗顯示重複相片的大圖。"""
        if not hasattr(self, 'current_dup_path') or not self.current_dup_path:
            return
        
        # 如果視窗已存在，先關閉旧的再開新的 (或直接 bring to front)
        if self.dup_popup and self.dup_popup.winfo_exists():
            self.dup_popup.destroy()

        popup = tk.Toplevel(self.root)
        self.dup_popup = popup
        
        filename = getattr(self, 'current_dup_filename', 'Unknown')
        popup.title(f"重複相片預覽：{filename}")
        
        # 設定視窗大小 (稍微大一點，但不超過螢幕)
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        popup.geometry(f"{int(screen_w*0.6)}x{int(screen_h*0.7)}+{int(screen_w*0.2)}+{int(screen_h*0.15)}")
        popup.configure(bg="#333333")

        # 載入大圖
        try:
            large_img = Image.open(self.current_dup_path)
            large_img = ImageOps.exif_transpose(large_img).convert("RGB")
            
            # 計算適應視窗的大小
            win_w = int(screen_w * 0.55)
            win_h = int(screen_h * 0.65)
            
            large_img.thumbnail((win_w, win_h), Image.Resampling.LANCZOS)
            tk_large_img = ImageTk.PhotoImage(large_img)
            
            # 使用 Label 顯示
            lbl = tk.Label(popup, image=tk_large_img, bg="#333333")
            lbl.image = tk_large_img  # 保持參考以免被 GC
            lbl.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            
        except Exception as e:
            lbl = tk.Label(popup, text=f"無法載入圖片：{e}", fg="white", bg="#333333")
            lbl.pack(expand=True)

        # 關閉按鈕
        btn_close = ttk.Button(popup, text="關閉 (Esc)", command=popup.destroy)
        btn_close.pack(pady=10)
        
        # 綁定 Esc 鍵關閉
        popup.bind("<Escape>", lambda e: popup.destroy())
        popup.focus_set()

    def start_drag(self, event):
        """開始拖動圖片"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_dragging = True

    def drag_image(self, event):
        """拖動圖片"""
        if not self.is_dragging:
            return
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.image_canvas.move("image", dx, dy)
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def stop_drag(self, event):
        """停止拖動"""
        self.is_dragging = False

    def update_image_display(self, offset_x=None, offset_y=None):
        """根據當前的縮放等級更新圖片顯示"""
        if self.original_image is None:
            return
            
        # 計算新的圖片尺寸
        orig_w, orig_h = self.original_image.size
        new_w = int(orig_w * self.zoom_level)
        new_h = int(orig_h * self.zoom_level)
        
        # 確保至少有 1x1 像素
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # 調整圖片大小
        self.display_image = self.original_image.resize(
            (new_w, new_h), 
            Image.Resampling.LANCZOS
        )
        
        # 轉換為 Tkinter 圖片
        self._tk_img = ImageTk.PhotoImage(self.display_image)
        
        # 清除 Canvas
        self.image_canvas.delete("all")
        
        # 取得 Canvas 尺寸
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 10:
            canvas_width = self.canvas_width
        if canvas_height <= 10:
            canvas_height = self.canvas_height
        
        # 計算圖片位置（居中，或使用指定的偏移）
        if offset_x is not None and offset_y is not None:
            x = offset_x
            y = offset_y
        else:
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
        
        # 繪製圖片
        self.image_canvas.create_image(x, y, anchor=tk.NW, image=self._tk_img, tags="image")
        
        # 更新縮放標籤
        self.zoom_label.config(text=f"縮放: {int(self.zoom_level * 100)}%")

                # 重新顯示疑似重複縮圖
        self.render_duplicate_preview()

    def show_record(self):
        # 切換圖片時，自動關閉之前的重複圖預覽視窗
        if self.dup_popup and self.dup_popup.winfo_exists():
            self.dup_popup.destroy()
            self.dup_popup = None
        if not self.records:
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert("1.0", "找不到任何結果 JSON。")
            self.status_label.config(text="0 / 0")
            return

        path, data = self.records[self.index]
        self.current_data = data
        filename = data.get("filename")

        # ======================
        # 圖片載入邏輯
        # ======================
        img_path = None

        # 優先順序1: 手動指定的圖片根目錄
        if self.image_root:
            try_path = os.path.join(self.image_root, filename)
            if os.path.exists(try_path):
                img_path = try_path

        # 優先順序2: 原本json內的絕對路徑
        if img_path is None:
            img_path = data.get("absolute_path")

        # 載入圖片
        if img_path and os.path.exists(img_path):
            try:
                # 載入原始圖片
                self.original_image = Image.open(img_path)
                self.original_image = ImageOps.exif_transpose(self.original_image).convert("RGB")
                
                # 以「適應畫面」方式顯示（看到完整影像）
                self.fit_to_screen()
                
            except Exception as e:
                self.original_image = None
                self.image_canvas.delete("all")
                self.image_canvas.create_text(
                    350, 300, 
                    text=f"無法載入圖片:\n{e}", 
                    fill="white",
                    font=("Microsoft JhengHei", 12),
                    justify=tk.CENTER
                )
        else:
            self.original_image = None
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                350, 300, 
                text=f"找不到圖片\n\n檔名: {filename}\n\n你可以加上 --path 參數指定圖片所在的資料夾", 
                fill="white",
                font=("Microsoft JhengHei", 12),
                justify=tk.CENTER
            )

        # 摘要
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", format_summary(data))

        self.status_label.config(
            text=f"{self.index + 1} / {len(self.records)}   檔案: {os.path.basename(path)}"
        )

    def mark(self, value):
        if not self.records:
            return
        path, data = self.records[self.index]
        filename = data.get('filename')
        cls = data.get('class', '未知')
        original_value = get_standard_result(data)

        data["reviewed"] = value
        save_record(path, data)

        if value == "O":
            remove_from_error_output(self.output_dir, filename)
            write_to_answer_txt(self.output_dir, filename, cls, original_value)

        elif value == "X":
            append_error_output(self.output_dir, data)

            # 跳出修改對話框
            new_value = simpledialog.askstring("修改數值", f"請輸入正確的數值:\n({cls})", initialvalue=original_value)

            if new_value is None:
                # 使用者按取消
                self.records[self.index] = (path, data)
                self.show_record()
                return

            new_value = new_value.strip()

            # 寫入答案庫
            if new_value == original_value:
                write_to_answer_txt(self.output_dir, filename, cls, new_value, modified=None)
            else:
                write_to_answer_txt(self.output_dir, filename, cls, new_value, modified=True, original_value=original_value)

        self.records[self.index] = (path, data)
        self.show_record()
        self.next_record()

    def next_record(self):
        if not self.records:
            return
        if self.index < len(self.records) - 1:
            self.index += 1
            self.show_record()

    def prev_record(self):
        if not self.records:
            return
        if self.index > 0:
            self.index -= 1
            self.show_record()


def main():
    parser = argparse.ArgumentParser(description="影像辨識結果可視化標註程式")
    parser.add_argument("--output", required=True, help="pipeline_watcher.py 的輸出資料夾")
    parser.add_argument("--path", help="可選: 手動指定原始圖片所在的根目錄，忽略json內的絕對路徑")
    args = parser.parse_args()

    root = tk.Tk()
    app = ViewerApp(root, args.output, args.path)
    root.mainloop()


if __name__ == "__main__":
    main()
