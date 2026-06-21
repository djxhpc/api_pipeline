# -*- coding: utf-8 -*-
"""
test_answers.py — 用 test/answer.txt 的標準答案，逐張驗證辨識結果。

驗證項目:
  1. 分類 (classify)        : 一等水準點/測量讀數/尺規 -> 水準點/座標/埋深
  2. 尺規 (ruler)           : 有無垂直 / 有無平行 / 是否交叉
  3. 水準點 (benchmark)     : 4 位數字
  4. 座標 (coord)           : N / E / H_Z

用法:  python test_answers.py
"""
import os
import re
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import vision_pipelinev3resnet as vp

TEST_DIR = os.path.join(os.path.dirname(__file__), "test")
ANSWER = os.path.join(TEST_DIR, "answer.txt")

# 答案類別字串 -> 程式 class
CAT_MAP = {"一等水準點": "水準點", "測量讀數": "座標", "尺規": "埋深"}

GREEN = "[O]"
RED = "[X]"


def parse_answers():
    """回傳 list of dict: {file, cat, ...}"""
    items = []
    with open(ANSWER, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 檔名含空格，用副檔名定位
            m = re.match(r'(.+?\.(?:jpe|jpeg|jpg|png|bmp))\s+(.*)', line, re.IGNORECASE)
            if not m:
                continue
            fname, rest = m.group(1).strip(), m.group(2).strip()
            item = {"file": fname, "raw": rest}
            # 類別
            for cat in CAT_MAP:
                if rest.startswith(cat):
                    item["cat"] = cat
                    break
            # 尺規細節
            if item.get("cat") == "尺規":
                item["exp_v"] = "有垂直" in rest
                item["exp_h"] = "有平行" in rest
                item["exp_cross"] = "(交叉)" in rest or ("交叉" in rest and "無交叉" not in rest)
            # 水準點數字
            elif item.get("cat") == "一等水準點":
                mnum = re.search(r'(\d{4})', rest)
                item["exp_num"] = mnum.group(1) if mnum else None
            # 座標 NEZ
            elif item.get("cat") == "測量讀數":
                nums = re.findall(r'-?\d+\.\d+', rest)
                if len(nums) >= 3:
                    item["exp_n"], item["exp_e"], item["exp_z"] = nums[0], nums[1], nums[2]
            items.append(item)
    return items


def main():
    items = parse_answers()
    passed = total = 0

    # 實際檔名可能含前後空格，用「去空白」當鍵做對應
    disk_map = {f.strip(): f for f in os.listdir(TEST_DIR)}

    for it in items:
        actual = disk_map.get(it["file"].strip())
        print(f"\n=== {it['file']}  (答案: {it['raw']}) ===")
        if actual is None:
            print(f"  {RED} 檔案不存在")
            continue
        fp = os.path.join(TEST_DIR, actual)

        # 1. 分類
        cls_name, cls_conf = vp.classify_image(fp)
        exp_cls = CAT_MAP.get(it.get("cat"), "?")
        ok_cls = (cls_name == exp_cls)
        total += 1
        passed += ok_cls
        print(f"  分類: {cls_name} ({cls_conf})  期望={exp_cls}  {GREEN if ok_cls else RED}")
        if not ok_cls:
            # 分類錯就不往下測子流程
            continue

        # 2. 子流程
        if it["cat"] == "尺規":
            rr = vp.run_ruler(fp)
            d = rr.get("detections", {})
            got_v = d.get("vertical_count", 0) > 0
            got_h = d.get("horizontal_count", 0) > 0
            got_cross = d.get("is_crossed", False)
            ok_v = got_v == it["exp_v"]
            ok_h = got_h == it["exp_h"]
            ok_cross = got_cross == it["exp_cross"]
            for label, ok, got, exp in [
                ("垂直", ok_v, got_v, it["exp_v"]),
                ("平行", ok_h, got_h, it["exp_h"]),
                ("交叉", ok_cross, got_cross, it["exp_cross"]),
            ]:
                total += 1
                passed += ok
                print(f"    {label}: 抓到={got}  期望={exp}  {GREEN if ok else RED}")
            print(f"    (counts: V={d.get('vertical_count')}, H={d.get('horizontal_count')}, "
                  f"iou={d.get('best_iou')}, status={rr.get('status')})")

        elif it["cat"] == "一等水準點":
            br = vp.run_benchmark(fp)
            got = br.get("number", "")
            ok = (got == it.get("exp_num"))
            total += 1
            passed += ok
            print(f"    數字: 辨識={got} (conf={br.get('ocr_conf')})  期望={it.get('exp_num')}  "
                  f"{GREEN if ok else RED}")

        elif it["cat"] == "測量讀數":
            cr = vp.run_coord(fp)
            for label, key, exp_key in [("N", "N", "exp_n"), ("E", "E", "exp_e"), ("H_Z", "H_Z", "exp_z")]:
                got = cr.get(key)
                exp = it.get(exp_key)
                ok = (got == exp)
                total += 1
                passed += ok
                print(f"    {label}: 辨識={got}  期望={exp}  {GREEN if ok else RED}")
            print(f"    (coord_class={cr.get('coord_class')}, needs_review={cr.get('needs_review')})")

    print("\n" + "=" * 55)
    print(f"通過 {passed}/{total} 項")


if __name__ == "__main__":
    main()
