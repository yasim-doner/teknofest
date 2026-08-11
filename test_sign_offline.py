#!/usr/bin/env python3
"""
Standalone sign detection test - no ROS 2 needed.
Doğrudan sign_detect.py'deki detection mantığını içerir.
Kullanım:
  python3 test_sign_offline.py                          # sentetik kare
  python3 test_sign_offline.py /path/to/screenshot.png  # gerçek görüntü
"""
import sys
import os
import cv2
import numpy as np
import glob
import math

TEXTURES_DIR = "/home/myazou/rover_ws/src/teknofest/textures"
TEMPLATE_MATCH_THRESHOLD = 0.25
MAX_STAGE_CANDIDATE_Y_RATIO = 0.85
USE_TEMPLATE_MATCHING = True

# ─────────────────────────────────────────────────────────────────────────────
# Şablon Yükleme
# ─────────────────────────────────────────────────────────────────────────────
def load_templates(textures_dir):
    templates = {}
    for filepath in glob.glob(os.path.join(textures_dir, "*.png")):
        key = os.path.basename(filepath).replace(".png", "")
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        h, w = gray.shape[:2]
        inner_gray = gray_norm[int(h * 0.12): int(h * 0.88), int(w * 0.12): int(w * 0.88)]
        inner_gray_norm = cv2.normalize(inner_gray, None, 0, 255, cv2.NORM_MINMAX)
        inner_edges = cv2.Canny(inner_gray_norm, 30, 120)
        _, inner_thresh = cv2.threshold(inner_gray_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        stage_num = None
        is_stop = (key == "sign_stop")
        if key.startswith("sign_") and not is_stop:
            num_part = key.replace("sign_", "").replace("_exit", "")
            if num_part.isdigit():
                stage_num = int(num_part)

        templates[key] = {
            "key": key,
            "gray": gray_norm,
            "inner_gray": inner_gray_norm,
            "inner_edges": inner_edges,
            "thresh": inner_thresh,
            "stage_num": stage_num,
        }
    print(f"[INFO] {len(templates)} şablon yüklendi: {list(templates.keys())}")
    return templates


# ─────────────────────────────────────────────────────────────────────────────
# Kırmızı Maske
# ─────────────────────────────────────────────────────────────────────────────
def make_red_mask(hsv):
    mask1 = cv2.inRange(hsv, np.array([0, 55, 45], dtype=np.uint8), np.array([16, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([160, 55, 45], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(mask1, mask2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k, iterations=2)
    return red


# ─────────────────────────────────────────────────────────────────────────────
# Template Eşleme
# ─────────────────────────────────────────────────────────────────────────────
def match_template(frame, box, templates):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 6 or rw < 6:
        return None, 0.0, None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray_norm = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
    inner = roi_gray_norm[int(rh * 0.12): int(rh * 0.88), int(rw * 0.12): int(rw * 0.88)]
    if inner.size == 0 or inner.shape[0] < 4 or inner.shape[1] < 4:
        return None, 0.0, None

    inner_norm = cv2.normalize(inner, None, 0, 255, cv2.NORM_MINMAX)
    inner_edges = cv2.Canny(inner_norm, 30, 120)

    roi_std = float(np.std(inner_norm))
    if roi_std > 5.0:
        _, roi_thresh = cv2.threshold(inner_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        roi_thresh = None

    best_key, best_score, best_tmpl = None, -1.0, None
    for key, tmpl in templates.items():
        t_gray = cv2.resize(tmpl["gray"], (rw, rh))
        res_gray = max(0.0, float(np.max(cv2.matchTemplate(roi_gray_norm, t_gray, cv2.TM_CCOEFF_NORMED))))

        ih, iw = inner_norm.shape[:2]
        t_inner = cv2.resize(tmpl["inner_gray"], (iw, ih))
        res_inner = max(0.0, float(np.max(cv2.matchTemplate(inner_norm, t_inner, cv2.TM_CCOEFF_NORMED))))

        t_edges = cv2.resize(tmpl["inner_edges"], (iw, ih))
        res_edge = max(0.0, float(np.max(cv2.matchTemplate(inner_edges, t_edges, cv2.TM_CCOEFF_NORMED))))

        if roi_thresh is not None:
            t_thresh = cv2.resize(tmpl["thresh"], (iw, ih))
            if float(np.std(roi_thresh)) > 0:
                res_thresh = max(0.0, float(np.max(cv2.matchTemplate(roi_thresh, t_thresh, cv2.TM_CCOEFF_NORMED))))
            else:
                res_thresh = res_inner
        else:
            res_thresh = res_inner

        score = 0.30 * res_gray + 0.40 * res_inner + 0.30 * max(res_edge, res_thresh)
        if score > best_score:
            best_score, best_key, best_tmpl = score, key, tmpl

    return best_key, best_score, best_tmpl


# ─────────────────────────────────────────────────────────────────────────────
# Aday Tespiti (Çift Strateji)
# ─────────────────────────────────────────────────────────────────────────────
def find_candidates(frame, hsv, red_mask, templates):
    height, width = frame.shape[:2]
    raw_candidates = []

    # Sadece tabela içindeki SAF BEYAZ diski bul (gök, zemin renkleri elenir)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 160], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8))
    kw = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_clean = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kw, iterations=2)
    w_contours, _ = cv2.findContours(white_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for wc in w_contours:
        wx, wy, ww, wh = cv2.boundingRect(wc)
        # En az 8px, en fazla frame'in %30'u kadar olmalı
        if ww < 8 or wh < 8 or ww > width * 0.30 or wh > height * 0.30:
            continue
        if not (0.40 <= ww / wh <= 2.5):
            continue

        # Şablon eşleme için beyaz iç disk + küçük margin (kırmızı halka için)
        small_margin = max(2, int(max(ww, wh) * 0.20))
        sx1, sy1 = max(0, wx - small_margin), max(0, wy - small_margin)
        sx2, sy2 = min(width, wx + ww + small_margin), min(height, wy + wh + small_margin)

        # Kırmızı halka kontrolü için daha geniş alan
        ring_margin = max(4, int(max(ww, wh) * 0.55))
        rx1, ry1 = max(0, wx - ring_margin), max(0, wy - ring_margin)
        rx2, ry2 = min(width, wx + ww + ring_margin), min(height, wy + wh + ring_margin)

        if np.count_nonzero(red_mask[ry1:ry2, rx1:rx2]) < 12:
            continue

        cx, cy = wx + ww // 2, wy + wh // 2
        radius = max(sx2 - sx1, sy2 - sy1) // 2

        # Şablon eşleşmesini sıkı ROI üzerinde yap
        best_key, match_score, tmpl_meta = match_template(frame, (sx1, sy1, sx2, sy2), templates)

        # Bulunan tabela için gerçek (daha büyük) bbox'ı göster
        disp_box = (rx1, ry1, rx2, ry2)

        if match_score >= 0.10:
            raw_candidates.append({
                "cx": cx, "cy": cy, "radius": radius, "box": disp_box,
                "white_ratio": 0.5, "red_ratio": 0.3,
                "score": 60.0 + match_score * 60.0,
                "matched_key": best_key, "match_score": match_score, "tmpl_meta": tmpl_meta,
                "strategy": "white_disc",
            })

    # STRATEJİ 2: Dış Kırmızı Kontur → Normal gri/beyaz zemin tabelaları
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kc, iterations=2)
    r_contours, _ = cv2.findContours(red_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in r_contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 10 or bh < 10 or bw > width * 0.60 or bh > height * 0.60:
            continue
        if not (0.40 <= bw / bh <= 2.5):
            continue

        cx, cy = x + bw // 2, y + bh // 2
        if cy > int(height * MAX_STAGE_CANDIDATE_Y_RATIO):
            continue

        radius = max(bw, bh) // 2
        expand = max(2, int(radius * 0.10))
        x1, y1 = max(0, x - expand), max(0, y - expand)
        x2, y2 = min(width, x + bw + expand), min(height, y + bh + expand)
        if x2 <= x1 or y2 <= y1:
            continue

        hsv_roi = hsv[y1:y2, x1:x2]
        red_roi = red_mask[y1:y2, x1:x2]
        w_mask = (hsv_roi[:, :, 1] < 90) & (hsv_roi[:, :, 2] > 100)
        white_ratio = float(np.count_nonzero(w_mask) / w_mask.size)
        red_fill = float(np.count_nonzero(red_roi) / red_roi.size)
        if white_ratio < 0.02 or red_fill < 0.01:
            continue

        best_key, match_score, tmpl_meta = match_template(frame, (x1, y1, x2, y2), templates)
        score = radius + 20.0 * white_ratio + 10.0 * red_fill + 30.0 * max(0.0, match_score)

        raw_candidates.append({
            "cx": cx, "cy": cy, "radius": radius, "box": (x1, y1, x2, y2),
            "white_ratio": white_ratio, "red_ratio": red_fill, "score": score,
            "matched_key": best_key, "match_score": match_score, "tmpl_meta": tmpl_meta,
            "strategy": "red_contour",
        })

    # NMS
    raw_candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = []
    for cand in raw_candidates:
        keep = True
        for ex in candidates:
            if math.hypot(cand["cx"] - ex["cx"], cand["cy"] - ex["cy"]) < max(cand["radius"], ex["radius"]) * 0.85:
                keep = False
                break
        if keep:
            candidates.append(cand)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Sentetik Test Karesi  
# ─────────────────────────────────────────────────────────────────────────────
def create_test_frame():
    h, w = 480, 640
    frame = np.ones((h, w, 3), dtype=np.uint8) * 170

    # Gök
    frame[:200, :] = np.array([230, 200, 180], dtype=np.uint8)

    # Sol kırmızı duvar
    cv2.rectangle(frame, (0, 170), (320, 480), (20, 20, 185), -1)

    # Sarı rampa
    pts = np.array([[200, 310], [500, 240], [500, 268], [200, 338]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 215, 255))

    # Taşlar (kahverengi)
    for (px, py, pr) in [(410, 325, 8), (235, 365, 10), (305, 425, 12), (515, 435, 9)]:
        cv2.circle(frame, (px, py), pr, (55, 70, 110), -1)

    # sign_3 tabelası — kırmızı duvarın üstüne
    sign_path = os.path.join(TEXTURES_DIR, "sign_3.png")
    if os.path.exists(sign_path):
        img = cv2.imread(sign_path, cv2.IMREAD_COLOR)
        if img is not None:
            img_r = cv2.resize(img, (72, 72))
            frame[100:172, 78:150] = img_r
    else:
        cv2.circle(frame, (115, 136), 36, (0, 0, 210), 8)
        cv2.circle(frame, (115, 136), 27, (255, 255, 255), -1)
        cv2.putText(frame, "3", (101, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None

    if img_path and os.path.exists(img_path):
        print(f"[TEST] Gerçek görüntü yükleniyor: {img_path}")
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        print("[TEST] Sentetik test karesi oluşturuluyor (Kırmızı duvar + Sign_3 + Taşlar)...")
        frame = create_test_frame()

    templates = load_templates(TEXTURES_DIR)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = make_red_mask(hsv)
    candidates = find_candidates(frame, hsv, red_mask, templates)

    print(f"\n{'='*60}")
    print(f"  SONUÇ: {len(candidates)} ADAY TESPİT EDİLDİ")
    print(f"{'='*60}")
    for i, c in enumerate(candidates):
        mk = c.get("matched_key", "-")
        ms = c.get("match_score", 0.0)
        stat = "✓ MATCH" if ms >= TEMPLATE_MATCH_THRESHOLD else "~ UNMATCHED"
        print(f"  [#{i+1}] Strateji={c['strategy']:12s}  Merkez=({c['cx']:3d},{c['cy']:3d})  "
              f"r={c['radius']:3d}px  {stat}  → '{mk}' ({ms:.2f})")

    # Debug görüntüsü çiz ve kaydet
    debug = frame.copy()
    for c in candidates:
        x1, y1, x2, y2 = c["box"]
        ms = c.get("match_score", 0.0)
        mk = c.get("matched_key", "?")
        color = (0, 255, 0) if ms >= TEMPLATE_MATCH_THRESHOLD else (0, 200, 255)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.circle(debug, (c["cx"], c["cy"]), 4, (0, 0, 255), -1)
        cv2.putText(debug, f"{mk} ({ms:.2f})", (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Red mask overlay (solda ince şerit)
    red_vis = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    red_vis[:, :, 1:] = 0  # Sadece kırmızı kanal

    out_path = "/home/myazou/rover_ws/src/teknofest/offline_test_result.png"
    cv2.imwrite(out_path, debug)

    red_out = "/home/myazou/rover_ws/src/teknofest/offline_test_redmask.png"
    cv2.imwrite(red_out, red_mask)

    print(f"\n[KAYDEDILDI] Debug görüntüsü: {out_path}")
    print(f"[KAYDEDILDI] Kırmızı maske:  {red_out}")


if __name__ == "__main__":
    main()
