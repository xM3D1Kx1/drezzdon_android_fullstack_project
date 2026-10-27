import io
import os
import random
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFilter, ImageFont

app = FastAPI(title="Drezzdon Weirdcore CV Engine")

# Разрешаем запросы с сайта (PWA) и локального ПК
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SKY_TEXTS = [
    "reminder:\nwater!!",
    "are you still\nawake??",
    "don't forget\nto breathe!!",
    "sky looks\nfake today",
]

DISTANT_TEXTS = [
    "what do we have\nto lose??",
    "someone used\nto live here",
    "no exit\nfound??",
    "it feels so\nfamiliar...",
]

SUBJECT_TEXTS = [
    "still moving forward <3",
    "trying to stay real <3",
    "do not look back <3",
    "running out of time <3",
]

CONNECTOR_WORDS = ["interest", "focus", "impulse", "anomaly", "contact"]


def draw_dashed_line(draw, p1, p2, color=(255, 30, 30, 255), width=3, dash_len=12, gap_len=10):
    x1, y1 = p1
    x2, y2 = p2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    curr = 0
    while curr < dist:
        end = min(curr + dash_len, dist)
        sx, sy = x1 + dx * curr, y1 + dy * curr
        ex, ey = x1 + dx * end, y1 + dy * end
        draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
        curr += dash_len + gap_len


def draw_arrow(draw, p1, p2, color=(255, 30, 30, 255), width=3, head_size=12):
    draw.line([p1, p2], fill=color, width=width)
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    x2, y2 = p2
    left = (
        x2 - head_size * np.cos(angle - np.pi / 6),
        y2 - head_size * np.sin(angle - np.pi / 6),
    )
    right = (
        x2 - head_size * np.cos(angle + np.pi / 6),
        y2 - head_size * np.sin(angle + np.pi / 6),
    )
    draw.polygon([(x2, y2), left, right], fill=color)


def detect_scene_geometry(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Поиск главного силуэта/объекта в центральной зоне
    roi = gray[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    inv = 255 - roi
    blurred = cv2.GaussianBlur(inv, (15, 15), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx = int(w * 0.2) + max_loc[0]
    cy = int(h * 0.2) + max_loc[1]

    # 2. Поиск реальных линий перспективы в нижней половине кадра (HoughLinesP)
    edges = cv2.Canny(gray[int(h * 0.5):, :], 60, 160)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=int(w * 0.18), maxLineGap=25
    )
    perspective_line = ((int(w * 0.60), int(h * 0.60)), (int(w * 0.25), int(h * 0.98)))
    if lines is not None:
        best_len = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y1 += int(h * 0.5)
            y2 += int(h * 0.5)
            length = np.hypot(x2 - x1, y2 - y1)
            # Выбираем диагональную линию перспективы (не строго горизонтальную)
            if length > best_len and abs(y2 - y1) > h * 0.08:
                best_len = length
                perspective_line = ((x1, y1), (x2, y2))

    return (cx, cy), perspective_line


@app.post("/api/render-drezzdon")
async def render_drezzdon(file: UploadFile = File(...)):
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGBA")
    w, h = pil_img.size

    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    (cx, cy), persp_line = detect_scene_geometry(img_bgr)

    # Локальное размытие и затемнение внутри рамки головы (эффект силуэта drezzdon)
    bw = int(w * 0.085)
    hx = max(int(w * 0.2), min(int(w * 0.7), cx - bw // 2))
    hy = max(int(h * 0.2), min(int(h * 0.55), cy - bw))
    head_crop = pil_img.crop((hx, hy, hx + bw, hy + bw)).filter(ImageFilter.GaussianBlur(8))
    pil_img.paste(head_crop, (hx, hy))

    # Слой для красной графики с неоновым свечением
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    red = (255, 30, 30, 255)
    lw = max(3, int(w * 0.004))

    # 1. Рамка головы и две рамки рук/деталей
    hand_l = (hx - int(w * 0.03), hy + int(h * 0.17), int(w * 0.035), int(w * 0.035))
    hand_r = (hx + int(w * 0.10), hy + int(h * 0.20), int(w * 0.04), int(w * 0.04))

    draw.rectangle([hx, hy, hx + bw, hy + bw], outline=red, width=lw)
    draw.rectangle([hand_l[0], hand_l[1], hand_l[0] + hand_l[2], hand_l[1] + hand_l[3]], outline=red, width=lw)
    draw.rectangle([hand_r[0], hand_r[1], hand_r[0] + hand_r[2], hand_r[1] + hand_r[3]], outline=red, width=lw)

    # 2. Ортогональные стрелки 90° между рамками
    elbow1_x = hand_r[0] + hand_r[2] // 2
    start1_y = hy + int(bw * 0.4)
    draw.line([(hx + bw + 6, start1_y), (elbow1_x, start1_y)], fill=red, width=lw)
    draw_arrow(draw, (elbow1_x, start1_y), (elbow1_x, hand_r[1] - 5), color=red, width=lw)

    start2_x = hand_l[0] + hand_l[2] // 2
    elbow2_y = hand_r[1] + hand_r[3] // 2
    draw.line([(start2_x, hand_l[1] + hand_l[3] + 4), (start2_x, elbow2_y)], fill=red, width=lw)
    draw_arrow(draw, (start2_x, elbow2_y), (hand_r[0] - 5, elbow2_y), color=red, width=lw)

    # 3. Пунктирная линия вдоль найденной грани перспективы
    draw_dashed_line(draw, persp_line[0], persp_line[1], color=red, width=lw)

    # 4. Кластер связанных пунктиром мелких рамок на фоне
    c_pts = [
        (int(w * 0.50), int(h * 0.15)),
        (int(w * 0.55), int(h * 0.19)),
        (int(w * 0.60), int(h * 0.15)),
    ]
    s = int(w * 0.03)
    for px, py in c_pts:
        draw.rectangle([px - s // 2, py - s // 2, px + s // 2, py + s // 2], outline=red, width=lw - 1)
    draw_dashed_line(draw, c_pts[0], c_pts[1], color=red, width=lw - 1, dash_len=6, gap_len=6)
    draw_dashed_line(draw, c_pts[1], c_pts[2], color=red, width=lw - 1, dash_len=6, gap_len=6)

    # 5. Тексты и стрелки к небу и фону
    font_lg = ImageFont.load_default(size=max(18, int(w * 0.032)))
    font_sm = ImageFont.load_default(size=max(12, int(w * 0.018)))

    draw.text((int(w * 0.12), int(h * 0.04)), random.choice(SKY_TEXTS), fill=red, font=font_lg)
    draw_arrow(draw, (int(w * 0.28), int(h * 0.07)), (int(w * 0.35), int(h * 0.03)), color=red, width=lw)

    draw.text((int(w * 0.68), int(h * 0.05)), random.choice(DISTANT_TEXTS), fill=red, font=font_lg)
    draw_arrow(draw, (int(w * 0.85), int(h * 0.11)), (int(w * 0.87), int(h * 0.18)), color=red, width=lw)

    draw.text((max(10, hx - int(w * 0.36)), hy + bw + int(h * 0.02)), random.choice(SUBJECT_TEXTS), fill=red, font=font_lg)
    draw.text(((start2_x + hand_r[0]) // 2 - 22, elbow2_y - int(h * 0.025)), random.choice(CONNECTOR_WORDS), fill=red, font=font_sm)

    # Мягкое красное свечение (glow) вокруг всех линий и текста
    glow = overlay.filter(ImageFilter.GaussianBlur(4))
    result = Image.alpha_composite(pil_img, glow)
    result = Image.alpha_composite(result, overlay).convert("RGB")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
