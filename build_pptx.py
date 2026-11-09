"""
build_pptx.py — сборка презентации кейса «Спутникс» (.pptx, python-pptx).

Дизайн: тёмная космическая тема, синий акцент, на каждом слайде —
реальные артефакты проекта (маски, скриншоты, чипы-цифры), футер
с нумерацией. Метрики подтягиваются из outputs/metrics/metrics.json.

Запуск:  python build_pptx.py
Результат: presentation/Sputnix_cloud_detection.pptx
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

BG = RGBColor(0x0B, 0x0F, 0x1A)
CARD = RGBColor(0x14, 0x1A, 0x26)
CARD2 = RGBColor(0x0E, 0x14, 0x20)
ACCENT = RGBColor(0x4D, 0x8D, 0xFF)
ACCENT2 = RGBColor(0x00, 0xE5, 0xC0)
WHITE = RGBColor(0xF2, 0xF5, 0xFA)
GRAY = RGBColor(0x8A, 0x94, 0xA6)
MUTED = GRAY
DARK = RGBColor(0x08, 0x10, 0x20)
RED = RGBColor(0xFF, 0x5C, 0x5C)
GOOD = RGBColor(0x35, 0xC7, 0x7B)
LINE = RGBColor(0x23, 0x2C, 0x3D)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
ASSETS = Path("assets")
SLIDES_ASSETS = ASSETS / "slides"


def load_metrics():
    p = Path("outputs/metrics/metrics.json")
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def load_train_summary():
    p = Path("outputs/logs/train_summary.json")
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def add_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(1, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return slide


def text(slide, x, y, w, h, s, size=18, color=WHITE, bold=False,
         align=PP_ALIGN.LEFT, spacing=1.15, font="Bahnschrift SemiBold"):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = spacing
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.bold = bold
            run.font.name = font
    return box


def title(slide, s, sub=None):
    text(slide, Inches(0.55), Inches(0.32), Inches(11), Inches(0.9),
         s, size=30, color=WHITE, bold=True)
    bar = slide.shapes.add_shape(1, Inches(0.6), Inches(1.05),
                                 Inches(2.6), Pt(3.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    if sub:
        text(slide, Inches(0.6), Inches(1.18), Inches(12), Inches(0.5),
             sub, size=14, color=GRAY)


def bullets(slide, x, y, w, h, items, size=15, color=WHITE):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.2
        r1 = p.add_run()
        r1.text = "—  "
        r1.font.color.rgb = ACCENT2
        r1.font.bold = True
        r1.font.size = Pt(size)
        r1.font.name = "Bahnschrift"
        r2 = p.add_run()
        r2.text = b
        r2.font.color.rgb = color
        r2.font.size = Pt(size)
        r2.font.name = "Bahnschrift"
    return box


def card(slide, x, y, w, h, fill=CARD, border=LINE, radius=0.12):
    c = slide.shapes.add_shape(5, x, y, w, h)
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.color.rgb = border
    c.line.width = Pt(1)
    c.shadow.inherit = False
    try:
        c.adjustments[0] = radius
    except Exception:
        pass
    return c


def chip(slide, x, y, w, big, small, color=ACCENT):
    card(slide, x, y, w, Inches(1.05), fill=DARK, border=color)
    text(slide, x + Inches(0.18), y + Inches(0.1), w - Inches(0.3),
         Inches(0.55), big, size=22, color=color, bold=True)
    text(slide, x + Inches(0.18), y + Inches(0.62), w - Inches(0.3),
         Inches(0.4), small, size=11.5, color=GRAY)


def image_card(slide, x, y, w, img_path, caption=None, h_img=None):
    from PIL import Image
    iw, ih = Image.open(img_path).size
    h = h_img or (w * ih / iw)
    pad = Inches(0.08)
    card_h = h + Inches(0.45)
    card(slide, x, y, w, card_h, fill=CARD)
    slide.shapes.add_picture(str(img_path), x + pad, y + pad,
                             width=w - pad * 2)
    if caption:
        text(slide, x + Inches(0.14), y + h + Inches(0.12), w - Inches(0.28),
             Inches(0.3), caption, size=11.5, color=MUTED)
    return card_h


def footer(slide, num):
    line = slide.shapes.add_shape(1, Inches(0.55), Inches(7.14),
                                  Inches(12.23), Pt(0.75))
    line.fill.solid()
    line.fill.fore_color.rgb = LINE
    line.line.fill.background()
    line.shadow.inherit = False
    text(slide, Inches(0.55), Inches(7.17), Inches(6), Inches(0.3),
         "Спутникс · Распознавание облачности · SK KIDS CHALLENGE",
         size=9.5, color=RGBColor(0x5A, 0x64, 0x78))
    text(slide, Inches(12.3), Inches(7.17), Inches(0.5), Inches(0.3),
         f"{num:02d}", size=10, color=GRAY, align=PP_ALIGN.RIGHT)


def slide_title(prs, meta):
    s = add_slide(prs)
    lockup = ASSETS / "sputnix_lockup_white.png"
    if lockup.exists():
        s.shapes.add_picture(str(lockup), Inches(4.42), Inches(0.7),
                             height=Inches(0.62))
    text(s, Inches(0.8), Inches(2.0), Inches(11.7), Inches(0.5),
         "SK KIDS CHALLENGE • Кейс компании «Спутникс» • 14+", size=15,
         color=GRAY, align=PP_ALIGN.CENTER)
    text(s, Inches(0.8), Inches(2.6), Inches(11.7), Inches(1.8),
         "Алгоритм ИИ для распознавания облачности\nна спутниковых снимках",
         size=37, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    text(s, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5),
         "U-Net • порог отбраковки 75% • IoU 0.904",
         size=17, color=ACCENT2, align=PP_ALIGN.CENTER)
    frame = card(s, Inches(4.57), Inches(5.3), Inches(4.2), Inches(1.15),
                 fill=CARD)
    text(s, Inches(4.72), Inches(5.42), Inches(3.9), Inches(0.45),
         "Автор: Белиевский Давид, 16 лет", size=14, color=WHITE,
         align=PP_ALIGN.CENTER, bold=True)
    text(s, Inches(4.72), Inches(5.88), Inches(3.9), Inches(0.4),
         "solo • PyTorch • Python 3.10+", size=11.5, color=GRAY,
         align=PP_ALIGN.CENTER)


def slide_problem(prs, meta):
    s = add_slide(prs)
    title(s, "Проблема и цель")
    text(s, Inches(0.7), Inches(1.6), Inches(6.2), Inches(0.4),
         "Проблема", size=17, color=RED, bold=True)
    bullets(s, Inches(0.7), Inches(2.1), Inches(6.2), Inches(2.6), [
        "Кубсаты «Спутникс» делают тысячи кадров — значительная часть "
        "закрыта облаками и бесполезна",
        "Разглядывать вручную невозможно: нужен автоматический фильтр",
        "По ТЗ: кадры закрыты более чем на 75% — отбраковываются",
    ], size=14.5)
    text(s, Inches(0.7), Inches(4.6), Inches(6.2), Inches(0.4),
         "Цель", size=17, color=ACCENT2, bold=True)
    bullets(s, Inches(0.7), Inches(5.1), Inches(6.2), Inches(1.6), [
        "Считать процент облачности по маске",
        "Отбраковывать и отчитываться оператору",
    ], size=14.5)
    image_card(s, Inches(7.4), Inches(1.55), Inches(5.35),
               SLIDES_ASSETS / "cloudy.jpg",
               "Реальный кадр кубсата: облачность 72.4% — оператору почти "
               "нечего смотреть")
    footer(s, 2)


def slide_data(prs, meta):
    s = add_slide(prs)
    title(s, "Входные данные")
    x, y = Inches(0.6), Inches(1.5)
    items = [
        ("Снимки «Спутникс» — из ТЗ",
         "4 213 кадров с камер кубсатов SITRO-AIS: 3280×2464, RGB, "
         "~200 м/пиксель + геопривязка KML. Использованы для инференса.",
         SLIDES_ASSETS / "raw_frame.jpg"),
        ("38-Cloud — обучение",
         "Sentinel-2: 8 400 патчей 384×384 с масками облаков. В наборе "
         "«Спутникс» масок нет — обучаем здесь (transfer learning).",
         SLIDES_ASSETS / "s2_patch.png"),
        ("Почему нужна нейросеть",
         "Эвристика «яркое = облако» путает облака со снегом, льдом, "
         "дымкой. Сеть учится текстуре и контексту.",
         SLIDES_ASSETS / "clear.jpg"),
        ("Резерв: синтетика",
         "Свой генератор: кадры с точными масками для smoke-теста "
         "пайплайна и запасного демо без интернета.",
         SLIDES_ASSETS / "synth.png"),
    ]
    pos = [(Inches(0.6), Inches(1.55)), (Inches(6.85), Inches(1.55)),
           (Inches(0.6), Inches(4.2)), (Inches(6.85), Inches(4.2))]
    for (t, b, img), (cx, cy) in zip(items, pos):
        card(s, cx, cy, Inches(5.9), Inches(2.45), fill=CARD)
        s.shapes.add_picture(str(img), cx + Inches(0.14), cy + Inches(0.14),
                             height=Inches(2.17))
        text(s, cx + Inches(2.45), cy + Inches(0.16), Inches(3.3),
             Inches(0.4), t, size=14.5, color=ACCENT2, bold=True)
        text(s, cx + Inches(2.45), cy + Inches(0.62), Inches(3.3),
             Inches(1.7), b, size=11.5, color=WHITE)
    footer(s, 3)


def slide_arch(prs, meta):
    s = add_slide(prs)
    title(s, "Архитектура алгоритма",
          "U-Net: сжимает кадр, извлекая признаки, и восстанавливает маску "
          "в исходном разрешении")
    boxes = [
        ("Снимок\n3×384×384", ACCENT),
        ("Энкодер ResNet18\nпризнаки", ACCENT),
        ("Бутылочное\nгорлышко", ACCENT2),
        ("Декодер\nвосстановление", ACCENT),
        ("Маска облаков\n1×384×384", RED),
    ]
    x, y = Inches(0.7), Inches(2.1)
    w, h, gap = Inches(2.3), Inches(1.45), Inches(0.28)
    for i, (label, c) in enumerate(boxes):
        box = card(s, x, y, w, h, fill=DARK, border=c)
        text(s, x + Inches(0.1), y + Inches(0.32), w - Inches(0.2),
             Inches(0.9), label, size=13.5, color=WHITE, bold=True,
             align=PP_ALIGN.CENTER)
        if i < 4:
            arrow = s.shapes.add_shape(13, x + w + Inches(0.04),
                                       y + h / 2 - Inches(0.14),
                                       gap - Inches(0.08), Inches(0.28))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = c
            arrow.line.fill.background()
            arrow.shadow.inherit = False
        x = x + w + gap
    card(s, Inches(0.7), Inches(4.0), Inches(11.9), Inches(0.02), fill=LINE)
    bullets(s, Inches(0.7), Inches(4.35), Inches(6.0), Inches(2.2), [
        "13.7 млн параметров, вход RGB 384×384",
        "Выход — вероятность облака в каждом пикселе (сигмоида)",
        "Лосс BCE + Dice, оптимизатор AdamW, точность смешанная (AMP)",
    ], size=13.5)
    bullets(s, Inches(7.0), Inches(4.35), Inches(5.6), Inches(2.2), [
        "Большие кадры режутся на тайлы 512×512 с перекрытием",
        "Fallback: собственная SmallUNet — работает без smp",
        "Аугментации: флипы, повороты, яркость — устойчивость "
        "к дымке и разным камерам",
    ], size=13.5)
    footer(s, 4)


def slide_pipeline(prs, meta):
    s = add_slide(prs)
    title(s, "Пайплайн обработки снимка")
    steps = [
        ("1 · Загрузка", "JPG/TIF/PNG,\nучёт NODATA-пикселей"),
        ("2 · Сегментация", "U-Net строит маску;\nкадры — тайлами 512²"),
        ("3 · Подсчёт %", "облачные пиксели /\nвалидные × 100"),
        ("4 · Порог 75%", "> 75% — отбраковка\n≤ 75% — сохранение"),
        ("5 · Отчёт", "консоль, CSV, JSON,\nсообщение оператору"),
    ]
    x, y, w, h, gap = Inches(0.55), Inches(1.8), Inches(2.35), Inches(1.9), \
        Inches(0.15)
    for i, (t, b) in enumerate(steps):
        c = RED if i == 3 else ACCENT2
        card(s, x, y, w, h, fill=DARK, border=c)
        text(s, x + Inches(0.14), y + Inches(0.14), w - Inches(0.28),
             Inches(0.4), t, size=15, color=c, bold=True)
        text(s, x + Inches(0.14), y + Inches(0.62), w - Inches(0.28),
             Inches(1.2), b, size=12, color=WHITE)
        x = x + w + gap
    image_card(s, Inches(0.55), Inches(4.05), Inches(3.6),
               SLIDES_ASSETS / "rejected.jpg",
               "кадр 77.4% — отбракован")
    image_card(s, Inches(4.45), Inches(4.05), Inches(3.6),
               SLIDES_ASSETS / "clear.jpg",
               "кадр 2.2% — сохранён")
    card(s, Inches(8.35), Inches(4.05), Inches(4.4), Inches(2.55),
         fill=DARK, border=ACCENT)
    text(s, Inches(8.55), Inches(4.2), Inches(4.0), Inches(0.4),
         "Сообщения оператору — ТЗ", size=13, color=ACCENT2, bold=True)
    text(s, Inches(8.55), Inches(4.62), Inches(4.0), Inches(1.9),
         "Сохранён: кадр — облачность 2.2%.\n"
         "Удалён: кадр — облачность 77.4%.\n\n"
         "Полные отчёты: report.csv, report.json,\noperator_report.txt",
         size=13, color=WHITE)
    footer(s, 5)


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def slide_metrics(prs, meta, train):
    s = add_slide(prs)
    title(s, "Обучение и метрики")
    mm = (meta or {}).get("mask_metrics", {})
    mg = (meta or {}).get("mask_metrics_global", {})
    conf = (meta or {}).get("threshold_75_confusion", {})
    ts = train or {}
    chip(s, Inches(0.7), Inches(1.6), Inches(2.9),
         fmt(mg.get("IoU")), "IoU глобальный (маска)", ACCENT)
    chip(s, Inches(3.75), Inches(1.6), Inches(2.9),
         fmt(mm.get("Dice")), "Dice / F1 маски", ACCENT)
    chip(s, Inches(6.8), Inches(1.6), Inches(2.9),
         f"{meta.get('cloud_pct_MAE')} п.п." if meta else "—",
         "ошибка процента облачности", ACCENT2)
    chip(s, Inches(9.85), Inches(1.6), Inches(2.9),
         f"{conf.get('decision_accuracy', 0) * 100:.1f}%",
         "точность решения по порогу", GOOD)
    text(s, Inches(0.7), Inches(3.0), Inches(6), Inches(0.4),
         "Матрица ошибок решения (820 кадров, порог >75%):", size=15,
         color=WHITE, bold=True)
    cells = [
        ("169", "TP — верно отбраковано", GOOD, Inches(0.7), Inches(3.5)),
        ("17", "FP — зря отбраковано", RED, Inches(3.9), Inches(3.5)),
        ("9", "FN — пропущен брак", RED, Inches(0.7), Inches(4.75)),
        ("625", "TN — верно сохранено", GOOD, Inches(3.9), Inches(4.75)),
    ]
    for val, label, c, cx, cy in cells:
        card(s, cx, cy, Inches(3.0), Inches(1.1), fill=DARK, border=c)
        text(s, cx + Inches(0.18), cy + Inches(0.08), Inches(2.7),
             Inches(0.5), val, size=22, color=WHITE, bold=True)
        text(s, cx + Inches(0.18), cy + Inches(0.66), Inches(2.7),
             Inches(0.4), label, size=11.5, color=GRAY)
    image_card(s, Inches(7.35), Inches(3.35), Inches(3.1),
               SLIDES_ASSETS / "rejected.jpg", "маска отбракованного кадра")
    image_card(s, Inches(10.55), Inches(3.35), Inches(2.2),
               SLIDES_ASSETS / "clear.jpg", "маска ясного кадра")
    info = [
        f"Обучение: {ts.get('epochs_note', '—')} • "
        f"{ts.get('total_time_min', '—')} мин на RTX 4060 Ti",
        f"Данные: train {ts.get('train_pairs', '—')} пар / "
        f"val {ts.get('val_pairs', '—')} • батч {ts.get('batch_size', '—')} "
        f"• {ts.get('img_size', '—')}px",
    ]
    bullets(s, Inches(7.35), Inches(6.3), Inches(5.4), Inches(0.8), info,
            size=12, color=GRAY)
    footer(s, 6)


def slide_threshold(prs, meta):
    s = add_slide(prs)
    title(s, "Логика порога 75% и отбраковки",
          "Требование ТЗ: если облака занимают больше 75% кадра — снимок "
          "не имеет ценности и удаляется")
    image_card(s, Inches(0.7), Inches(1.85), Inches(3.45),
               SLIDES_ASSETS / "rejected.jpg",
               "Удалён: облачность 77.4% (> 75)")
    image_card(s, Inches(4.4), Inches(1.85), Inches(3.45),
               SLIDES_ASSETS / "clear.jpg",
               "Сохранён: облачность 2.2% (≤ 75)")
    card(s, Inches(9.0), Inches(1.9), Inches(3.75), Inches(3.1),
         fill=DARK, border=ACCENT2)
    text(s, Inches(9.2), Inches(2.05), Inches(3.4), Inches(0.4),
         "ОПЕРАТОРУ (реальный вывод)", size=12, color=ACCENT2, bold=True)
    rep = Path("outputs/results/report.json")
    lines = []
    if rep.exists():
        rep = json.load(open(rep, encoding="utf-8"))
        saved = [r for r in rep["files"] if r["decision"] == "accepted"][:2]
        rej = [r for r in rep["files"] if r["decision"] == "rejected"][:2]
        for r in saved + rej:
            verb = "Сохранён" if r["decision"] == "accepted" else "Удалён"
            lines.append(f"{verb}: {r['filename'][:28]} — {r['cloud_pct']}%.")
    for line in lines or ["(запустите predict.py)"]:
        text(s, Inches(9.2), Inches(2.5) + Inches(0.42 * lines.index(line)),
             Inches(3.4), Inches(0.4), line, size=12, color=WHITE,
             font="Consolas")
    bullets(s, Inches(0.7), Inches(6.05), Inches(11.9), Inches(1.1), [
        "> 75% — перемещение в rejected/ (безопасно) • ≤ 75% — копия в "
        "accepted/ • --hard-delete — удаление по букве ТЗ",
        "Отчёты: report.csv • report.json • operator_report.txt — "
        "оператору не нужно ничего искать",
    ], size=14)
    footer(s, 7)


def slide_demo(prs, meta):
    s = add_slide(prs)
    title(s, "Демонстрация работы",
          "маска наложена на снимок, подписан процент, решение по порогу")
    vis = Path("outputs/results/visualizations")
    pick = ["2026-01-31", "2026-04-16", "2025-11-03"]
    imgs = []
    for key in pick:
        for p in sorted(vis.glob("*.jpg")):
            if key in p.name and p not in imgs:
                imgs.append(p)
                break
    caps = ["отбракован 77.4%", "облачный океан 72.4%", "ясная река 2.2%"]
    x = Inches(0.6)
    for img, cap in zip(imgs[:3], caps):
        image_card(s, x, Inches(1.65), Inches(4.0), img, cap)
        x = Inches(x.inches_ if False else 0) if False else x + Inches(4.15)
    bullets(s, Inches(0.7), Inches(6.35), Inches(12), Inches(0.9), [
        "Бирюзовым выделена распознанная облачность, красным — её граница",
        "Файлы с облачностью выше 75% уходят в rejected/, остальные — "
        "в accepted/",
    ], size=14)
    footer(s, 8)


def slide_gui(prs, meta):
    s = add_slide(prs)
    title(s, "Графический интерфейс оператора")
    shots = Path("outputs/gui_shots")
    done, idle = shots / "gui_done.png", shots / "gui_idle.png"
    if done.exists():
        s.shapes.add_picture(str(done), Inches(0.55), Inches(1.45),
                             width=Inches(7.4))
    if idle.exists():
        s.shapes.add_picture(str(idle), Inches(8.2), Inches(1.45),
                             width=Inches(4.55))
        text(s, Inches(8.2), Inches(4.2), Inches(4.55), Inches(0.5),
             "экран запуска: выбор папки, порог, статус системы",
             size=12, color=GRAY)
    bullets(s, Inches(0.7), Inches(6.35), Inches(12), Inches(1.0), [
        "После обработки 26 снимков: лог оператора, список результатов "
        "с процентами, превью маски облаков",
        "Порог отбраковки двигается ползунком; брак уходит в rejected/",
    ], size=13, color=GRAY)
    footer(s, 9)


def slide_limits(prs, meta):
    s = add_slide(prs)
    title(s, "Ограничения и развитие")
    card(s, Inches(0.7), Inches(1.6), Inches(5.9), Inches(4.9), fill=CARD)
    text(s, Inches(0.95), Inches(1.8), Inches(5.4), Inches(0.45),
         "Ограничения", size=17, color=RED, bold=True)
    bullets(s, Inches(0.95), Inches(2.4), Inches(5.4), Inches(3.9), [
        "Domain gap: обучение на Sentinel-2, инференс на камерах "
        "кубсатов — цвета и разрешение различаются",
        "Тонкие перистые облака и дымка — самые сложные классы",
        "Снег и лёд иногда распознаются как облака",
        "Граница ровно 75% — зона шума метрики",
    ], size=13.5)
    card(s, Inches(6.9), Inches(1.6), Inches(5.8), Inches(4.9), fill=CARD)
    text(s, Inches(7.15), Inches(1.8), Inches(5.3), Inches(0.45),
         "Как развивать", size=17, color=ACCENT2, bold=True)
    bullets(s, Inches(7.15), Inches(2.4), Inches(5.3), Inches(3.9), [
        "Дообучение (fine-tune) на вручную размеченных снимках «Спутникс»",
        "KML-телеметрия: широта кадра → климатология облачности",
        "Более сильные модели: DeepLabV3+, SegFormer",
        "Полуавтоматическая доразметка: модель предлагает, человек правит",
    ], size=13.5)
    footer(s, 10)


def slide_conclusions(prs, meta):
    s = add_slide(prs)
    title(s, "Выводы")
    mm = (meta or {}).get("mask_metrics", {}) if meta else {}
    conf = (meta or {}).get("threshold_75_confusion", {}) if meta else {}
    chip(s, Inches(0.7), Inches(1.7), Inches(3.9),
         "IoU " + fmt(mm.get("IoU")) if mm else "IoU —", "качество маски",
         ACCENT)
    chip(s, Inches(4.75), Inches(1.7), Inches(3.9),
         f"{conf.get('decision_accuracy', 0) * 100:.1f}%" if conf else "—",
         "точность решения 75%", GOOD)
    chip(s, Inches(8.8), Inches(1.7), Inches(3.9), "26", "кадров в демо",
         ACCENT2)
    bullets(s, Inches(0.7), Inches(3.3), Inches(11.9), Inches(3.2), [
        "Полный пайплайн собран и работает: данные → обучение → метрики → "
        "инференс → отчёт оператору",
        "Маска облаков: IoU 0.904 (глобальный) / 0.876 (средний), "
        "Dice 0.904, ошибка процента — 2.26 п.п.",
        "Все требования ТЗ выполнены: процент облачности, порог 75%, "
        "отбраковка, отчёт с сохранёнными и удалёнными файлами",
        "Проект открыт к развитию: дообучение на данных «Спутникс», "
        "SegFormer, доразметка",
    ], size=16)
    footer(s, 11)


def slide_thanks(prs, meta):
    s = add_slide(prs)
    lockup = ASSETS / "sputnix_lockup_white.png"
    if lockup.exists():
        s.shapes.add_picture(str(lockup), Inches(4.42), Inches(1.9),
                             height=Inches(0.62))
    text(s, Inches(0.8), Inches(2.9), Inches(11.7), Inches(1.0),
         "Спасибо за внимание!", size=40, color=WHITE, bold=True,
         align=PP_ALIGN.CENTER)
    text(s, Inches(0.8), Inches(4.1), Inches(11.7), Inches(0.5),
         "Живая демонстрация: predict.py на снимках «Спутникс»",
         size=17, color=ACCENT2, align=PP_ALIGN.CENTER)
    text(s, Inches(0.8), Inches(4.9), Inches(11.7), Inches(0.5),
         "github.com/Kot3loy/sputnix-cloud-detection",
         size=14, color=GRAY, align=PP_ALIGN.CENTER)


def main():
    meta = load_metrics()
    train = load_train_summary()
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    slide_title(prs, meta)
    slide_problem(prs, meta)
    slide_data(prs, meta)
    slide_arch(prs, meta)
    slide_pipeline(prs, meta)
    slide_metrics(prs, meta, train)
    slide_threshold(prs, meta)
    slide_demo(prs, meta)
    slide_gui(prs, meta)
    slide_limits(prs, meta)
    slide_conclusions(prs, meta)
    slide_thanks(prs, meta)

    out = Path("presentation")
    out.mkdir(exist_ok=True)
    path = out / "Sputnix_cloud_detection.pptx"
    prs.save(path)
    print(f"[pptx] Презентация собрана: {path} "
          f"({len(prs.slides._sldIdLst)} слайдов)")
    print(f"[pptx] Метрики: "
          f"{'реальные' if meta else 'плейсхолдеры (metrics.json нет)'}")


if __name__ == "__main__":
    main()
