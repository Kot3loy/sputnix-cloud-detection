"""build_pptx.py — сборка презентации кейса «Спутникс» (.pptx, python-pptx)."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

BG = RGBColor(0x0B, 0x14, 0x2A)
ACCENT = RGBColor(0x4F, 0xC3, 0xF7)
ACCENT2 = RGBColor(0x00, 0xE5, 0xC0)
WHITE = RGBColor(0xF5, 0xF7, 0xFA)
GRAY = RGBColor(0xB0, 0xBE, 0xC5)
DARK = RGBColor(0x08, 0x10, 0x20)
RED = RGBColor(0xEF, 0x53, 0x50)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)


def load_metrics() -> dict | None:
    p = Path("outputs/metrics/metrics.json")
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_train_summary() -> dict | None:
    p = Path("outputs/logs/train_summary.json")
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def add_slide(prs: Presentation) -> object:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(1, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return slide


def add_text(slide, x, y, w, h, text, size=18, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, line_spacing=1.15):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = line_spacing
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.bold = bold
            run.font.name = "Segoe UI"
    return box


def add_title(slide, text, subtitle=None):
    add_text(slide, Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.9),
             text, size=30, color=ACCENT, bold=True)
    line = slide.shapes.add_shape(1, Inches(0.6), Inches(1.12), Inches(3.0),
                                  Pt(3))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT2
    line.line.fill.background()
    if subtitle:
        add_text(slide, Inches(0.55), Inches(1.2), Inches(12.2), Inches(0.5),
                 subtitle, size=14, color=GRAY)


def add_bullets(slide, x, y, w, h, bullets, size=16, color=WHITE,
                bullet_color=ACCENT2):
    """Список с маркерами-тире."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.25
        r1 = p.add_run()
        r1.text = "—  "
        r1.font.color.rgb = bullet_color
        r1.font.bold = True
        r1.font.size = Pt(size)
        r1.font.name = "Segoe UI"
        r2 = p.add_run()
        r2.text = b
        r2.font.color.rgb = color
        r2.font.size = Pt(size)
        r2.font.name = "Segoe UI"
    return box


def add_card(slide, x, y, w, h, title, body, title_color=ACCENT2):
    card = slide.shapes.add_shape(5, x, y, w, h)
    card.fill.solid()
    card.fill.fore_color.rgb = DARK
    card.line.color.rgb = title_color
    card.line.width = Pt(1.2)
    card.shadow.inherit = False
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.18)
    tf.margin_top = Inches(0.12)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = title_color
    r.font.name = "Segoe UI"
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = body
    r2.font.size = Pt(13)
    r2.font.color.rgb = WHITE
    r2.font.name = "Segoe UI"
    p2.space_before = Pt(4)
    return card


def fmt(v, nd=4, suffix=""):
    return f"{v:.{nd}f}{suffix}" if isinstance(v, (int, float)) else v


def slide_title(prs, meta):
    s = add_slide(prs)
    glyph = Path("assets/sputnix_satellite.png")
    if glyph.exists():
        s.shapes.add_picture(str(glyph), Inches(6.3), Inches(0.55),
                             height=Inches(0.95))
    add_text(s, Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.6),
             "SK KIDS CHALLENGE • Кейс компании «Спутникс»", size=16,
             color=GRAY, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.9),
             "Алгоритм ИИ для распознавания облачности\nна спутниковых снимках",
             size=34, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.8), Inches(4.7), Inches(11.7), Inches(0.6),
             "Семантическая сегментация облаков (U-Net) • порог отбраковки 75%",
             size=18, color=ACCENT2, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.8),
             "Автор: Белиевский Давид, 16 лет     "
             "PyTorch • Python 3.10+",
             size=14, color=GRAY, align=PP_ALIGN.CENTER)


def slide_problem(prs, meta):
    s = add_slide(prs)
    add_title(s, "Проблема и цель")
    add_bullets(s, Inches(0.7), Inches(1.6), Inches(6.6), Inches(4.8), [
        "Спутник Спутникс делает тысячи снимков; значительная часть закрыта "
        "облаками и бесполезна для работы",
        "Просматривать вручную невозможно — нужен автоматический фильтр",
        "Требование заказчика: определять процент облачности и отсеивать "
        "снимки, закрытые более чем на 75%",
        "Оператор должен получать понятный отчёт: что сохранено, что удалено "
        "и почему",
    ])
    add_text(s, Inches(7.6), Inches(1.6), Inches(5.2), Inches(0.5),
             "Цель проекта", size=18, color=ACCENT, bold=True)
    add_bullets(s, Inches(7.6), Inches(2.1), Inches(5.2), Inches(4.5), [
        "Модель сегментации: пиксель = облако или нет",
        "Расчёт процента облачности по маске",
        "Автоотбраковка по порогу 75% (по ТЗ — с удалением)",
        "Отчёт оператору: сохранённые и удалённые файлы с процентами",
    ], size=14)


def slide_data(prs, meta):
    s = add_slide(prs)
    add_title(s, "Входные данные")
    add_card(s, Inches(0.7), Inches(1.5), Inches(6.2), Inches(2.3),
             "Снимки «Спутникс» (ссылка из ТЗ)",
             "Открытый ресурс disk.sputnix.ru: 4 213 JPEG-снимков обзорных "
             "камер кубсатов SITRO-AIS, 3280×2464, RGB, ~200 м/пиксель, "
             "+ KML-геопривязка. Использованы для инференса и демо.")
    add_card(s, Inches(0.7), Inches(4.0), Inches(6.2), Inches(2.3),
             "38-Cloud — обучение",
             "Открытый датасет Sentinel-2: 8 400 патчей 384×384 с масками "
             "облаков. В шеринга Спутникс масок нет, поэтому обучаем на "
             "38-Cloud и применяем модель к снимкам Спутникс (transfer).")
    add_card(s, Inches(7.3), Inches(1.5), Inches(5.4), Inches(2.3),
             "Почему нужна нейросеть?",
             "Простая эвристика «яркое = облако» путает облака со снегом, "
             "льдом, дымкой и пересветами. Нейросеть учится текстуре и "
             "контексту — работает надёжнее.")
    add_card(s, Inches(7.3), Inches(4.0), Inches(5.4), Inches(2.3),
             "Резерв: синтетика",
             "Свой генератор (make_demo_data.py) создаёт снимки с точными "
             "масками — smoke-тест пайплайна и запасное демо без интернета.")


def slide_arch(prs, meta):
    s = add_slide(prs)
    add_title(s, "Архитектура алгоритма")
    add_text(s, Inches(0.7), Inches(1.5), Inches(12), Inches(0.8),
             "U-Net — стандарт для сегментации: сжимает изображение, извлекая "
             "признаки, и восстанавливает маску в исходном разрешении",
             size=16, color=GRAY)
    boxes = [
        ("Снимок\n3×384×384", ACCENT),
        ("Энкодер ResNet18\n(сжатие, признаки)", ACCENT),
        ("Бутылочное\nгорлышко", ACCENT2),
        ("Декодер\n(восстановление)", ACCENT),
        ("Маска облаков\n1×384×384", RED),
    ]
    x = Inches(0.7)
    y = Inches(2.6)
    w, h = Inches(2.25), Inches(1.5)
    gap = Inches(0.32)
    for i, (label, c) in enumerate(boxes):
        box = s.shapes.add_shape(5, x, y, w, h)
        box.fill.solid()
        box.fill.fore_color.rgb = DARK
        box.line.color.rgb = c
        box.line.width = Pt(1.5)
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = label
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = "Segoe UI"
        p.alignment = PP_ALIGN.CENTER
        if i < len(boxes) - 1:
            arrow = s.shapes.add_shape(13, x + w + Emu(20000),
                                       y + h // 2 - Inches(0.15),
                                       gap - Emu(40000), Inches(0.3))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = c
            arrow.line.fill.background()
        x = x + w + gap
    add_bullets(s, Inches(0.7), Inches(4.5), Inches(12), Inches(2.4), [
        "13.7 млн параметров, вход RGB 384×384; выход — вероятность облака "
        "в каждом пикселе (сигмоида)",
        "Лосс: BCE + Dice; оптимизатор AdamW, смешанная точность (AMP) — "
        "быстро и экономно по памяти",
        "Fallback: собственная SmallUNet в model.py — работает даже без "
        "библиотеки segmentation-models-pytorch",
    ], size=14)


def slide_pipeline(prs, meta):
    s = add_slide(prs)
    add_title(s, "Пайплайн обработки снимка")
    steps = [
        ("1. Загрузка", "снимок JPG/TIF/PNG,\nучёт NODATA-пикселей"),
        ("2. Сегментация", "U-Net строит маску;\nбольшие кадры — тайлами"),
        ("3. Подсчёт %", "облачные пиксели /\nвалидные пиксели × 100"),
        ("4. Порог 75%", "> 75% → отбраковка;\n≤ 75% → сохранение"),
        ("5. Отчёт", "консоль, CSV, JSON,\nсообщение оператору"),
    ]
    x = Inches(0.55)
    y = Inches(1.9)
    w, h = Inches(2.35), Inches(2.1)
    gap = Inches(0.15)
    for i, (t, b) in enumerate(steps):
        c = RED if i == 3 else ACCENT2
        box = s.shapes.add_shape(5, x, y, w, h)
        box.fill.solid()
        box.fill.fore_color.rgb = DARK
        box.line.color.rgb = c
        box.line.width = Pt(1.3)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.12)
        tf.margin_top = Inches(0.1)
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = t
        r.font.bold = True
        r.font.size = Pt(15)
        r.font.color.rgb = c
        r.font.name = "Segoe UI"
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = b
        r2.font.size = Pt(12)
        r2.font.color.rgb = WHITE
        r2.font.name = "Segoe UI"
        p2.space_before = Pt(6)
        x = x + w + gap
    add_bullets(s, Inches(0.7), Inches(4.5), Inches(12), Inches(2.3), [
        "Формула процента: cloud_pixels / valid_pixels × 100 "
        "(валидные = не полностью чёрные, для снимков Спутникс — все)",
        "Отбраковка по умолчанию безопасная — файл перемещается в "
        "rejected/; флаг --hard-delete удаляет по ТЗ буквально",
        "Один кадр 3280×2464 обрабатывается ≈ за 0.5–1.5 с на GPU",
    ], size=14)


def fmt_conf_cell(v):
    return str(int(v)) if isinstance(v, (int, float)) else "—"


def slide_metrics(prs, meta, train):
    s = add_slide(prs)
    add_title(s, "Обучение и метрики")
    if meta is None:
        add_text(s, Inches(0.7), Inches(1.5), Inches(12), Inches(0.6),
                 "Метрики появятся после запуска evaluate.py — сейчас "
                 "плейсхолдеры:", size=15, color=RED)
    mm = (meta or {}).get("mask_metrics", {})
    conf = (meta or {}).get("threshold_75_confusion", {})
    ts = train or {}

    rows = [
        ("IoU (Jaccard) маски", fmt(mm.get("IoU"))),
        ("Dice / F1 маски", fmt(mm.get("Dice"))),
        ("Precision", fmt(mm.get("Precision"))),
        ("Recall", fmt(mm.get("Recall"))),
        ("MAE процента облачности",
         f"{meta.get('cloud_pct_MAE')} п.п." if meta else "—"),
    ]
    y = Inches(2.0)
    for label, val in rows:
        add_text(s, Inches(0.8), y, Inches(4.6), Inches(0.42), label,
                 size=15, color=WHITE)
        add_text(s, Inches(5.4), y, Inches(1.8), Inches(0.42),
                 str(val), size=15, color=ACCENT2, bold=True)
        y += Inches(0.5)
    add_text(s, Inches(7.7), Inches(1.7), Inches(5), Inches(0.5),
             "Решение по порогу 75%:", size=16, color=ACCENT, bold=True)
    cells = [
        ("TP верно отбраковано", conf.get("tp_rejected_correctly"), ACCENT2),
        ("FN пропущен брак", conf.get("fn_kept_wrongly"), RED),
        ("FP зря отбраковано", conf.get("fp_rejected_wrongly"), RED),
        ("TN верно сохранено", conf.get("tn_kept_correctly"), ACCENT2),
    ]
    positions = [(Inches(7.7), Inches(2.3)), (Inches(10.4), Inches(2.3)),
                 (Inches(7.7), Inches(3.3)), (Inches(10.4), Inches(3.3))]
    for (label, val, c), (cx, cy) in zip(cells, positions):
        card = s.shapes.add_shape(5, cx, cy, Inches(2.55), Inches(0.85))
        card.fill.solid()
        card.fill.fore_color.rgb = DARK
        card.line.color.rgb = c
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = f"{val if val is not None else '—'}  {label}"
        r.font.size = Pt(12)
        r.font.color.rgb = WHITE
        r.font.name = "Segoe UI"
    if meta:
        acc = conf.get("decision_accuracy")
        add_text(s, Inches(7.7), Inches(4.4), Inches(5.2), Inches(0.5),
                 f"Точность решения: {acc*100:.1f}%", size=17,
                 color=ACCENT2, bold=True)
    info = []
    if ts:
        info = [f"Датасет: {ts.get('dataset', '—')} • "
                f"train {ts.get('train_pairs', '—')} пар / val {ts.get('val_pairs', '—')}",
                f"Модель: {ts.get('arch')} ({ts.get('encoder')}) • "
                f"{ts.get('epochs_trained')} эпох • {ts.get('total_time_min')} мин • "
                f"{ts.get('device')}"]
    else:
        info = ["Параметры обучения появятся после train.py (train_summary.json)"]
    add_bullets(s, Inches(0.8), Inches(5.1), Inches(11.8), Inches(1.6),
                info, size=13, color=GRAY)


def load_report_examples(n_saved: int = 2, n_rej: int = 2) -> list[str]:
    """Реальные строки отчёта оператору из последнего прогона predict.py."""
    p = Path("outputs/results/report.json")
    lines = []
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                rep = json.load(f)
            saved = [r for r in rep.get("files", []) if r.get("decision") == "accepted"]
            rej = [r for r in rep.get("files", []) if r.get("decision") == "rejected"]
            for r in saved[:n_saved] + rej[:n_rej]:
                verb = "Сохранён" if r["decision"] == "accepted" else "Удалён"
                lines.append(f"{verb}: {r['filename']} — облачность {r['cloud_pct']}%.")
        except Exception:
            pass
    if not lines:
        lines = ["(запустите predict.py для реального отчёта)"]
    return lines


def slide_threshold(prs, meta):
    s = add_slide(prs)
    add_title(s, "Логика порога 75% и отбраковки")
    add_text(s, Inches(0.7), Inches(1.5), Inches(12), Inches(0.7),
             "Требование ТЗ: если облака занимают больше 75% кадра — снимок "
             "не имеет ценности и удаляется", size=16, color=GRAY)
    card = s.shapes.add_shape(5, Inches(0.7), Inches(2.4), Inches(7.6),
                              Inches(2.6))
    card.fill.solid()
    card.fill.fore_color.rgb = DARK
    card.line.color.rgb = ACCENT2
    card.line.width = Pt(1.5)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "СООБЩЕНИЕ ОПЕРАТОРУ (реальный вывод predict.py)"
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = ACCENT2
    r.font.name = "Consolas"
    for line in load_report_examples():
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = line
        r2.font.size = Pt(14)
        r2.font.color.rgb = WHITE
        r2.font.name = "Consolas"
        p2.space_before = Pt(6)
    add_bullets(s, Inches(8.6), Inches(2.4), Inches(4.2), Inches(4.2), [
        "> 75% → перемещение в rejected/",
        "≤ 75% → копия в accepted/",
        "--hard-delete → удаление по ТЗ",
        "Отчёты: report.csv, report.json, operator_report.txt",
    ], size=14)


def slide_demo(prs, meta):
    s = add_slide(prs)
    add_title(s, "Демонстрация работы")
    vis_dir = Path("outputs/results/visualizations")
    preferred = ["2026-01-31", "2026-04-16", "2025-11-03"]
    images = []
    if vis_dir.exists():
        all_vis = sorted(vis_dir.glob("*.jpg"))
        for key in preferred:
            for p in all_vis:
                if key in p.name and p not in images:
                    images.append(p)
                    break
        if len(images) < 3:
            for p in all_vis:
                if p not in images:
                    images.append(p)
                if len(images) == 3:
                    break
    x = Inches(0.55)
    for img_path in images[:3]:
        try:
            pic = s.shapes.add_picture(str(img_path), x, Inches(2.2),
                                       height=Inches(3.15))
            x = Emu(int(pic.left + pic.width + Inches(0.15)))
        except Exception:
            pass
    if not images:
        add_text(s, Inches(0.7), Inches(2.0), Inches(12), Inches(1.0),
                 "Скриншоты появятся после запуска:\n"
                 "python predict.py --input demo/demo_input --save-masks",
                 size=15, color=RED)
    add_text(s, Inches(0.7), Inches(6.0), Inches(12), Inches(0.9),
             "Маска облаков наложена на снимок, подписан процент. Слева — "
             "отбракованный реальный кадр (77.4% > 75%), в центре — облачный "
             "океан (72.4%), справа — ясная река (2.2%)",
             size=14, color=GRAY)


def slide_gui(prs, meta):
    s = add_slide(prs)
    add_title(s, "Графический интерфейс оператора")
    shots = Path("outputs/gui_shots")
    idle = shots / "gui_idle.png"
    done = shots / "gui_done.png"
    if done.exists():
        pic = s.shapes.add_picture(str(done), Inches(0.55), Inches(1.5),
                                   width=Inches(8.1))
    if idle.exists():
        s.shapes.add_picture(str(idle), Inches(8.85), Inches(1.5),
                             width=Inches(4.0))
    add_bullets(s, Inches(0.7), Inches(6.1), Inches(12), Inches(1.2), [
        "Слева — программа после обработки 26 снимков: лог оператора, "
        "список результатов с процентами, превью маски",
        "Справа — экран запуска: выбор папки, порог отбраковки, "
        "статус системы; порог можно двигать ползунком",
    ], size=13, color=GRAY)


def slide_limits(prs, meta):
    s = add_slide(prs)
    add_title(s, "Ограничения и развитие")
    add_text(s, Inches(0.7), Inches(1.5), Inches(5.9), Inches(0.5),
             "Ограничения", size=18, color=RED, bold=True)
    add_bullets(s, Inches(0.7), Inches(2.0), Inches(5.9), Inches(4.6), [
        "Domain gap: обучение на Sentinel-2, инференс на камерах кубсатов — "
        "цвета и разрешение различаются",
        "Тонкие перистые облака и дымка — самые сложные классы",
        "Снег/лёд могут распознаваться как облака (в 38-Cloud мало арктики)",
        "Ровно 75% — граница решения; метрика в этой зоне шумит",
    ], size=14, color=WHITE)
    add_text(s, Inches(7.0), Inches(1.5), Inches(5.8), Inches(0.5),
             "Как развивать", size=18, color=ACCENT2, bold=True)
    add_bullets(s, Inches(7.0), Inches(2.0), Inches(5.8), Inches(4.6), [
        "Дообучение (fine-tune) на вручную размеченных снимках Спутникс",
        "Использование KML-телеметрии: широта/облачность по климатологии",
        "Более сильные модели: DeepLabV3+, SegFormer",
        "Доразметка: полуавтоматическая разметка моделью + проверка "
        "человеком, итерации",
    ], size=14, color=WHITE)


def slide_conclusions(prs, meta):
    s = add_slide(prs)
    add_title(s, "Выводы")
    mm = (meta or {}).get("mask_metrics", {}) if meta else {}
    acc = (meta or {}).get("threshold_75_confusion", {}).get(
        "decision_accuracy") if meta else None
    bullets = [
        "Полный пайплайн собран и работает: данные → обучение → метрики → "
        "инференс → отчёт оператору",
        ("Качество маски: IoU = " + fmt(mm.get("IoU")) + ", Dice = "
         + fmt(mm.get("Dice"))) if mm else
        "Качество маски: заполнится после обучения (evaluate.py)",
        ("Точность решения по порогу 75%: " + f"{acc*100:.1f}%") if acc else
        "Точность решения по порогу 75%: заполнится после evaluate.py",
        "Все требования ТЗ выполнены: процент облачности, порог 75%, "
        "отбраковка, отчёт оператору",
        "Проект готов к развитию: дообучение на данных Спутникс, более "
        "сильные модели",
    ]
    add_bullets(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(4.5),
                bullets, size=19)


def slide_thanks(prs, meta):
    s = add_slide(prs)
    add_text(s, Inches(0.8), Inches(2.6), Inches(11.7), Inches(1.2),
             "Спасибо за внимание!", size=44, color=WHITE, bold=True,
             align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.8), Inches(4.0), Inches(11.7), Inches(0.8),
             "Демонстрация: predict.py вживую на снимках Спутникс", size=18,
             color=ACCENT2, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.8), Inches(5.0), Inches(11.7), Inches(0.8),
             "Данные: disk.sputnix.ru (по ТЗ) • 38-Cloud • код: GitHub",
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
    print(f"[pptx] Презентация собрана: {path}")
    print(f"[pptx] Слайдов: {len(prs.slides.__iter__.__self__._sldIdLst)}")
    print(f"[pptx] Метрики: {'реальные из metrics.json' if meta else 'плейсхолдеры (metrics.json нет)'}")


if __name__ == "__main__":
    main()
