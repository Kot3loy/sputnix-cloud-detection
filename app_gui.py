"""Пульт обработки спутниковых снимков «Спутникс» — десктопный интерфейс.

Запуск: ЗАПУСК_ПРОГРАММЫ.bat (или `python app_gui.py`).
Тёмная тема, логика: маска облаков -> процент облачности -> порог 75%.
"""
from __future__ import annotations

import ctypes
import os
import queue
import threading
import traceback
from pathlib import Path

import cv2
import torch
import customtkinter as ctk

from dataset import load_image_rgb
from model import build_model
from predict import adaptive_preprocess, predict_mask
from utils import (accept_file, cloud_percentage, format_operator_line,
                   overlay_mask, reject_file, valid_pixels_mask,
                   write_reports)

BG = "#0B0F1A"
RAIL = "#070B12"
CARD = "#141A26"
CARD_HI = "#1C2536"
FIELD = "#0E1420"
ACCENT = "#4D8DFF"
ACCENT_DK = "#3B74E0"
GOOD = "#35C77B"
BAD = "#FF5C5C"
WARN = "#FFB84D"
TEXT = "#F2F5FA"
MUTED = "#8A94A6"
LINE = "#232C3D"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _pick_font(*candidates):
    import tkinter.font as tkfont
    root = ctk.CTk()
    root.withdraw()
    fams = set(tkfont.families(root))
    root.destroy()
    for f in candidates:
        if f in fams:
            return f
    return "Segoe UI"


FONT = _pick_font("Bahnschrift", "Segoe UI Variable Display", "Segoe UI")
F = lambda s, w="normal": (FONT, s, w)  # noqa: E731

CKPT_CANDIDATES = [Path("outputs/checkpoints/best_model.pth"),
                   Path("model_trained/best_model.pth")]
RESULTS_DIR = Path("outputs/results")


def find_ckpt():
    for p in CKPT_CANDIDATES:
        if p.exists():
            return p
    return None


class CloudApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Спутникс · Обработка облачности")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1280, sw - 60)}x{min(800, sh - 70)}")
        self.minsize(min(1100, sw - 40), min(680, sh - 50))
        self.configure(fg_color=BG)
        self._dark_titlebar()
        if Path("assets/sputnix_icon.ico").exists():
            try:
                self.iconbitmap("assets/sputnix_icon.ico")
            except Exception:
                pass

        self.model = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.log_q = queue.Queue()
        self.results = []
        self._photo = None
        self.current = None

        self._build_rail()
        self._build_main()
        self.after(100, self._drain_log)
        self.after(250, self._refresh_status)

    def _dark_titlebar(self):
        try:
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            val = ctypes.c_int(1)
            for attr in (20, 19):
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, attr, ctypes.byref(val),
                        ctypes.sizeof(val)) == 0:
                    break
        except Exception:
            pass

    # -------------------------------------------------------- левая панель
    def _build_rail(self):
        rail = ctk.CTkFrame(self, width=232, corner_radius=0, fg_color=RAIL)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)

        logo = Path("assets/sputnix_lockup_white.png")
        if logo.exists():
            from PIL import Image
            img = Image.open(logo)
            w = 186
            self._logo = ctk.CTkImage(light_image=img, dark_image=img,
                                      size=(w, int(img.height * w / img.width)))
            ctk.CTkLabel(rail, image=self._logo, text=""
                         ).pack(anchor="w", padx=20, pady=(26, 2))
        else:
            head = ctk.CTkFrame(rail, fg_color="transparent")
            head.pack(fill="x", padx=20, pady=(26, 0))
            ctk.CTkLabel(head, text="☁", font=("Segoe UI Emoji", 28),
                         text_color=ACCENT).pack(side="left")
            ctk.CTkLabel(head, text=" СПУТНИКС", font=F(20, "bold"),
                         text_color=TEXT).pack(side="left")
        ctk.CTkLabel(rail, text="пульт распознавания\nоблачности",
                     font=F(12), text_color=MUTED, justify="left"
                     ).pack(anchor="w", padx=24, pady=(0, 4))

        ctk.CTkFrame(rail, height=1, fg_color=LINE).pack(
            fill="x", padx=22, pady=16)

        ctk.CTkLabel(rail, text="Состояние системы", font=F(12, "bold"),
                     text_color=MUTED, anchor="w").pack(fill="x", padx=24)
        self.chips_box = ctk.CTkFrame(rail, fg_color="transparent")
        self.chips_box.pack(fill="x", padx=18, pady=(8, 0))

        bottom = ctk.CTkFrame(rail, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=14, pady=14)
        for icon, label, path in [("≡", "Отчёты", RESULTS_DIR),
                                  ("✓", "Сохранённые", RESULTS_DIR / "accepted"),
                                  ("✗", "Отбракованные", RESULTS_DIR / "rejected")]:
            ctk.CTkButton(bottom, text=f"{icon}   {label}", height=36,
                          corner_radius=10, fg_color="transparent",
                          text_color=MUTED, hover_color=CARD_HI, anchor="w",
                          font=F(13), command=lambda p=path: self._open(p)
                          ).pack(fill="x", pady=2)

    def _chip(self, parent, text_, color):
        chip = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=LINE)
        chip.pack(fill="x", pady=3)
        inner = ctk.CTkFrame(chip, fg_color="transparent")
        inner.pack(anchor="w", padx=10, pady=6)
        ctk.CTkLabel(inner, text="●", font=("Segoe UI", 11),
                     text_color=color).pack(side="left")
        ctk.CTkLabel(inner, text="  " + text_, font=F(12),
                     text_color=TEXT).pack(side="left")

    # -------------------------------------------------------- главная часть
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(side="left", fill="both", expand=True, padx=20, pady=14)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text="Обработка снимков", font=F(26, "bold"),
                     text_color=TEXT, anchor="w").pack(side="left")
        ctk.CTkLabel(head, text="маска облаков · процент облачности · "
                                "отбраковка > 75%",
                     font=F(13), text_color=MUTED).pack(side="right",
                                                        pady=(10, 0))

        # --- панель управления ---
        panel = ctk.CTkFrame(main, fg_color=CARD, corner_radius=16)
        panel.pack(fill="x", pady=(14, 10))
        prow = ctk.CTkFrame(panel, fg_color="transparent")
        prow.pack(fill="x", padx=16, pady=(12, 0))

        ctk.CTkLabel(prow, text="Папка со снимками", font=F(12.5),
                     text_color=MUTED, anchor="w").grid(
            row=0, column=0, sticky="w")
        self.folder_var = ctk.StringVar(value=str(Path("demo/demo_input")))
        self.folder_entry = ctk.CTkEntry(prow, textvariable=self.folder_var,
                                         height=38, corner_radius=9,
                                         fg_color=FIELD, border_color=LINE,
                                         font=F(12))
        self.folder_entry.grid(row=1, column=0, sticky="ew", pady=(2, 0),
                               padx=(0, 10))
        ctk.CTkButton(prow, text="Обзор", width=84, height=38,
                      corner_radius=9, fg_color=CARD_HI,
                      hover_color="#26314A", text_color=TEXT,
                      font=F(12.5, "bold"), command=self._browse
                      ).grid(row=1, column=1, sticky="n", pady=(2, 0))

        ctk.CTkLabel(prow, text="Порог отбраковки", font=F(12.5),
                     text_color=MUTED, anchor="w").grid(
            row=0, column=2, sticky="w", padx=(24, 0))
        self.thr_value = ctk.StringVar(value="75%")
        self.thr_slider = ctk.CTkSlider(prow, from_=30, to=95,
                                        button_color=ACCENT,
                                        button_hover_color=ACCENT_DK,
                                        progress_color=ACCENT,
                                        fg_color=LINE, height=22,
                                        command=self._on_slider)
        self.thr_slider.set(75)
        self.thr_slider.grid(row=1, column=2, sticky="ew", padx=(24, 0),
                             pady=(8, 0))
        ctk.CTkLabel(prow, textvariable=self.thr_value, font=F(17, "bold"),
                     text_color=ACCENT).grid(row=0, column=3, padx=(10, 0))

        self.hard_var = ctk.BooleanVar(value=False)
        self.hard_switch = ctk.CTkSwitch(prow, text="", width=48,
                                         variable=self.hard_var,
                                         progress_color=BAD,
                                         button_color="#DDE3EC",
                                         command=self._on_switch)
        self.hard_switch.grid(row=1, column=3, sticky="e", pady=(4, 0),
                              padx=(6, 0))

        prow.grid_columnconfigure(0, weight=1)
        prow.grid_columnconfigure(2, weight=1)

        brow = ctk.CTkFrame(panel, fg_color="transparent")
        brow.pack(fill="x", padx=16, pady=(10, 12))
        self.run_btn = ctk.CTkButton(brow, text="Обработать снимки",
                                     height=46, corner_radius=23,
                                     fg_color=ACCENT, hover_color=ACCENT_DK,
                                     text_color="#06101F", font=F(15, "bold"),
                                     command=self._start)
        self.run_btn.pack(side="left", fill="x", expand=True)
        self.progress = ctk.CTkProgressBar(brow, height=6, corner_radius=3,
                                           progress_color=ACCENT,
                                           fg_color=LINE, width=180)
        self.progress.set(0)
        self.progress.pack(side="right", padx=(14, 0), fill="x",
                           expand=True)

        # --- средняя зона: превью + результаты ---
        mid = ctk.CTkFrame(main, fg_color="transparent")
        mid.pack(fill="both", expand=True)
        mid.grid_columnconfigure(0, weight=3)
        mid.grid_columnconfigure(1, weight=2)
        mid.grid_rowconfigure(0, weight=1)

        prev_card = ctk.CTkFrame(mid, fg_color=CARD, corner_radius=16)
        prev_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.preview_label = ctk.CTkLabel(
            prev_card,
            text="выберите файл в результатах —\nздесь появится маска облаков",
            font=F(13), text_color="#5A6478")
        self.preview_label.pack(fill="both", expand=True, padx=14,
                                pady=(10, 0))
        self.hero_caption = ctk.CTkLabel(prev_card, text="—", font=F(12.5),
                                         text_color=MUTED, anchor="w")
        self.hero_caption.pack(fill="x", padx=16, pady=(2, 10))

        res_card = ctk.CTkFrame(mid, fg_color=CARD, corner_radius=16)
        res_card.grid(row=0, column=1, sticky="nsew")
        rhead = ctk.CTkFrame(res_card, fg_color="transparent")
        rhead.pack(fill="x", padx=16, pady=(10, 0))
        ctk.CTkLabel(rhead, text="Результаты", font=F(14, "bold"),
                     text_color=TEXT).pack(side="left")
        self.summary_label = ctk.CTkLabel(rhead, text="", font=F(12),
                                          text_color=MUTED)
        self.summary_label.pack(side="right")
        self.res_list = ctk.CTkScrollableFrame(res_card,
                                               fg_color="transparent")
        self.res_list.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        ctk.CTkLabel(self.res_list, text="здесь появятся снимки",
                     font=F(12), text_color="#5A6478").pack(pady=20)

        # --- лог ---
        log_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=16)
        log_card.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(log_card, text="Сообщение оператору", font=F(13, "bold"),
                     text_color=TEXT, anchor="w").pack(fill="x", padx=16,
                                                       pady=(8, 0))
        self.log_box = ctk.CTkTextbox(log_card, height=150,
                                      font=("Consolas", 11.5),
                                      fg_color=FIELD, text_color="#79E0A5",
                                      corner_radius=10, wrap="none")
        self.log_box.pack(fill="x", padx=12, pady=(4, 12))
        self.log_box.insert("0.0", "Ожидание запуска обработки…")
        self.log_box.configure(state="disabled")

    # -------------------------------------------------------- события
    def _on_slider(self, value):
        self.thr_value.set(f"{int(value)}%")

    def _on_switch(self):
        pass

    def _browse(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Папка со спутниковыми снимками")
        if folder:
            self.folder_var.set(folder)

    def _open(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # noqa: S606
        except Exception as e:
            self.log(f"[!] не удалось открыть: {e}")

    def _refresh_status(self):
        gpu = self.device.type == "cuda"
        self._chip(self.chips_box,
                   f"GPU · {torch.cuda.get_device_name(0)}" if gpu
                   else "CPU (медленнее)", GOOD if gpu else WARN)
        ckpt = find_ckpt()
        if ckpt:
            import torch as _t
            ck = _t.load(ckpt, map_location="cpu", weights_only=False)
            self._chip(self.chips_box,
                       f"модель · готова (IoU {ck.get('val_iou', 0):.4f})",
                       GOOD)
        else:
            self._chip(self.chips_box, "модель · не найдена", BAD)
        demo_ok = Path("demo/demo_input").exists() or \
            Path("demo_input").exists()
        self._chip(self.chips_box,
                   "демо-данные · есть" if demo_ok else "демо-данные · нет",
                   GOOD if demo_ok else WARN)

    def log(self, line: str):
        self.log_q.put(line)

    def _drain_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", line + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    # -------------------------------------------------------- пайплайн
    def _start(self):
        folder = Path(self.folder_var.get())
        if not folder.exists():
            self.log(f"[X] Папка не найдена: {folder}")
            return
        ckpt = find_ckpt()
        if ckpt is None:
            self.log("[X] Модель не найдена — обучите её: train.py")
            return
        thr = float(self.thr_slider.get())
        self.run_btn.configure(state="disabled", text="Обработка…")
        for w in self.res_list.winfo_children():
            w.destroy()
        self.results = []
        n = len([p for p in folder.iterdir()
                 if p.is_file() and p.suffix in IMAGE_EXTS])
        self.progress.set(0)
        self.log("=" * 58)
        self.log(f"Папка: {folder}  ({n} снимков)")
        self.log(f"Порог > {thr:.0f}%  |  отбраковка: "
                 f"{'УДАЛЕНИЕ' if self.hard_var.get() else 'в rejected/'}")
        self.log("-" * 58)
        threading.Thread(target=self._worker, args=(folder, thr, ckpt),
                         daemon=True).start()

    def _worker(self, folder: Path, thr: float, ckpt_path: Path):
        try:
            if self.model is None:
                self.log("Загрузка модели…")
                ckpt = torch.load(ckpt_path, map_location=self.device,
                                  weights_only=False)
                self.model, arch = build_model(ckpt.get("arch", "unet"),
                                               ckpt.get("encoder", "resnet18"),
                                               pretrained=False)
                self.model.load_state_dict(ckpt["model_state"])
                self.model.to(self.device).eval()
                self.log(f"Модель {arch} готова (эпоха {ckpt.get('epoch')})")

            files = sorted(p for p in folder.iterdir()
                           if p.is_file() and p.suffix in IMAGE_EXTS)
            vis_dir = RESULTS_DIR / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            results = []

            for i, f in enumerate(files, 1):
                try:
                    image = load_image_rgb(f)
                    mask = predict_mask(self.model,
                                        adaptive_preprocess(image),
                                        self.device, tile_size=512,
                                        overlap=0.25)
                    pct = cloud_percentage(mask, valid_pixels_mask(image))
                    if pct > thr:
                        reject_file(f, RESULTS_DIR / "rejected",
                                    hard_delete=self.hard_var.get())
                        decision = "rejected"
                    else:
                        accept_file(f, RESULTS_DIR / "accepted", copy=True)
                        decision = "accepted"
                    row = {"filename": f.name, "cloud_pct": round(pct, 1),
                           "decision": decision}
                    results.append(row)
                    self.log("  " + format_operator_line(row))

                    vis = overlay_mask(image, mask)
                    fs = max(0.9, image.shape[1] / 2400)
                    label = (f"Cloud: {pct:.1f}%  ->  "
                             f"{'REJECTED' if decision == 'rejected' else 'OK'}")
                    cv2.putText(vis, label, (16, int(40 * fs + 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, (60, 60, 255),
                                max(2, int(fs * 3)), cv2.LINE_AA)
                    from utils import imwrite_unicode
                    imwrite_unicode(vis_dir / f"{f.stem}_mask.jpg",
                                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    self.log(f"  ОШИБКА {f.name}: {type(e).__name__}: {e}")
                self.progress.after(0, lambda v=i / max(len(files), 1):
                                    self.progress.set(v))

            n_rej = sum(1 for r in results if r["decision"] == "rejected")
            self.log("-" * 58)
            self.log(f"ИТОГО: обработано {len(results)} | сохранено "
                     f"{len(results) - n_rej} | отбраковано {n_rej} "
                     f"(> {thr:.0f}%)")
            write_reports(results, RESULTS_DIR,
                          run_meta={"cloud_threshold_pct": thr,
                                    "hard_delete": self.hard_var.get(),
                                    "input": str(folder)})
            self.results = results
            self.after(0, self._fill_results)
        except Exception:
            self.log("КРИТИЧЕСКАЯ ОШИБКА:\n" + traceback.format_exc())
        finally:
            self.after(0, lambda: self.run_btn.configure(
                state="normal", text="Обработать снимки"))
            self.after(0, lambda: self.progress.set(0))

    def _fill_results(self):
        for w in self.res_list.winfo_children():
            w.destroy()
        for r in self.results:
            rej = r["decision"] == "rejected"
            color = BAD if rej else GOOD
            name = r["filename"]
            if len(name) > 34:
                name = name[:30] + "…" + name[-4:]
            row = ctk.CTkFrame(self.res_list, fg_color=FIELD,
                               corner_radius=8, height=34)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)
            mark = ctk.CTkLabel(row, text="✗" if rej else "✓", width=26,
                                font=F(13, "bold"), text_color=color)
            mark.pack(side="left", padx=(10, 0))
            ctk.CTkLabel(row, text=name, anchor="w", font=F(12),
                         text_color=TEXT).pack(side="left", padx=(6, 0))
            badge = ctk.CTkFrame(row, fg_color=color, corner_radius=11)
            badge.pack(side="right", padx=8, pady=5)
            ctk.CTkLabel(badge, text=f"{r['cloud_pct']:.1f}%",
                         font=F(11.5, "bold"), text_color="#08101C"
                         ).pack(padx=9, pady=2)
            for w in (row, mark):
                w.bind("<Button-1>", lambda _e, r=r: self._show_preview(r))
            for ch in row.winfo_children():
                ch.bind("<Button-1>", lambda _e, r=r: self._show_preview(r))
        n_rej = sum(1 for r in self.results if r["decision"] == "rejected")
        self.summary_label.configure(
            text=f"{len(self.results)} шт · ✓ {len(self.results) - n_rej} · "
                 f"✗ {n_rej}")
        if self.results:
            self._show_preview(self.results[0])

    def _show_preview(self, row: dict):
        vis = RESULTS_DIR / "visualizations" / \
            f"{Path(row['filename']).stem}_mask.jpg"
        if not vis.exists():
            return
        try:
            from PIL import Image
            img = Image.open(vis)
            w = max(self.preview_label.winfo_width() - 28, 220)
            h = max(self.preview_label.winfo_height() - 28, 180)
            scale = min(w / img.width, h / img.height, 1.0)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
            self._photo = ctk.CTkImage(light_image=img, dark_image=img,
                                       size=img.size)
            self.preview_label.configure(image=self._photo, text="")
            rej = row["decision"] == "rejected"
            self.hero_caption.configure(
                text=f"{'✗ отбракован' if rej else '✓ сохранён'}   ·   "
                     f"{row['filename']}   ·   облачность {row['cloud_pct']}%",
                text_color=BAD if rej else GOOD)
        except Exception as e:
            self.preview_label.configure(image=None, text=f"ошибка: {e}")


if __name__ == "__main__":
    app = CloudApp()
    app.mainloop()
