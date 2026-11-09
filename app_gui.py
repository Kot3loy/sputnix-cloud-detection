"""app_gui.py — «Пульт обработки» спутниковых снимков (десктоп, тёмная тема)."""
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
CKPT_CANDIDATES = [Path("outputs/checkpoints/best_model.pth"),
                   Path("model_trained/best_model.pth")]


def find_ckpt():
    for p in CKPT_CANDIDATES:
        if p.exists():
            return p
    return None
RESULTS_DIR = Path("outputs/results")


def _pick_font(*candidates: str) -> str:
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
F = lambda s, w="normal": (FONT, s, w)


class CloudApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Спутникс · Обработка облачности")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        win_w, win_h = min(1320, sw - 60), min(840, sh - 80)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(min(1150, sw - 40), min(700, sh - 60))
        self.configure(fg_color=BG)
        self._dark_titlebar()
        if Path("assets/sputnix_icon.ico").exists():
            try:
                self.iconbitmap("assets/sputnix_icon.ico")
            except Exception:
                pass

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_q: queue.Queue = queue.Queue()
        self.results: list[dict] = []
        self._photo = None
        self.current_row = None

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

    def _build_rail(self):
        rail = ctk.CTkFrame(self, width=236, corner_radius=0, fg_color=RAIL)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)

        logo_path = Path("assets/sputnix_lockup_white.png")
        if logo_path.exists():
            from PIL import Image
            img = Image.open(logo_path)
            scale = 190 / img.width
            self._logo = ctk.CTkImage(light_image=img, dark_image=img,
                                      size=(190, int(img.height * scale)))
            ctk.CTkLabel(rail, image=self._logo, text=""
                         ).pack(anchor="w", padx=22, pady=(26, 0))
        else:
            head = ctk.CTkFrame(rail, fg_color="transparent")
            head.pack(fill="x", padx=22, pady=(30, 0))
            ctk.CTkLabel(head, text="☁", font=("Segoe UI Emoji", 30),
                         text_color=ACCENT).pack(side="left")
            ctk.CTkLabel(head, text=" СПУТНИКС", font=F(21, "bold"),
                         text_color=TEXT).pack(side="left")
        ctk.CTkLabel(rail, text="пульт распознавания\nоблачности",
                     font=F(12), text_color=MUTED, justify="left"
                     ).pack(anchor="w", padx=24, pady=(0, 4))

        ctk.CTkFrame(rail, height=1, fg_color=LINE).pack(
            fill="x", padx=22, pady=18)

        ctk.CTkLabel(rail, text="Состояние системы", font=F(12, "bold"),
                     text_color=MUTED, anchor="w").pack(fill="x", padx=24)
        self.chips_box = ctk.CTkFrame(rail, fg_color="transparent")
        self.chips_box.pack(fill="x", padx=18, pady=(8, 0))

        ctk.CTkLabel(rail, text="Формат отчёта — ТЗ:\n«Сохранён: файл — N%.»\n"
                                "«Удалён: файл — N%.»",
                     font=F(11), text_color="#5A6478", justify="left",
                     wraplength=190).pack(anchor="w", padx=24, pady=(18, 0))

        bottom = ctk.CTkFrame(rail, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=14, pady=14)
        for icon, text, path in [("≡", "Отчёты", RESULTS_DIR),
                                 ("✓", "Сохранённые", RESULTS_DIR / "accepted"),
                                 ("✗", "Отбракованные", RESULTS_DIR / "rejected")]:
            ctk.CTkButton(bottom, text=f"{icon}   {text}", height=38,
                          corner_radius=10, fg_color="transparent",
                          text_color=MUTED, hover_color=CARD_HI, anchor="w",
                          font=F(13),
                          command=lambda p=path: self._open(p)
                          ).pack(fill="x", pady=2)

    def _chip(self, parent, text: str, color: str) -> ctk.CTkFrame:
        chip = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=LINE)
        chip.pack(fill="x", pady=3)
        inner = ctk.CTkFrame(chip, fg_color="transparent")
        inner.pack(anchor="w", padx=10, pady=6)
        ctk.CTkLabel(inner, text="●", font=("Segoe UI", 11), text_color=color
                     ).pack(side="left")
        ctk.CTkLabel(inner, text="  " + text, font=F(12), text_color=TEXT
                     ).pack(side="left")
        return chip

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(side="left", fill="both", expand=True, padx=22, pady=18)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text="Обработка снимков", font=F(28, "bold"),
                     text_color=TEXT, anchor="w").pack(side="left")
        ctk.CTkLabel(head, text="маска облаков · процент облачности · "
                                "отбраковка по порогу ТЗ",
                     font=F(13), text_color=MUTED).pack(side="right", pady=(12, 0))

        hero = ctk.CTkFrame(main, fg_color="transparent")
        hero.pack(fill="x", pady=(16, 12))
        hero.grid_columnconfigure(0, weight=11)
        hero.grid_columnconfigure(1, weight=9)
        hero.grid_rowconfigure(0, weight=1)

        self._build_hero(hero)
        self._build_panel(hero)

        low = ctk.CTkFrame(main, fg_color="transparent")
        low.pack(fill="both", expand=True)
        low.grid_columnconfigure(0, weight=1)
        low.grid_columnconfigure(1, weight=1)
        low.grid_rowconfigure(0, weight=1)

        log_card = ctk.CTkFrame(low, fg_color=CARD, corner_radius=16)
        log_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ctk.CTkLabel(log_card, text="Сообщение оператору", font=F(14, "bold"),
                     text_color=TEXT, anchor="w").pack(fill="x", padx=18,
                                                       pady=(12, 2))
        self.log_box = ctk.CTkTextbox(log_card, font=("Consolas", 12),
                                      fg_color=FIELD, text_color="#79E0A5",
                                      corner_radius=10, wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        self.log_box.insert("0.0", "Ожидание запуска обработки…\n")
        self.log_box.configure(state="disabled")

        res_card = ctk.CTkFrame(low, fg_color=CARD, corner_radius=16)
        res_card.grid(row=0, column=1, sticky="nsew")
        rhead = ctk.CTkFrame(res_card, fg_color="transparent")
        rhead.pack(fill="x", padx=18, pady=(12, 2))
        ctk.CTkLabel(rhead, text="Результаты", font=F(14, "bold"),
                     text_color=TEXT).pack(side="left")
        self.summary_label = ctk.CTkLabel(rhead, text="", font=F(12),
                                          text_color=MUTED)
        self.summary_label.pack(side="right")
        self.res_list = ctk.CTkScrollableFrame(res_card, fg_color="transparent")
        self.res_list.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        ctk.CTkLabel(self.res_list, text="здесь появятся обработанные снимки",
                     font=F(12), text_color="#5A6478").pack(pady=24)

        self.progress = ctk.CTkProgressBar(main, height=6, corner_radius=3,
                                           progress_color=ACCENT,
                                           fg_color=LINE)
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(12, 0))

    def _build_hero(self, hero):
        card = ctk.CTkFrame(hero, fg_color=CARD, corner_radius=18)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ctk.CTkLabel(card, text="Контроль маски", font=F(14, "bold"),
                     text_color=TEXT, anchor="w").pack(fill="x", padx=18,
                                                       pady=(12, 0))
        self.preview_label = ctk.CTkLabel(
            card, text="обработайте снимки —\nздесь появится маска облаков",
            font=F(13), text_color="#5A6478")
        self.preview_label.pack(fill="both", expand=True, padx=16, pady=12)
        self.hero_caption = ctk.CTkLabel(card, text="—", font=F(12),
                                         text_color=MUTED, anchor="w")
        self.hero_caption.pack(fill="x", padx=18, pady=(0, 12))

    def _build_panel(self, hero):
        card = ctk.CTkFrame(hero, fg_color=CARD, corner_radius=18)
        card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(card, text="Папка со снимками", font=F(13),
                     text_color=MUTED, anchor="w").pack(fill="x", padx=18,
                                                        pady=(14, 2))
        frow = ctk.CTkFrame(card, fg_color="transparent")
        frow.pack(fill="x", padx=16)
        self.folder_var = ctk.StringVar(value=str(Path("demo/demo_input")))
        self.folder_entry = ctk.CTkEntry(frow, textvariable=self.folder_var,
                                         height=42, corner_radius=10,
                                         fg_color=FIELD, border_color=LINE,
                                         font=F(12))
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(frow, text="Обзор", width=86, height=42, corner_radius=10,
                      fg_color=CARD_HI, hover_color="#26314A", text_color=TEXT,
                      font=F(13, "bold"), command=self._browse).pack(side="left")

        ctk.CTkLabel(card, text="Порог отбраковки  (облачность выше — в утиль)",
                     font=F(13), text_color=MUTED, anchor="w"
                     ).pack(fill="x", padx=18, pady=(16, 0))
        srow = ctk.CTkFrame(card, fg_color="transparent")
        srow.pack(fill="x", padx=18, pady=(2, 0))
        self.thr_value = ctk.StringVar(value="75%")
        ctk.CTkLabel(srow, textvariable=self.thr_value, font=F(26, "bold"),
                     text_color=ACCENT, width=96).pack(side="left")
        self.thr_slider = ctk.CTkSlider(srow, from_=30, to=95,
                                        button_color=ACCENT,
                                        button_hover_color=ACCENT_DK,
                                        progress_color=ACCENT,
                                        fg_color=LINE, height=22,
                                        command=self._on_slider)
        self.thr_slider.set(75)
        self.thr_slider.pack(side="left", fill="x", expand=True, padx=(6, 4),
                             pady=10)

        drow = ctk.CTkFrame(card, fg_color="transparent")
        drow.pack(fill="x", padx=18, pady=(4, 0))
        ctk.CTkLabel(drow, text="Удалять безвозвратно (ТЗ)", font=F(13),
                     text_color=TEXT).pack(side="left")
        self.hard_var = ctk.BooleanVar(value=False)
        self.hard_switch = ctk.CTkSwitch(drow, text="", variable=self.hard_var,
                                         width=50, progress_color=BAD,
                                         button_color="#DDE3EC",
                                         command=self._on_switch)
        self.hard_switch.pack(side="right")
        self.hard_hint = ctk.CTkLabel(card, text="по умолчанию снимки "
                                      "перемещаются в rejected/ — данные "
                                      "не теряются", font=F(11),
                                      text_color="#5A6478", anchor="w",
                                      wraplength=380, justify="left")
        self.hard_hint.pack(fill="x", padx=18, pady=(2, 0))

        self.run_btn = ctk.CTkButton(card, text="Обработать снимки", height=54,
                                     corner_radius=27, fg_color=ACCENT,
                                     hover_color=ACCENT_DK, text_color="#06101F",
                                     font=F(17, "bold"), command=self._start)
        self.run_btn.pack(fill="x", padx=16, pady=(18, 16))

    def _on_slider(self, value):
        self.thr_value.set(f"{int(value)}%")

    def _on_switch(self):
        self.hard_hint.configure(
            text="внимание: файлы будут удаляться безвозвратно — как в ТЗ"
            if self.hard_var.get() else
            "по умолчанию снимки перемещаются в rejected/ — данные не теряются",
            text_color=BAD if self.hard_var.get() else "#5A6478")

    def _browse(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Папка со спутниковыми снимками")
        if folder:
            self.folder_var.set(folder)

    def _open(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except Exception as e:
            self.log(f"[!] не удалось открыть: {e}")

    def _refresh_status(self):
        gpu = self.device.type == "cuda"
        self._chip(self.chips_box,
                   f"GPU · {torch.cuda.get_device_name(0)}" if gpu
                   else "CPU (медленнее)", GOOD if gpu else WARN)
        ckpt = find_ckpt()
        ok = ckpt is not None
        if ok:
            import torch as _t
            ck = _t.load(ckpt, map_location="cpu", weights_only=False)
            self._chip(self.chips_box,
                       f"модель · готова (IoU {ck.get('val_iou', 0):.4f})",
                       GOOD)
        else:
            self._chip(self.chips_box, "модель · не найдена", BAD)
        self._chip(self.chips_box,
                   "демо-данные · есть" if Path("demo_input").exists()
                   else "демо-данные · нет", GOOD if ok else WARN)

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

    def _start(self):
        folder = Path(self.folder_var.get())
        if not folder.exists():
            self.log(f"[!] Папка не найдена: {folder}")
            return
        if find_ckpt() is None:
            self.log("[!] Модель не найдена — обучите её: train.py")
            return
        thr = float(self.thr_slider.get())
        self.run_btn.configure(state="disabled", text="Обработка…")
        for w in self.res_list.winfo_children():
            w.destroy()
        self.results = []
        n = len([p for p in folder.iterdir()
                 if p.is_file() and p.suffix in IMAGE_EXTS])
        self.progress.set(0)
        self.log("=" * 62)
        self.log(f"Обработка: {folder}  ({n} снимков)")
        self.log(f"Порог > {thr:.0f}%  |  отбраковка: "
                 f"{'УДАЛЕНИЕ' if self.hard_var.get() else 'в rejected/'}")
        self.log("-" * 62)
        threading.Thread(target=self._worker, args=(folder, thr),
                         daemon=True).start()

    def _worker(self, folder: Path, thr: float):
        try:
            if self.model is None:
                self.log("Загрузка модели…")
                ckpt = torch.load(find_ckpt(), map_location=self.device,
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
                    mask = predict_mask(self.model, adaptive_preprocess(image),
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
            self.log("-" * 62)
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

    def _fill_results(self):
        for w in self.res_list.winfo_children():
            w.destroy()
        for r in self.results:
            rej = r["decision"] == "rejected"
            color = BAD if rej else GOOD
            row = ctk.CTkFrame(self.res_list, fg_color=FIELD, corner_radius=10)
            row.pack(fill="x", pady=3)
            ctk.CTkFrame(row, width=4, corner_radius=2, fg_color=color
                         ).pack(side="left", fill="y", padx=(10, 0), pady=10)
            name_col = ctk.CTkFrame(row, fg_color="transparent")
            name_col.pack(side="left", fill="x", expand=True, padx=(12, 4),
                          pady=8)
            ctk.CTkLabel(name_col, text=r["filename"], anchor="w", font=F(12.5),
                         text_color=TEXT).pack(anchor="w")
            ctk.CTkLabel(name_col,
                         text="отбракован — облачность выше порога" if rej
                         else "сохранён",
                         font=F(11), text_color=MUTED, anchor="w").pack(
                anchor="w")
            badge = ctk.CTkFrame(row, fg_color=color, corner_radius=12)
            badge.pack(side="right", padx=12, pady=10)
            ctk.CTkLabel(badge, text=f"{'✗' if rej else '✓'} "
                         f"{r['cloud_pct']:.1f}%",
                         font=F(12, "bold"), text_color="#08101C"
                         ).pack(padx=10, pady=3)
            for w in (row, name_col):
                w.bind("<Button-1>", lambda _e, r=r: self._show_preview(r))
            for ch in (*row.winfo_children(), *name_col.winfo_children()):
                ch.bind("<Button-1>", lambda _e, r=r: self._show_preview(r))
        n_rej = sum(1 for r in self.results if r["decision"] == "rejected")
        self.summary_label.configure(
            text=f"{len(self.results)} снимков · ✓ {len(self.results)-n_rej}"
                 f" · ✗ {n_rej}")
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
            w = max(self.preview_label.winfo_width() - 32, 220)
            h = max(self.preview_label.winfo_height() - 32, 180)
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
