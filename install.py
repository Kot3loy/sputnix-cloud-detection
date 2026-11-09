"""Установщик проекта: создаёт окружение и ставит зависимости."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
VENV = ROOT / ".venv"
REQ = ["torch", "cv2", "customtkinter"]


def venv_python():
    return VENV / "Scripts" / "python.exe"


def imports_ok():
    if not venv_python().exists():
        return False
    try:
        r = subprocess.run([str(venv_python()), "-c",
                            "import " + ", ".join(REQ)],
                           capture_output=True)
        return r.returncode == 0
    except OSError:
        return False


def pause():
    try:
        input("Enter — выход...")
    except EOFError:
        pass


def main():
    if shutil.which("python") is None and shutil.which("py") is None:
        print("[X] Python не найден. Установите с python.org (галочка "
              '"Add to PATH") и запустите снова.')
        pause()
        return 1

    print("=" * 50)
    print("  Установка «Спутникс · Облачность»")
    print("=" * 50)

    if imports_ok():
        print("Всё уже установлено — можно запускать ЗАПУСК_ПРОГРАММЫ.bat")
        pause()
        return 0

    if not venv_python().exists():
        print("Создаю виртуальное окружение...")
        r = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
        if r.returncode != 0:
            print("[X] Не удалось создать окружение.")
            pause()
            return 1

    if imports_ok():
        print("Библиотеки уже на месте.")
        pause()
        return 0

    gpu = shutil.which("nvidia-smi") is not None
    hint = "обнаружена — можно выбрать GPU" if gpu else "не обнаружена — выбирайте CPU"
    ans = input(f"\nВидеокарта NVIDIA {hint}.\n"
                "Ставить GPU-версию PyTorch (~3 ГБ, быстрее)? [y/N]: ")
    use_gpu = ans.strip().lower() in ("y", "yes", "д", "да")

    if use_gpu:
        print("\nСтавлю PyTorch для GPU (10–20 минут)...")
        r = subprocess.run([str(venv_python()), "-m", "pip", "install",
                            "torch", "torchvision",
                            "--index-url",
                            "https://download.pytorch.org/whl/cu128"])
    else:
        print("\nСтавлю PyTorch для CPU (~300 МБ)...")
        r = subprocess.run([str(venv_python()), "-m", "pip", "install",
                            "torch", "torchvision"])
    if r.returncode != 0:
        print("[X] PyTorch не установился. Проверьте интернет и повторите.")
        pause()
        return 1

    print("Ставлю остальные библиотеки...")
    r = subprocess.run([str(venv_python()), "-m", "pip", "install",
                        "-r", "requirements.txt"])
    if r.returncode != 0:
        print("[X] Библиотеки не установились.")
        pause()
        return 1

    if imports_ok():
        print("\n" + "=" * 50)
        print("  Готово! Запускайте ЗАПУСК_ПРОГРАММЫ.bat")
        print("=" * 50)
        pause()
        return 0

    print("[X] Проверка не прошла — запустите установку заново.")
    pause()
    return 1


if __name__ == "__main__":
    sys.exit(main())
