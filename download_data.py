"""download_data.py — загрузка данных для кейса."""
from __future__ import annotations

import argparse
import base64
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SPUTNIX_TOKEN = "HbkKc34batcjkSX"
SPUTNIX_WEBDAV = f"https://disk.sputnix.ru/public.php/webdav/"
CLOUD38_FILES = [
    "train-00000-of-00002-ed5a0c46ae8399b6.parquet",
    "train-00001-of-00002-2ab415203996145c.parquet",
]
CLOUD38_BASE = ("https://huggingface.co/datasets/jaygala223/"
                "38-cloud-train-only-v1/resolve/main/data/")


def _auth_header() -> dict:
    tok = base64.b64encode(f"{SPUTNIX_TOKEN}:".encode()).decode()
    return {"Authorization": f"Basic {tok}"}


def sputnix_list() -> list[str]:
    """Список JPG-файлов шеринга через PROPFIND (глубина 1)."""
    req = urllib.request.Request(
        SPUTNIX_WEBDAV, method="PROPFIND", headers=_auth_header() | {"Depth": "1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        xml_data = r.read()
    root = ET.fromstring(xml_data)
    ns = {"d": "DAV:"}
    names = []
    for resp in root.findall("d:response", ns):
        href = resp.find("d:href", ns).text
        if href.lower().endswith(".jpg"):
            names.append(href.split("/")[-1])
    return names


def download_sputnix(out_dir: Path, limit: int, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sputnix] Запрашиваю список файлов шеринга (~4213 снимков)...")
    names = sputnix_list()
    print(f"[sputnix] В шеринге {len(names)} JPEG-снимков; скачиваю {limit} (seed={seed})")
    rng = random.Random(seed)
    selected = rng.sample(names, min(limit, len(names)))
    ok = 0
    for i, name in enumerate(selected, 1):
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            ok += 1
            continue
        url = SPUTNIX_WEBDAV + urllib.parse.quote(name)
        req = urllib.request.Request(url, headers=_auth_header())
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dst.write_bytes(r.read())
            ok += 1
            print(f"  [{i}/{len(selected)}] {name} "
                  f"({dst.stat().st_size/1e6:.1f} МБ)")
        except Exception as e:
            print(f"  [{i}/{len(selected)}] ОШИБКА {name}: {e}")
    print(f"[sputnix] Готово: {ok}/{len(selected)} снимков в {out_dir}")


def download_38cloud(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in CLOUD38_FILES:
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 1e8:
            print(f"[38cloud] {name} уже скачан, пропускаю")
            continue
        url = CLOUD38_BASE + name
        print(f"[38cloud] Скачиваю {name} (~377 МБ)...")
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"[38cloud] OK: {dst.stat().st_size/1e6:.1f} МБ")
        except Exception as e:
            print(f"[38cloud] ОШИБКА: {e}")
    print(f"[38cloud] Готово: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    p1 = sub.add_parser("sputnix", help="снимки Спутникс (WebDAV)")
    p1.add_argument("--out", default="data/raw/sputnix/samples")
    p1.add_argument("--limit", type=int, default=12)
    p2 = sub.add_parser("38cloud", help="датасет 38-Cloud (HuggingFace)")
    p2.add_argument("--out", default="data/raw/38cloud")
    args = parser.parse_args()

    if args.source == "sputnix":
        download_sputnix(Path(args.out), args.limit)
    else:
        download_38cloud(Path(args.out))
