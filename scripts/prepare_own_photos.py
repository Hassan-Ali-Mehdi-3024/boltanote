"""Prepare our own photos for the repo: upright, longest side 1024 px, all metadata (EXIF, GPS) removed,
renamed <value>_<nn>.jpg.

    python scripts/prepare_own_photos.py <folder with 10/ 20/ ... 5000/ none/> data/own_photos
"""
import os
import sys

from PIL import Image, ImageOps

try:
    import pillow_heif  # iPhone HEIC photos, if the package is installed

    pillow_heif.register_heif_opener()
except ImportError:
    pass

VALUES = ["10", "20", "50", "100", "500", "1000", "5000", "none"]
src, dst = sys.argv[1], sys.argv[2]
total = 0
for v in VALUES:
    d = os.path.join(src, v)
    if not os.path.isdir(d):
        continue
    os.makedirs(os.path.join(dst, v), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".webp")):
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(os.path.join(d, f))).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print("skip", f, e)
            continue
        im.thumbnail((1024, 1024), Image.LANCZOS)
        n += 1
        clean = Image.new("RGB", im.size)
        clean.paste(im)  # a fresh image carries no metadata
        clean.save(os.path.join(dst, v, f"{v}_{n:02d}.jpg"), quality=90)
    print(v, n)
    total += n
print("total", total)
