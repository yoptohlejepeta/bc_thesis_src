"""
Skript pro vytvoření bounding boxů.

Tento skript načte segmentované obrázky z adresáře vytvoří orientované a neorientované bounding boxy.
Výsledky se uloží do specifikovaných adresářů.
Do JSON souborů se uloží informace o bounding boxech (je používáné v `gen_images.py`).
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import json

watershed_dir = Path("results_data/watershed")
obb_output_dir = Path("results_data/watershed_with_obbs")
bbox_output_dir = Path("results_data/watershed_with_bbox")

obb_output_dir.mkdir(exist_ok=True)
bbox_output_dir.mkdir(exist_ok=True)

all_aabb_data = {}
all_obb_data = {}

for npy_file in watershed_dir.glob("*_segmented_watershed.npy"):
    pic_num = npy_file.stem.split("_segmented_watershed")[0]

    watershed_img = np.load(npy_file)

    original_image_path = Path(f"images/{pic_num}.png")
    if original_image_path.exists():
        original_image = plt.imread(original_image_path)
    else:
        print(f"Original image {original_image_path} not found. Skipping...")
        continue

    watershed_img = clear_border(watershed_img)
    regions = regionprops(label(watershed_img))

    aabb_data = []
    obb_data = []

    fig_obb = plt.figure(figsize=(9, 6))
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    for region in regions:
        coords = region.coords[:, ::-1]
        rect = cv2.minAreaRect(coords.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        color = np.random.rand(
            3,
        )

        plt.plot(
            *zip(*np.vstack([box, box[0]]), strict=False), color=color, linewidth=1
        )

        obb_data.append(box.tolist())

    obb_output_path = obb_output_dir / f"{pic_num}_with_obbs.png"
    plt.savefig(obb_output_path, bbox_inches="tight", dpi=300, pad_inches=0)
    plt.close(fig_obb)
    print(f"Saved OBB image: {obb_output_path}")

    fig_bbox = plt.figure(figsize=(9, 6))
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = np.array([[minc, minr], [maxc, minr], [maxc, maxr], [minc, maxr]])

        color = np.random.rand(
            3,
        )

        plt.plot(
            *zip(*np.vstack([rect, rect[0]]), strict=False), color=color, linewidth=1
        )

        aabb_data.append([minc, minr, maxc, maxr])

    bbox_output_path = bbox_output_dir / f"{pic_num}_with_bbox.png"
    plt.savefig(bbox_output_path, bbox_inches="tight", dpi=300, pad_inches=0)
    plt.close(fig_bbox)
    print(f"Saved BBOX image: {bbox_output_path}")

    all_aabb_data[pic_num] = aabb_data
    all_obb_data[pic_num] = obb_data

aabb_json_path = bbox_output_dir / "all_aabb.json"
with open(aabb_json_path, "w") as f:
    json.dump(all_aabb_data, f, indent=2)
print(f"Saved AABB JSON file: {aabb_json_path}")

obb_json_path = obb_output_dir / "all_obb.json"
with open(obb_json_path, "w") as f:
    json.dump(all_obb_data, f, indent=2)
print(f"Saved OBB JSON file: {obb_json_path}")

print("Processing complete.")
